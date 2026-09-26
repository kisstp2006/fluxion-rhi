// SPDX-License-Identifier: BSD-2-Clause

//! A device: one backend, the resources made on it, and the frame being
//! recorded.
//!
//! ```zig
//! var device = try rhi.Device.init(gpa, .{ .gl = window.hooks() });
//! defer device.deinit();
//!
//! const surface = try device.createSurface(.{});
//! const pipeline = try device.createPipeline(.{ ... });
//!
//! while (running) {
//!     const cmd = device.begin();
//!     try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
//!     try cmd.setPipeline(pipeline);
//!     try cmd.setVertexBuffer(0, quad, 0);
//!     try cmd.draw(.{ .vertex_count = 4 });
//!     try cmd.endPass();
//!     try device.submit();
//!     try device.present(surface);
//! }
//! ```
//!
//! **Handles, not pointers.** Everything `create` returns is eight bytes with
//! a generation in them. A destroyed handle is `error.InvalidHandle`, from
//! `Device` and before the backend sees it.
//!
//! **Validation is here, once.** A draw outside a pass, a vertex buffer bound
//! where an index buffer was wanted, a texture that cannot be rendered to:
//! `submit` refuses these with `error.InvalidArgument` and `diagnostics` says
//! which command, so a backend can assume a list that makes sense and a
//! program gets the same answer on every backend.
//!
//! **One thread.** A device is used from the thread that made it, which on
//! OpenGL is the thread the context is current on and on the others is
//! merely the simplest rule.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const math = @import("fluxion_math");

const types = @import("types.zig");
const resources = @import("resources.zig");
const commands = @import("commands.zig");
const backend = @import("backend.zig");

/// Whether this is a build for a browser. The architecture decides it, as it
/// does in `fluxion-webgl`: a wasm module reaches WebGL through its imports
/// whatever the operating system field says.
const is_wasm = switch (builtin.target.cpu.arch) {
    .wasm32, .wasm64 => true,
    else => false,
};

const none_backend = @import("backend/none.zig");
/// Everywhere but the web, where there is no OpenGL to load - only WebGL,
/// which is imported rather than found.
const gl_backend = if (!is_wasm) @import("backend/gl.zig") else void;
const d3d11_backend = if (builtin.os.tag == .windows) @import("backend/d3d11.zig") else void;
const d3d12_backend = if (builtin.os.tag == .windows) @import("backend/d3d12.zig") else void;
/// Vulkan, wherever a loader can be; a machine without one answers
/// `error.NoDevice` when it is opened.
const vulkan_backend = if (!is_wasm) @import("backend/vulkan.zig") else void;
/// On the web, and under test everywhere: off wasm, `fluxion-webgl` answers
/// from its stub, which is what lets the backend's bookkeeping be checked on
/// a machine with no browser. A desktop program that asks for it outside a
/// test is refused, because a stub draws nothing.
const webgl_backend = if (is_wasm or builtin.is_test) @import("backend/webgl.zig") else void;

const Device = @This();

pub const Error = types.Error;

gpa: Allocator,
impl: backend.Impl,
vtable: *const backend.Vtable,
tag: types.Backend,
/// From the opener; borrowed.
name: []const u8,
clip_space: math.Clip,
/// What the backend said it can do, asked once. Every request is checked
/// against it here, before a backend is called.
capabilities: types.Caps,

buffers: resources.BufferTable = .empty,
textures: resources.TextureTable = .empty,
samplers: resources.SamplerTable = .empty,
shaders: resources.ShaderTable = .empty,
pipelines: resources.PipelineTable = .empty,
surfaces: resources.SurfaceTable = .empty,

list: commands.CommandList,

/// The last thing worth reading: a shader log, or why a submit was refused.
log_buffer: [8192]u8 = undefined,
log_len: usize = 0,

// -------------------------------------------------------------------------
// Opening and closing
// -------------------------------------------------------------------------

/// Which backends this build could open. `.none` is always among them.
///
/// `.vulkan` and `.d3d12` are experimental - see `backend/vulkan.zig` and
/// `backend/d3d12.zig` - and neither is `.auto`'s pick: `init` resolves
/// `.auto` to `.d3d11`/`.gl` on Windows, so a program gets either only by
/// naming it.
pub fn available() []const types.Backend {
    return if (is_wasm)
        &.{ .webgl, .none }
    else if (builtin.os.tag == .windows)
        &.{ .d3d11, .d3d12, .vulkan, .gl, .none }
    else
        &.{ .vulkan, .gl, .none };
}

/// Whether this machine has a Vulkan loader that Fluxion can open right now.
/// `Select.vulkan` no longer needs asking separately - it is in `available`
/// - this stays for a caller that wants to know before choosing `.auto`
/// would (`.auto` itself still prefers `.d3d11`/`.gl`; see `init`).
pub fn vulkanLoaderAvailable() bool {
    return if (is_wasm) false else vulkan_backend.loaderAvailable();
}

/// The live `VkInstance` and its `vkGetInstanceProcAddr`, as integers, of a
/// device opened on `.vulkan` - for a caller that wants a `VkSurfaceKHR` made
/// from this device's own instance, through `fluxion-platform`'s
/// `Window.createVulkanSurface` or the equivalent, before calling
/// `createSurface` with `SurfaceDesc.vulkan_surface`. Null on every other
/// backend, and null on `.vulkan` before the backend has a real instance to
/// hand back.
pub fn vulkanInstanceHandles(self: *Device) ?backend.VulkanInstanceHandles {
    return if (is_wasm)
        null
    else if (self.tag != .vulkan)
        null
    else
        vulkan_backend.instanceHandles(self.impl);
}

/// How to open one of the backends this build brings, or null if it does not
/// bring it: `.other` is whatever a caller supplies, and the rest depend on
/// the target.
///
/// A program that keeps its backends in a registry registers these by `name`.
pub fn opener(which: types.Backend) ?backend.Opener {
    return switch (which) {
        .none => .{ .name = "none", .tag = .none, .clip = which.clip(), .open = none_backend.open },
        .gl => if (!is_wasm) .{ .name = "gl", .tag = .gl, .clip = which.clip(), .open = gl_backend.open } else null,
        .d3d11 => if (builtin.os.tag == .windows) .{ .name = "d3d11", .tag = .d3d11, .clip = which.clip(), .open = d3d11_backend.open } else null,
        .d3d12 => if (builtin.os.tag == .windows) .{ .name = "d3d12", .tag = .d3d12, .clip = which.clip(), .open = d3d12_backend.open } else null,
        .webgl => if (is_wasm or builtin.is_test) .{ .name = "webgl", .tag = .webgl, .clip = which.clip(), .open = webgl_backend.open } else null,
        .vulkan => if (!is_wasm) .{ .name = "vulkan", .tag = .vulkan, .clip = which.clip(), .open = vulkan_backend.open } else null,
        .other => null,
    };
}

/// Opens a device on the backend an opener describes: one of `opener`'s, or one
/// the caller made. `desc.backend` is not looked at, the opener has already
/// chosen. The opener's name is borrowed until `deinit`.
pub fn initWith(gpa: Allocator, desc: types.DeviceDesc, how: backend.Opener) Error!Device {
    const opened = try how.open(gpa, desc);
    return .{
        .gpa = gpa,
        .impl = opened[0],
        .vtable = opened[1],
        .tag = how.tag,
        .name = how.name,
        .clip_space = how.clip,
        .capabilities = opened[1].caps(opened[0]),
        .list = .init(gpa),
    };
}

pub fn init(gpa: Allocator, desc: types.DeviceDesc) Error!Device {
    const chosen: types.Backend = switch (desc.backend) {
        .none => .none,
        .gl => .gl,
        .d3d11 => .d3d11,
        .d3d12 => .d3d12,
        .webgl => .webgl,
        .vulkan => .vulkan,
        .auto => if (desc.gl != null)
            .gl
        else if (builtin.os.tag == .windows)
            .d3d11
        else if (is_wasm)
            .webgl
        else
            return error.Unsupported,
    };

    return initWith(gpa, desc, opener(chosen) orelse return error.Unsupported);
}

/// Destroy everything still alive, then the backend.
pub fn deinit(self: *Device) void {
    // Pipelines before shaders, everything before surfaces: the order a
    // backend would want if it cared, and none of them do.
    var pipelines = self.pipelines.iterator();
    while (pipelines.next()) |entry| self.vtable.destroyPipeline(self.impl, entry.value.native);
    var shaders = self.shaders.iterator();
    while (shaders.next()) |entry| self.vtable.destroyShader(self.impl, entry.value.native);
    var samplers = self.samplers.iterator();
    while (samplers.next()) |entry| self.vtable.destroySampler(self.impl, entry.value.native);
    var textures = self.textures.iterator();
    while (textures.next()) |entry| self.vtable.destroyTexture(self.impl, entry.value.native);
    var buffers = self.buffers.iterator();
    while (buffers.next()) |entry| self.vtable.destroyBuffer(self.impl, entry.value.native);
    var surfaces = self.surfaces.iterator();
    while (surfaces.next()) |entry| self.vtable.destroySurface(self.impl, entry.value.native);

    self.pipelines.deinit(self.gpa);
    self.shaders.deinit(self.gpa);
    self.samplers.deinit(self.gpa);
    self.textures.deinit(self.gpa);
    self.buffers.deinit(self.gpa);
    self.surfaces.deinit(self.gpa);
    self.list.deinit();

    self.vtable.deinit(self.impl);
    self.* = undefined;
}

/// Which of the built-in backends this is, or `.other` for one that came from
/// `initWith`. `info().name` says which in either case.
pub fn backendTag(self: *const Device) types.Backend {
    return self.tag;
}

pub fn info(self: *const Device) types.Info {
    var answer = self.vtable.info(self.impl);
    answer.backend = self.tag;
    answer.name = self.name;
    return answer;
}

/// Which clip space a projection for this device is built for. Hand it to
/// `fluxion-math`'s `perspective` and `orthographic`.
pub fn clip(self: *const Device) math.Clip {
    return self.clip_space;
}

/// What this device can do: its limits, its features, and what each format
/// can be used for. Read it to choose - a compressed format for a phone, a
/// multisample count for a setting - instead of trying and failing.
pub fn caps(self: *const Device) *const types.Caps {
    return &self.capabilities;
}

/// The last shader log or validation message. Empty when nothing went wrong.
pub fn diagnostics(self: *const Device) []const u8 {
    return self.log_buffer[0..self.log_len];
}

fn logWriter(self: *Device) Io.Writer {
    self.log_len = 0;
    return .fixed(&self.log_buffer);
}

fn keepLog(self: *Device, w: *const Io.Writer) void {
    self.log_len = w.buffered().len;
}

// -------------------------------------------------------------------------
// Resources
// -------------------------------------------------------------------------

pub fn createBuffer(self: *Device, desc: types.BufferDesc) Error!types.Buffer {
    if (desc.size == 0) return self.refuse("createBuffer: a buffer of zero bytes");
    if (desc.data) |data| if (data.len > desc.size) return self.refuse("createBuffer: more data than size");
    const native = try self.vtable.createBuffer(self.impl, desc);
    errdefer self.vtable.destroyBuffer(self.impl, native);
    return self.buffers.add(self.gpa, .{
        .native = native,
        .kind = desc.kind,
        .size = desc.size,
        .dynamic = desc.dynamic or desc.kind == .uniform,
    });
}

pub fn destroyBuffer(self: *Device, h: types.Buffer) void {
    if (self.buffers.remove(h)) |entry| self.vtable.destroyBuffer(self.impl, entry.native);
}

/// Replace `bytes.len` bytes at `offset`.
pub fn updateBuffer(self: *Device, h: types.Buffer, offset: usize, bytes: []const u8) Error!void {
    const entry = self.buffers.get(h) orelse return error.InvalidHandle;
    if (offset + bytes.len > entry.size) return self.refuse("updateBuffer: past the end of the buffer");
    try self.vtable.updateBuffer(self.impl, entry.native, offset, bytes);
}

pub fn createTexture(self: *Device, desc: types.TextureDesc) Error!types.Texture {
    const limits = self.capabilities.limits;
    if (desc.width == 0 or desc.height == 0) return self.refuse("createTexture: a texture with no area");
    // Which shapes a format comes in is a fact about the device, and is asked
    // before its sizes: a device with no volumes has no limit worth quoting.
    if (!self.capabilities.formatSupport(desc.format).dimensions.contains(desc.dimension)) {
        return self.unsupported("createTexture: this device cannot make that shape of texture in that format");
    }

    // The shape: which of the size fields mean what, and how big is too big.
    var largest = @max(desc.width, desc.height);
    switch (desc.dimension) {
        .d2 => {
            if (desc.depth_or_layers != 1) return self.refuse("createTexture: a .d2 texture has one layer - .d2_array has more");
            if (largest > limits.max_texture_2d) return self.refuse("createTexture: larger than the device's max_texture_2d");
        },
        .d2_array => {
            if (desc.depth_or_layers == 0) return self.refuse("createTexture: an array of no layers");
            if (largest > limits.max_texture_2d) return self.refuse("createTexture: larger than the device's max_texture_2d");
            if (desc.depth_or_layers > limits.max_texture_layers) return self.refuse("createTexture: more layers than the device's max_texture_layers");
        },
        .cube => {
            if (desc.width != desc.height) return self.refuse("createTexture: the faces of a cube are square");
            if (largest > limits.max_texture_cube) return self.refuse("createTexture: larger than the device's max_texture_cube");
        },
        .d3 => {
            if (desc.depth_or_layers == 0) return self.refuse("createTexture: a volume with no depth");
            largest = @max(largest, desc.depth_or_layers);
            if (largest > limits.max_texture_3d) return self.refuse("createTexture: larger than the device's max_texture_3d");
        },
    }

    const mips = desc.mipCount();
    if (mips > types.fullMipCount(desc.width, desc.height, desc.depth())) return self.refuse("createTexture: more mip levels than the size has");

    // Mistakes first, then what this device happens to lack: the same request
    // is wrong everywhere in the first case and only here in the second.
    if (!desc.usage.sampled and !desc.usage.render_target) return self.refuse("createTexture: a texture that is neither sampled nor a render target");
    if (desc.format.isCompressed() and desc.usage.render_target) return self.refuse("createTexture: a compressed format cannot be drawn into");
    const support = self.capabilities.formatSupport(desc.format);
    if (desc.usage.sampled and !support.sampled) return self.unsupported("createTexture: this device cannot sample that format");
    if (desc.usage.render_target and !support.render_target) return self.unsupported("createTexture: this device cannot draw into that format");
    if (desc.format.isCompressed() and !self.capabilities.features.compressed_partial_blocks) {
        const block = desc.format.info();
        if (desc.width % block.block_width != 0 or desc.height % block.block_height != 0) {
            return self.unsupported("createTexture: this device wants a compressed texture whose size is whole blocks");
        }
    }

    if (desc.samples != 1) {
        if (!desc.usage.render_target) return self.refuse("createTexture: samples above one need usage.render_target");
        if (desc.usage.sampled) return self.refuse("createTexture: a multisampled texture cannot be sampled - resolve it into one that can");
        if (desc.dimension != .d2) return self.refuse("createTexture: only a .d2 texture can be multisampled");
        if (mips != 1) return self.refuse("createTexture: a multisampled texture has one mip level");
        if (desc.data != null) return self.refuse("createTexture: a multisampled texture cannot be filled from memory");
        if (!support.supportsSamples(desc.samples)) return self.unsupported("createTexture: this device cannot multisample that format that many times");
    }

    if (desc.format.isDepth() and desc.data != null) return self.refuse("createTexture: a depth texture cannot be filled from memory");
    if (desc.data) |data| {
        const tight = desc.tightRowPitch();
        const pitch = desc.effectiveRowPitch();
        if (pitch < tight) return self.refuse("createTexture: a row pitch narrower than a row");
        const rows = desc.format.rowCount(desc.height);
        const slices = @as(usize, desc.layers()) * desc.depth();
        const needed = pitch * rows * (slices - 1) + pitch * (rows - 1) + tight;
        if (data.len < needed) return self.refuse("createTexture: fewer bytes than the size and pitch need");
    }

    var resolved = desc;
    resolved.mip_levels = mips;
    const native = try self.vtable.createTexture(self.impl, resolved);
    errdefer self.vtable.destroyTexture(self.impl, native);
    return self.textures.add(self.gpa, .{
        .native = native,
        .dimension = desc.dimension,
        .width = desc.width,
        .height = desc.height,
        .depth_or_layers = switch (desc.dimension) {
            .d2 => 1,
            .cube => 6,
            .d3, .d2_array => desc.depth_or_layers,
        },
        .mip_levels = mips,
        .samples = desc.samples,
        .format = desc.format,
        .usage = desc.usage,
    });
}

pub fn destroyTexture(self: *Device, h: types.Texture) void {
    if (self.textures.remove(h)) |entry| self.vtable.destroyTexture(self.impl, entry.native);
}

/// Replace every texel of mip level zero: every layer, face or slice of it,
/// one after another. `row_pitch` of zero means tightly packed.
pub fn updateTexture(self: *Device, h: types.Texture, bytes: []const u8, row_pitch: usize) Error!void {
    return self.writeTexture(h, .{}, bytes, row_pitch, 0);
}

/// Replace a box of texels in one mip level. A zero size in `region` runs to
/// the end of that level, so `.{ .mip = 3 }` is all of level three.
///
/// The bytes are laid out top row first. `row_pitch` is the distance in bytes
/// from one row to the next and `slice_pitch` from one slice, layer or face
/// to the next; zero is tightly packed for either. Rows of a compressed
/// format are rows of blocks.
pub fn writeTexture(
    self: *Device,
    h: types.Texture,
    region: types.TextureRegion,
    bytes: []const u8,
    row_pitch: usize,
    slice_pitch: usize,
) Error!void {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    if (entry.format.isDepth()) return self.refuse("writeTexture: a depth texture cannot be filled from memory");
    if (entry.samples != 1) return self.refuse("writeTexture: a multisampled texture cannot be filled from memory");

    if (region.mip >= entry.mip_levels) return self.refuse("writeTexture: a mip level the texture does not have");
    const level_width = types.mipExtent(entry.width, region.mip);
    const level_height = types.mipExtent(entry.height, region.mip);
    const level_slices = entry.slices(region.mip);
    if (region.x >= level_width or region.y >= level_height or region.z >= level_slices) return self.refuse("writeTexture: the region starts outside the level");

    var box = region;
    if (box.width == 0) box.width = level_width - region.x;
    if (box.height == 0) box.height = level_height - region.y;
    if (box.depth == 0) box.depth = level_slices - region.z;
    if (box.width > level_width - box.x or box.height > level_height - box.y or box.depth > level_slices - box.z) {
        return self.refuse("writeTexture: the region reaches past the level");
    }

    // A compressed format is written in whole blocks: a box that starts
    // inside one, or stops inside one short of the edge, has no bytes to say.
    if (entry.format.isCompressed()) {
        const row = entry.format.info();
        const ragged = box.x % row.block_width != 0 or box.y % row.block_height != 0 or
            (box.width % row.block_width != 0 and box.x + box.width != level_width) or
            (box.height % row.block_height != 0 and box.y + box.height != level_height);
        if (ragged) return self.refuse("writeTexture: a region of a compressed texture must be whole blocks");
    }

    const tight = entry.format.rowBytes(box.width);
    const pitch = if (row_pitch == 0) tight else row_pitch;
    if (pitch < tight) return self.refuse("writeTexture: a row pitch narrower than a row");
    const rows = entry.format.rowCount(box.height);
    const slice = if (slice_pitch == 0) pitch * rows else slice_pitch;
    if (box.depth > 1 and slice < pitch * rows) return self.refuse("writeTexture: a slice pitch smaller than a slice");
    const needed = slice * (box.depth - 1) + pitch * (rows - 1) + tight;
    if (bytes.len < needed) return self.refuse("writeTexture: fewer bytes than the region needs");

    try self.vtable.writeTexture(self.impl, entry.native, box, bytes, pitch, slice);
}

/// Mip level zero of a texture as RGBA, eight bits a channel, top row first,
/// tightly packed: `width * height * 4` bytes the caller owns. The one way to
/// look at what was drawn without a window, and what every test here does.
pub fn readTexture(self: *Device, h: types.Texture, gpa: Allocator) Error![]u8 {
    return self.readSubresource(h, .{}, gpa);
}

/// The same for any one image of a texture: a mip level of a layer, a face or
/// a slice, at that level's size. Not for a depth or a compressed format, and
/// not for a multisampled texture - resolve it first.
pub fn readSubresource(self: *Device, h: types.Texture, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    if (entry.format.isDepth()) return self.refuse("readTexture: a depth texture cannot be read as colour");
    if (entry.format.isCompressed()) return self.refuse("readTexture: a compressed texture cannot be read as colour");
    if (entry.samples != 1) return self.refuse("readTexture: a multisampled texture cannot be read - resolve it first");
    if (sub.mip >= entry.mip_levels) return self.refuse("readTexture: a mip level the texture does not have");
    if (sub.layer >= entry.slices(sub.mip)) return self.refuse("readTexture: a layer, face or slice the texture does not have");
    return self.vtable.readTexture(self.impl, entry.native, sub, gpa);
}

/// Level zero's width and height. `textureInfo` has the rest.
pub fn textureSize(self: *Device, h: types.Texture) Error!types.Extent {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    return .{ .width = entry.width, .height = entry.height };
}

pub fn textureInfo(self: *Device, h: types.Texture) Error!types.TextureInfo {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    return .{
        .dimension = entry.dimension,
        .width = entry.width,
        .height = entry.height,
        .depth_or_layers = entry.depth_or_layers,
        .mip_levels = entry.mip_levels,
        .samples = entry.samples,
        .format = entry.format,
        .usage = entry.usage,
    };
}

pub fn createSampler(self: *Device, desc: types.SamplerDesc) Error!types.Sampler {
    const features = self.capabilities.features;
    if (desc.max_anisotropy == 0) return self.refuse("createSampler: max_anisotropy of zero - one is off");
    if (desc.lod_min > desc.lod_max) return self.refuse("createSampler: lod_min above lod_max");
    const border = desc.wrap_u == .border or desc.wrap_v == .border or desc.wrap_w == .border;
    if (border and !features.sampler_border) return self.unsupported("createSampler: this device has no border wrap");
    if (desc.lod_bias != 0 and !features.sampler_lod_bias) return self.unsupported("createSampler: this device has no sampler lod bias");

    // More taps than the hardware has is not an error, just the most it has.
    var resolved = desc;
    resolved.max_anisotropy = @intCast(@min(desc.max_anisotropy, @max(1, self.capabilities.limits.max_anisotropy)));

    const native = try self.vtable.createSampler(self.impl, resolved);
    errdefer self.vtable.destroySampler(self.impl, native);
    return self.samplers.add(self.gpa, .{ .native = native });
}

pub fn destroySampler(self: *Device, h: types.Sampler) void {
    if (self.samplers.remove(h)) |entry| self.vtable.destroySampler(self.impl, entry.native);
}

/// Compile and link. On failure the driver's log is in `diagnostics`.
pub fn createShader(self: *Device, desc: types.ShaderDesc) Error!types.Shader {
    var log = self.logWriter();
    defer self.keepLog(&log);
    const native = try self.vtable.createShader(self.impl, desc, &log);
    errdefer self.vtable.destroyShader(self.impl, native);
    return self.shaders.add(self.gpa, .{ .native = native });
}

pub fn destroyShader(self: *Device, h: types.Shader) void {
    if (self.shaders.remove(h)) |entry| self.vtable.destroyShader(self.impl, entry.native);
}

pub fn createPipeline(self: *Device, desc: types.PipelineDesc) Error!types.Pipeline {
    const shader = self.shaders.get(desc.shader) orelse return error.InvalidHandle;
    var slots: u32 = 0;
    for (desc.attributes) |attribute| {
        if (attribute.buffer >= desc.buffers.len) return self.refuse("createPipeline: an attribute reads a buffer slot the pipeline has not got");
        if (attribute.offset + attribute.format.size() > desc.buffers[attribute.buffer].stride) return self.refuse("createPipeline: an attribute reaches past its vertex stride");
        slots = @max(slots, attribute.buffer + 1);
    }
    if (desc.depth_format) |format| if (!format.isDepth()) return self.refuse("createPipeline: depth_format is not a depth format");
    if (desc.color_format) |format| if (format.isDepth()) return self.refuse("createPipeline: color_format is a depth format");
    if (desc.color_format == null and desc.extra_color_formats.len > 0) return self.refuse("createPipeline: extra_color_formats without a color_format");
    if (desc.color_format == null and desc.depth_format == null) return self.refuse("createPipeline: a pipeline that draws into nothing");
    if (desc.samples == 0 or !std.math.isPowerOfTwo(desc.samples)) return self.refuse("createPipeline: samples must be a power of two");

    var log = self.logWriter();
    defer self.keepLog(&log);
    const native = try self.vtable.createPipeline(self.impl, desc, shader.native, &log);
    errdefer self.vtable.destroyPipeline(self.impl, native);
    return self.pipelines.add(self.gpa, .{
        .native = native,
        .topology = desc.topology,
        .buffer_slots = slots,
        .color_format = desc.color_format,
        .depth_format = desc.depth_format,
        .samples = desc.samples,
    });
}

pub fn destroyPipeline(self: *Device, h: types.Pipeline) void {
    if (self.pipelines.remove(h)) |entry| self.vtable.destroyPipeline(self.impl, entry.native);
}

// -------------------------------------------------------------------------
// Surfaces
// -------------------------------------------------------------------------

pub fn createSurface(self: *Device, desc: types.SurfaceDesc) Error!types.Surface {
    const native = try self.vtable.createSurface(self.impl, desc);
    errdefer self.vtable.destroySurface(self.impl, native);
    return self.surfaces.add(self.gpa, .{ .native = native, .present_mode = desc.present_mode });
}

pub fn destroySurface(self: *Device, h: types.Surface) void {
    if (self.surfaces.remove(h)) |entry| self.vtable.destroySurface(self.impl, entry.native);
}

/// Tell the surface the window changed size. Nothing may be drawing.
pub fn resizeSurface(self: *Device, h: types.Surface, width: u32, height: u32) Error!void {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    if (width == 0 or height == 0) return; // minimised; nothing to resize to
    try self.vtable.resizeSurface(self.impl, entry.native, width, height);
}

pub fn surfaceSize(self: *Device, h: types.Surface) Error!types.Extent {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    const size = self.vtable.surfaceSize(self.impl, entry.native);
    return .{ .width = size[0], .height = size[1] };
}

/// How the surface's frames are shown against the display's refresh, from
/// the next `present`. See `types.PresentMode`.
pub fn setPresentMode(self: *Device, h: types.Surface, mode: types.PresentMode) Error!void {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    entry.present_mode = mode;
}

pub fn presentMode(self: *Device, h: types.Surface) Error!types.PresentMode {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    return entry.present_mode;
}

/// Show what the last submit drew into the surface.
pub fn present(self: *Device, h: types.Surface) Error!void {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    try self.vtable.present(self.impl, entry.native, entry.present_mode);
}

// -------------------------------------------------------------------------
// Frames
// -------------------------------------------------------------------------

/// Start recording. The list belongs to the device and is emptied by
/// `submit`; the pointer is good until then.
pub fn begin(self: *Device) *commands.CommandList {
    self.list.reset();
    return &self.list;
}

/// Check the list, hand it to the backend, and empty it - whether or not
/// the backend liked it, so a refused frame does not poison the next.
pub fn submit(self: *Device) Error!void {
    defer self.list.reset();
    try self.validate(self.list.commands());
    try self.vtable.submit(self.impl, self, self.list.commands());
}

/// Everything about a list that is wrong on every backend, found before any
/// backend sees it.
fn validate(self: *Device, list: []const commands.Command) Error!void {
    var in_pass = false;
    var pass_samples: u32 = 1;
    var pipeline: ?*resources.PipelineEntry = null;
    var bound_slots: u32 = 0; // a bit per vertex buffer slot
    var index_bound = false;

    for (list, 0..) |command, at| {
        switch (command) {
            .begin_pass => |pass| {
                if (in_pass) return self.refuseAt(at, "beginPass inside a pass");
                if (pass.color == null and pass.depth == null) return self.refuseAt(at, "beginPass: a pass with nothing to draw into");
                if (pass.color == null and pass.extra_colors.len > 0) return self.refuseAt(at, "beginPass: extra colour attachments without a colour attachment");
                if (pass.extra_colors.len > 0 and pass.color.?.target != .texture) {
                    return self.refuseAt(at, "beginPass: a multi-attachment pass cannot target the surface");
                }
                const colors: usize = pass.extra_colors.len + @as(usize, if (pass.color != null) 1 else 0);
                if (colors > self.capabilities.limits.max_color_attachments) return self.refuseAt(at, "beginPass: more colour attachments than the device's max_color_attachments");

                var shape: PassShape = .{};
                if (pass.color) |color| try self.checkColorAttachment(at, color, &shape);
                for (pass.extra_colors) |extra| {
                    if (extra.target == .surface) return self.refuseAt(at, "beginPass: an extra color attachment cannot target the surface");
                    try self.checkColorAttachment(at, extra, &shape);
                }
                if (pass.depth) |depth| {
                    const texture = self.textures.get(depth.texture) orelse return self.refuseAt(at, "beginPass: the depth texture is not alive");
                    if (!texture.usage.render_target) return self.refuseAt(at, "beginPass: the depth texture was not made with usage.render_target");
                    if (!texture.format.isDepth()) return self.refuseAt(at, "beginPass: the depth attachment is not a depth format");
                    try self.checkSubresource(at, texture, depth.mip_level, depth.layer, &shape);
                }
                in_pass = true;
                pass_samples = shape.samples;
                pipeline = null;
                bound_slots = 0;
                index_bound = false;
            },
            .end_pass => {
                if (!in_pass) return self.refuseAt(at, "endPass outside a pass");
                in_pass = false;
            },
            .set_pipeline => |h| {
                if (!in_pass) return self.refuseAt(at, "setPipeline outside a pass");
                pipeline = self.pipelines.get(h) orelse return self.refuseAt(at, "setPipeline: the pipeline is not alive");
            },
            .set_viewport, .set_scissor => if (!in_pass) return self.refuseAt(at, "viewport or scissor outside a pass"),
            .set_vertex_buffer => |binding| {
                if (!in_pass) return self.refuseAt(at, "setVertexBuffer outside a pass");
                if (binding.slot >= 32) return self.refuseAt(at, "setVertexBuffer: slot too high");
                const buffer = self.buffers.get(binding.buffer) orelse return self.refuseAt(at, "setVertexBuffer: the buffer is not alive");
                if (buffer.kind != .vertex) return self.refuseAt(at, "setVertexBuffer: not a vertex buffer");
                if (binding.offset >= buffer.size) return self.refuseAt(at, "setVertexBuffer: offset past the end");
                bound_slots |= @as(u32, 1) << @intCast(binding.slot);
            },
            .set_index_buffer => |binding| {
                if (!in_pass) return self.refuseAt(at, "setIndexBuffer outside a pass");
                const buffer = self.buffers.get(binding.buffer) orelse return self.refuseAt(at, "setIndexBuffer: the buffer is not alive");
                if (buffer.kind != .index) return self.refuseAt(at, "setIndexBuffer: not an index buffer");
                index_bound = true;
            },
            .set_uniform_buffer => |binding| {
                if (!in_pass) return self.refuseAt(at, "setUniformBuffer outside a pass");
                const buffer = self.buffers.get(binding.buffer) orelse return self.refuseAt(at, "setUniformBuffer: the buffer is not alive");
                if (buffer.kind != .uniform) return self.refuseAt(at, "setUniformBuffer: not a uniform buffer");
            },
            .set_texture => |binding| {
                if (!in_pass) return self.refuseAt(at, "setTexture outside a pass");
                const texture = self.textures.get(binding.texture) orelse return self.refuseAt(at, "setTexture: the texture is not alive");
                if (!texture.usage.sampled) return self.refuseAt(at, "setTexture: the texture was not made with usage.sampled");
                if (!self.samplers.contains(binding.sampler)) return self.refuseAt(at, "setTexture: the sampler is not alive");
            },
            .draw => |draw| {
                const current = try self.drawable(at, in_pass, pass_samples, pipeline, bound_slots);
                _ = current;
                if (draw.vertex_count == 0 or draw.instance_count == 0) return self.refuseAt(at, "draw: nothing to draw");
            },
            .draw_indexed => |draw| {
                _ = try self.drawable(at, in_pass, pass_samples, pipeline, bound_slots);
                if (!index_bound) return self.refuseAt(at, "drawIndexed: no index buffer bound in this pass");
                if (draw.index_count == 0 or draw.instance_count == 0) return self.refuseAt(at, "drawIndexed: nothing to draw");
            },
            .generate_mips => |h| {
                if (in_pass) return self.refuseAt(at, "generateMips inside a pass");
                const texture = self.textures.get(h) orelse return self.refuseAt(at, "generateMips: the texture is not alive");
                if (texture.mip_levels < 2) return self.refuseAt(at, "generateMips: the texture has one mip level, so there is nothing below it to fill");
                if (texture.samples != 1) return self.refuseAt(at, "generateMips: a multisampled texture has no levels");
                if (texture.format.isDepth() or texture.format.isCompressed()) return self.refuseAt(at, "generateMips: not for a depth or a compressed format");
                if (!self.capabilities.formatSupport(texture.format).generate_mips) return self.unsupportedAt(at, "generateMips: this device cannot generate levels of that format");
            },
        }
    }
    if (in_pass) return self.refuse("submit: a pass was begun and never ended");
}

/// What the attachments of one pass have in common, gathered as they are
/// looked at. Zero is "not known yet", and stays that for a surface, whose
/// size the platform owns and may have changed a moment ago.
const PassShape = struct {
    width: u32 = 0,
    height: u32 = 0,
    /// Zero until the first attachment has spoken.
    samples: u32 = 0,
};

/// One attachment's part of `beginPass`: is it alive, is it a target, is
/// the level and layer it names there, does it agree with the others.
fn checkColorAttachment(self: *Device, at: usize, color: types.ColorAttachment, shape: *PassShape) Error!void {
    var source_samples: u32 = 1;
    switch (color.target) {
        .surface => |h| {
            if (!self.surfaces.contains(h)) return self.refuseAt(at, "beginPass: the surface is not alive");
            if (color.mip_level != 0 or color.layer != 0) return self.refuseAt(at, "beginPass: a surface has one level and one layer");
            try self.joinShape(at, shape, 0, 0, 1);
            if (color.resolve != null) return self.refuseAt(at, "beginPass: a surface is not multisampled, so there is nothing to resolve");
        },
        .texture => |h| {
            const texture = self.textures.get(h) orelse return self.refuseAt(at, "beginPass: the colour texture is not alive");
            if (!texture.usage.render_target) return self.refuseAt(at, "beginPass: the colour texture was not made with usage.render_target");
            if (texture.format.isDepth()) return self.refuseAt(at, "beginPass: a depth texture as a colour attachment");
            try self.checkSubresource(at, texture, color.mip_level, color.layer, shape);
            source_samples = texture.samples;
        },
    }

    if (color.resolve) |target| {
        const source = switch (color.target) {
            .texture => |h| self.textures.get(h).?,
            .surface => unreachable, // refused above
        };
        if (source_samples == 1) return self.refuseAt(at, "beginPass: resolve on an attachment that is not multisampled");
        switch (target) {
            .surface => |h| {
                const surface = self.surfaces.get(h) orelse return self.refuseAt(at, "beginPass: the surface to resolve into is not alive");
                const size = self.vtable.surfaceSize(self.impl, surface.native);
                if (size[0] != source.width or size[1] != source.height) return self.refuseAt(at, "beginPass: the surface to resolve into has another size");
            },
            .texture => |h| {
                const into = self.textures.get(h) orelse return self.refuseAt(at, "beginPass: the texture to resolve into is not alive");
                if (into.samples != 1) return self.refuseAt(at, "beginPass: the texture to resolve into is itself multisampled");
                if (!into.usage.render_target) return self.refuseAt(at, "beginPass: the texture to resolve into was not made with usage.render_target");
                if (into.format != source.format) return self.refuseAt(at, "beginPass: the texture to resolve into has another format");
                if (into.width != source.width or into.height != source.height) return self.refuseAt(at, "beginPass: the texture to resolve into has another size");
            },
        }
    }
}

/// A texture attachment's level and layer are inside it, and its size and
/// sample count are the ones the rest of the pass has.
fn checkSubresource(self: *Device, at: usize, texture: *resources.TextureEntry, mip: u32, layer: u32, shape: *PassShape) Error!void {
    if (mip >= texture.mip_levels) return self.refuseAt(at, "beginPass: an attachment names a mip level the texture does not have");
    if (layer >= texture.slices(mip)) return self.refuseAt(at, "beginPass: an attachment names a layer, face or slice the texture does not have");
    try self.joinShape(at, shape, types.mipExtent(texture.width, mip), types.mipExtent(texture.height, mip), texture.samples);
}

fn joinShape(self: *Device, at: usize, shape: *PassShape, width: u32, height: u32, samples: u32) Error!void {
    if (shape.samples != 0 and shape.samples != samples) return self.refuseAt(at, "beginPass: attachments with different sample counts");
    shape.samples = samples;
    if (width == 0 or height == 0) return;
    if (shape.width != 0 and (shape.width != width or shape.height != height)) return self.refuseAt(at, "beginPass: attachments of different sizes");
    shape.width = width;
    shape.height = height;
}

fn drawable(self: *Device, at: usize, in_pass: bool, pass_samples: u32, pipeline: ?*resources.PipelineEntry, bound_slots: u32) Error!*resources.PipelineEntry {
    if (!in_pass) return self.refuseAt(at, "draw outside a pass");
    const current = pipeline orelse return self.refuseAt(at, "draw with no pipeline set in this pass");
    const needed: u32 = if (current.buffer_slots == 0) 0 else (@as(u32, 1) << @intCast(current.buffer_slots)) - 1;
    if (bound_slots & needed != needed) return self.refuseAt(at, "draw: the pipeline reads a vertex buffer slot nothing is bound to");
    if (current.samples != pass_samples) return self.refuseAt(at, "draw: the pipeline's samples are not the pass's");
    return current;
}

fn refuse(self: *Device, message: []const u8) Error {
    var w = self.logWriter();
    w.writeAll(message) catch {};
    self.keepLog(&w);
    return error.InvalidArgument;
}

fn refuseAt(self: *Device, at: usize, message: []const u8) Error {
    var w = self.logWriter();
    w.print("command {d}: {s}", .{ at, message }) catch {};
    self.keepLog(&w);
    return error.InvalidArgument;
}

/// The request is fine, and this device cannot do it: `caps` says so, and
/// the answer is `Unsupported` rather than `InvalidArgument` because another
/// device might.
fn unsupported(self: *Device, message: []const u8) Error {
    var w = self.logWriter();
    w.writeAll(message) catch {};
    self.keepLog(&w);
    return error.Unsupported;
}

fn unsupportedAt(self: *Device, at: usize, message: []const u8) Error {
    var w = self.logWriter();
    w.print("command {d}: {s}", .{ at, message }) catch {};
    self.keepLog(&w);
    return error.Unsupported;
}

// -------------------------------------------------------------------------
// Tests - on the backend that needs no GPU
// -------------------------------------------------------------------------

const testing = std.testing;

fn nothing() !Device {
    return Device.init(testing.allocator, .{ .backend = .none });
}

test "a device that draws nothing still keeps its books" {
    var device = try nothing();
    defer device.deinit();

    try testing.expectEqual(types.Backend.none, device.info().backend);

    const vertices = try device.createBuffer(.{ .kind = .vertex, .size = 64 });
    const texture = try device.createTexture(.{ .width = 4, .height = 4 });
    try testing.expectEqual(@as(u32, 4), (try device.textureSize(texture)).width);

    // A read comes back the right size, and black.
    const pixels = try device.readTexture(texture, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(@as(usize, 64), pixels.len);
    try testing.expectEqual(@as(u8, 0), pixels[7]);

    // And a destroyed handle is refused, not followed.
    device.destroyBuffer(vertices);
    try testing.expectError(error.InvalidHandle, device.updateBuffer(vertices, 0, "abcd"));
    try testing.expectError(error.InvalidHandle, device.updateBuffer(vertices, 0, "abcd"));
}

test "creation refuses what no backend could do" {
    var device = try nothing();
    defer device.deinit();

    try testing.expectError(error.InvalidArgument, device.createBuffer(.{ .kind = .vertex, .size = 0 }));
    try testing.expectError(error.InvalidArgument, device.createTexture(.{ .width = 0, .height = 1 }));
    try testing.expectError(error.InvalidArgument, device.createTexture(.{ .width = 2, .height = 2, .data = "short" }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "fewer bytes") != null);

    const shader = try device.createShader(.{});
    try testing.expectError(error.InvalidArgument, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float4, .offset = 0, .buffer = 1 }},
        .buffers = &.{.{ .stride = 16 }},
    }));
    try testing.expectError(error.InvalidArgument, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float4, .offset = 8, .buffer = 0 }},
        .buffers = &.{.{ .stride = 16 }},
    }));
    try testing.expectError(error.InvalidHandle, device.createPipeline(.{
        .shader = .none,
        .attributes = &.{},
        .buffers = &.{},
    }));
}

test "a surface is shown in the mode it was made with, and in any it is set to" {
    var device = try nothing();
    defer device.deinit();
    const surface = try device.createSurface(.{ .width = 8, .height = 8, .present_mode = .mailbox });
    try std.testing.expectEqual(types.PresentMode.mailbox, try device.presentMode(surface));
    for (std.enums.values(types.PresentMode)) |mode| {
        try device.setPresentMode(surface, mode);
        try std.testing.expectEqual(mode, try device.presentMode(surface));
        try device.present(surface);
    }
    // Only enabled and adaptive wait for the refresh.
    try std.testing.expect(types.PresentMode.enabled.waits() and types.PresentMode.adaptive.waits());
    try std.testing.expect(!types.PresentMode.disabled.waits() and !types.PresentMode.mailbox.waits());
}

test "a frame that makes sense goes through, and one that does not is named" {
    var device = try nothing();
    defer device.deinit();

    const surface = try device.createSurface(.{ .width = 64, .height = 64 });
    const shader = try device.createShader(.{});
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
    });
    const quad = try device.createBuffer(.{ .kind = .vertex, .size = 32 });
    const indices = try device.createBuffer(.{ .kind = .index, .size = 12 });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const plain = try device.createTexture(.{ .width = 8, .height = 8 });

    // The good frame.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, quad, 0);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.setIndexBuffer(indices, .u16);
        try cmd.drawIndexed(.{ .index_count = 6 });
        try cmd.endPass();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
        try cmd.endPass();
        try device.submit();
        try testing.expectEqual(@as(usize, 0), device.diagnostics().len);
    }

    // Draw with nothing bound.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try cmd.setPipeline(pipeline);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "command 2") != null);
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "vertex buffer slot") != null);
    }

    // Draw outside a pass.
    {
        const cmd = device.begin();
        try cmd.draw(.{ .vertex_count = 3 });
        try testing.expectError(error.InvalidArgument, device.submit());
    }

    // A pass into a texture that cannot be drawn into.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = plain } } });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "render_target") != null);
    }

    // A pass never ended.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try testing.expectError(error.InvalidArgument, device.submit());
    }

    // The wrong kind of buffer in the wrong slot.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try cmd.setIndexBuffer(quad, .u16);
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
    }

    // And after all that, a good frame still goes through: a refused list
    // does not leave anything behind.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try cmd.endPass();
        try device.submit();
    }
}

test "a multi-attachment pass validates every extra colour attachment too" {
    var device = try nothing();
    defer device.deinit();

    const color = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const normal = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const not_a_target = try device.createTexture(.{ .width = 8, .height = 8 });
    const surface = try device.createSurface(.{ .width = 8, .height = 8 });

    // A good multi-attachment pass goes through.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .texture = color } },
            .extra_colors = &.{.{ .target = .{ .texture = normal } }},
        });
        try cmd.endPass();
        try device.submit();
        try testing.expectEqual(@as(usize, 0), device.diagnostics().len);
    }

    // An extra attachment that isn't a render target is refused.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .texture = color } },
            .extra_colors = &.{.{ .target = .{ .texture = not_a_target } }},
        });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "render_target") != null);
    }

    // A multi-attachment pass cannot target the surface, primary or extra.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .surface = surface } },
            .extra_colors = &.{.{ .target = .{ .texture = normal } }},
        });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "surface") != null);
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .texture = color } },
            .extra_colors = &.{.{ .target = .{ .surface = surface } }},
        });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
    }
}

// -------------------------------------------------------------------------
// Tests - textures, samplers and passes for 3D
// -------------------------------------------------------------------------

/// The `none` backend, saying it can do less: nothing compressed, no border
/// wrap, no sampler bias, and multisampling only on colour at four samples.
/// What a real device that is not the best one looks like, without a GPU.
fn modest() !Device {
    return tweaked(struct {
        fn apply(answer: *types.Caps) void {
            answer.features = .{};
            answer.limits.max_texture_2d = 1024;
            answer.limits.max_anisotropy = 4;
            answer.limits.max_color_attachments = 2;
            for (std.enums.values(types.Format)) |format| {
                var support = answer.formats.get(format);
                if (format.isCompressed()) support = .{};
                if (support.sample_counts != 0) support.sample_counts = if (format.isDepth() or format.isCompressed()) 0b1 else 0b101;
                answer.formats.set(format, support);
            }
        }
    }.apply);
}

/// The `none` backend with whatever its caps are made to say. A device that
/// is a fact about hardware - no volumes, whole blocks only - is a function
/// here, not a GPU.
fn tweaked(comptime change: fn (*types.Caps) void) !Device {
    const Tweaked = struct {
        fn tweakedCaps(impl: backend.Impl) types.Caps {
            var answer = none_backend.vtable.caps(impl);
            change(&answer);
            return answer;
        }
        const table: backend.Vtable = blk: {
            var v = none_backend.vtable;
            v.caps = tweakedCaps;
            break :blk v;
        };
        fn open(gpa: Allocator, desc: types.DeviceDesc) Error!backend.Opened {
            const opened = try none_backend.open(gpa, desc);
            return .{ opened[0], &table };
        }
    };
    return Device.initWith(testing.allocator, .{}, .{ .name = "tweaked", .clip = .gl, .open = Tweaked.open });
}

test "a shape a format does not come in is Unsupported before any size is looked at" {
    var device = try tweaked(struct {
        fn apply(answer: *types.Caps) void {
            // Direct3D 11's answer for depth: no volume. And a device with no volumes at all.
            var depth = answer.formats.get(.depth32_float);
            depth.dimensions.remove(.d3);
            answer.formats.set(.depth32_float, depth);
            answer.limits.max_texture_3d = 0;
        }
    }.apply);
    defer device.deinit();

    try testing.expect(device.caps().formatSupport(.rgba8_unorm).dimensions.contains(.d3));
    try testing.expect(!device.caps().formatSupport(.depth32_float).dimensions.contains(.d3));
    try testing.expectError(error.Unsupported, device.createTexture(.{
        .dimension = .d3,
        .width = 4,
        .height = 4,
        .depth_or_layers = 4,
        .format = .depth32_float,
        .usage = .{ .render_target = true },
    }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "shape") != null);
    // The colour volume is not refused for its shape, but it is past this device's zero limit.
    try testing.expectError(error.InvalidArgument, device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4 }));
    _ = try device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .render_target = true } });
}

test "a device that wants whole blocks refuses a compressed texture that is not" {
    var whole = try tweaked(struct {
        fn apply(answer: *types.Caps) void {
            answer.features.compressed_partial_blocks = false;
        }
    }.apply);
    defer whole.deinit();
    var any = try nothing();
    defer any.deinit();

    // 10 by 10 is two and a half BC1 blocks each way.
    try testing.expectError(error.Unsupported, whole.createTexture(.{ .width = 10, .height = 10, .format = .bc1_rgba_unorm }));
    try testing.expect(std.mem.indexOf(u8, whole.diagnostics(), "whole blocks") != null);
    _ = try whole.createTexture(.{ .width = 12, .height = 8, .format = .bc1_rgba_unorm });
    // The levels below the first may end in a partial block on any device: 12 by 8 goes 6 by 4, 3 by 2, 1 by 1.
    _ = try whole.createTexture(.{ .width = 12, .height = 8, .format = .bc1_rgba_unorm, .mip_levels = 0 });
    // A device that allows it, does.
    _ = try any.createTexture(.{ .width = 10, .height = 10, .format = .bc1_rgba_unorm });
}

test "resolving into a surface needs the surface to be the same size" {
    var device = try nothing();
    defer device.deinit();

    const msaa = try device.createTexture(.{ .width = 16, .height = 16, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const right = try device.createSurface(.{ .width = 16, .height = 16 });
    const wrong = try device.createSurface(.{ .width = 8, .height = 16 });

    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .resolve = .{ .surface = right } } });
        try cmd.endPass();
        try device.submit();
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .resolve = .{ .surface = wrong } } });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "another size") != null);
    }
}

test "a texture is a shape, and the device says what shape it is" {
    var device = try nothing();
    defer device.deinit();

    const plain = try device.textureInfo(try device.createTexture(.{ .width = 16, .height = 8 }));
    try testing.expectEqual(types.Dimension.d2, plain.dimension);
    try testing.expectEqual(@as(u32, 1), plain.mip_levels);
    try testing.expectEqual(@as(u32, 1), plain.depth_or_layers);

    // Zero mip levels is the whole chain, counted for you.
    const chain = try device.textureInfo(try device.createTexture(.{ .width = 16, .height = 8, .mip_levels = 0 }));
    try testing.expectEqual(@as(u32, 5), chain.mip_levels);

    const cube = try device.textureInfo(try device.createTexture(.{ .dimension = .cube, .width = 32, .height = 32, .mip_levels = 0 }));
    try testing.expectEqual(@as(u32, 6), cube.depth_or_layers);
    try testing.expectEqual(@as(u32, 6), cube.mip_levels);

    const volume = try device.textureInfo(try device.createTexture(.{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 16, .mip_levels = 0 }));
    try testing.expectEqual(@as(u32, 16), volume.depth_or_layers);
    try testing.expectEqual(@as(u32, 5), volume.mip_levels);

    const array = try device.textureInfo(try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 4 }));
    try testing.expectEqual(@as(u32, 4), array.depth_or_layers);
}

test "a texture that does not add up is refused, with the reason" {
    var device = try nothing();
    defer device.deinit();

    const cases = [_]struct { types.TextureDesc, []const u8 }{
        .{ .{ .dimension = .cube, .width = 8, .height = 4 }, "square" },
        .{ .{ .width = 8, .height = 8, .depth_or_layers = 3 }, ".d2_array" },
        .{ .{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 0 }, "no layers" },
        .{ .{ .width = 8, .height = 8, .mip_levels = 5 }, "more mip levels" },
        .{ .{ .width = 8, .height = 8, .usage = .{ .sampled = false } }, "neither sampled" },
        .{ .{ .width = 8, .height = 8, .format = .bc1_rgba_unorm, .usage = .{ .render_target = true } }, "compressed" },
        .{ .{ .width = 8, .height = 8, .samples = 4 }, "usage.render_target" },
        .{ .{ .width = 8, .height = 8, .samples = 4, .usage = .{ .render_target = true } }, "cannot be sampled" },
        .{ .{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 8, .samples = 4, .usage = .{ .sampled = false, .render_target = true } }, "only a .d2" },
        .{ .{ .width = 8, .height = 8, .samples = 4, .mip_levels = 2, .usage = .{ .sampled = false, .render_target = true } }, "one mip level" },
        .{ .{ .width = 99999, .height = 8 }, "max_texture_2d" },
    };
    for (cases) |case| {
        try testing.expectError(error.InvalidArgument, device.createTexture(case[0]));
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), case[1]) != null);
    }

    // Data for a cube is six faces, and a short one is named.
    const face = [_]u8{0} ** (8 * 8 * 4);
    try testing.expectError(error.InvalidArgument, device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .data = &face }));
    const faces = [_]u8{0} ** (8 * 8 * 4 * 6);
    _ = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .data = &faces });
}

test "what the device cannot do is Unsupported, and its caps said so first" {
    var device = try modest();
    defer device.deinit();

    // The caps are readable, so a program can ask instead of trying.
    try testing.expect(!device.caps().formatSupport(.bc7_rgba_unorm).sampled);
    try testing.expect(device.caps().formatSupport(.rgba8_unorm).supportsSamples(4));
    try testing.expect(!device.caps().formatSupport(.rgba8_unorm).supportsSamples(8));

    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .format = .bc7_rgba_unorm }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "cannot sample") != null);
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .samples = 8, .usage = .{ .sampled = false, .render_target = true } }));
    _ = try device.createTexture(.{ .width = 8, .height = 8, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });

    // Beyond a limit is not a mistake in the request but a fact about the device.
    try testing.expectError(error.InvalidArgument, device.createTexture(.{ .width = 2048, .height = 8 }));

    try testing.expectError(error.Unsupported, device.createSampler(.{ .wrap_u = .border }));
    try testing.expectError(error.Unsupported, device.createSampler(.{ .lod_bias = 0.5 }));
    // More taps than it has is clamped, not refused.
    _ = try device.createSampler(.{ .max_anisotropy = 16, .mip_filter = .linear });
    try testing.expectError(error.InvalidArgument, device.createSampler(.{ .max_anisotropy = 0 }));
    try testing.expectError(error.InvalidArgument, device.createSampler(.{ .lod_min = 4, .lod_max = 2 }));
}

test "a region is checked against the level it names" {
    var device = try nothing();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0 });
    const level1 = [_]u8{0} ** (4 * 4 * 4);
    // Level one is four by four: the whole of it, and a corner of it.
    try device.writeTexture(texture, .{ .mip = 1 }, &level1, 0, 0);
    try device.writeTexture(texture, .{ .mip = 1, .x = 2, .y = 2, .width = 2, .height = 2 }, level1[0 .. 2 * 2 * 4], 0, 0);
    // And what does not fit is named.
    try testing.expectError(error.InvalidArgument, device.writeTexture(texture, .{ .mip = 9 }, &level1, 0, 0));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "mip level") != null);
    try testing.expectError(error.InvalidArgument, device.writeTexture(texture, .{ .mip = 1, .x = 3, .width = 2 }, &level1, 0, 0));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "past the level") != null);
    try testing.expectError(error.InvalidArgument, device.writeTexture(texture, .{ .mip = 1 }, level1[0..8], 0, 0));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "fewer bytes") != null);
    try testing.expectError(error.InvalidArgument, device.writeTexture(texture, .{ .mip = 1 }, &level1, 8, 0));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "narrower") != null);

    // A padded row is fine, and needs the padding in the bytes.
    const padded = [_]u8{0} ** (32 * 3 + 16);
    try device.writeTexture(texture, .{ .mip = 1 }, &padded, 32, 0);

    // A cube is written a face at a time, or all six.
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4 });
    const face = [_]u8{0} ** (4 * 4 * 4);
    try device.writeTexture(cube, .{ .z = 3, .depth = 1 }, &face, 0, 0);
    try testing.expectError(error.InvalidArgument, device.writeTexture(cube, .{ .z = 6 }, &face, 0, 0));
    try testing.expectError(error.InvalidArgument, device.writeTexture(cube, .{ .z = 5, .depth = 2 }, &face, 0, 0));
    try testing.expectError(error.InvalidArgument, device.writeTexture(cube, .{}, &face, 0, 0)); // six faces need six faces of bytes
    const all = [_]u8{0} ** (4 * 4 * 4 * 6);
    try device.writeTexture(cube, .{}, &all, 0, 0);

    // A depth texture, and a multisampled one, take nothing from memory.
    const depth = try device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .render_target = true } });
    try testing.expectError(error.InvalidArgument, device.writeTexture(depth, .{}, &face, 0, 0));
}

test "a compressed texture is written in whole blocks" {
    var device = try nothing();
    defer device.deinit();

    // 8 by 8 of BC1 is four blocks of eight bytes.
    const texture = try device.createTexture(.{ .width = 8, .height = 8, .format = .bc1_rgba_unorm, .mip_levels = 0 });
    const blocks = [_]u8{0} ** 32;
    try device.writeTexture(texture, .{}, &blocks, 0, 0);
    // A block at a time is fine; half a block is not.
    try device.writeTexture(texture, .{ .x = 4, .y = 4, .width = 4, .height = 4 }, blocks[0..8], 0, 0);
    try testing.expectError(error.InvalidArgument, device.writeTexture(texture, .{ .x = 2, .width = 4, .height = 4 }, blocks[0..8], 0, 0));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "whole blocks") != null);
    // The last level is one texel, and is still one block.
    try device.writeTexture(texture, .{ .mip = 3 }, blocks[0..8], 0, 0);
    // Compressed pixels are not something to read back as colour.
    try testing.expectError(error.InvalidArgument, device.readTexture(texture, testing.allocator));
}

test "a read names the image it wants" {
    var device = try nothing();
    defer device.deinit();

    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 4, .depth_or_layers = 3, .mip_levels = 0 });
    const level = try device.readSubresource(array, .{ .mip = 2, .layer = 2 }, testing.allocator);
    defer testing.allocator.free(level);
    try testing.expectEqual(@as(usize, 2 * 1 * 4), level.len);

    try testing.expectError(error.InvalidArgument, device.readSubresource(array, .{ .mip = 4 }, testing.allocator));
    try testing.expectError(error.InvalidArgument, device.readSubresource(array, .{ .layer = 3 }, testing.allocator));

    // A volume's slices shrink with its levels; an array's layers do not.
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 8, .mip_levels = 0 });
    const slice = try device.readSubresource(volume, .{ .mip = 1, .layer = 3 }, testing.allocator);
    testing.allocator.free(slice);
    try testing.expectError(error.InvalidArgument, device.readSubresource(volume, .{ .mip = 1, .layer = 4 }, testing.allocator));
    const last = try device.readSubresource(array, .{ .mip = 3, .layer = 2 }, testing.allocator);
    testing.allocator.free(last);
}

test "a pass can draw depth alone, into a level, a face, or a multisampled target" {
    var device = try nothing();
    defer device.deinit();

    const shader = try device.createShader(.{});
    const shadow_pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float3, .offset = 0 }},
        .buffers = &.{.{ .stride = 12 }},
        .color_format = null,
        .depth_format = .depth32_float,
        .depth = .standard,
    });
    const vertices = try device.createBuffer(.{ .kind = .vertex, .size = 36 });

    const shadow_map = try device.createTexture(.{ .width = 16, .height = 16, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    const shadow_cube = try device.createTexture(.{ .dimension = .cube, .width = 16, .height = 16, .format = .depth32_float, .usage = .{ .render_target = true } });
    const atlas = try device.createTexture(.{ .width = 16, .height = 16, .mip_levels = 0, .usage = .{ .render_target = true } });

    // Depth only, then one face of a cube, then a level of a chain.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = shadow_map } });
        try cmd.setPipeline(shadow_pipeline);
        try cmd.setVertexBuffer(0, vertices, 0);
        try cmd.draw(.{ .vertex_count = 3 });
        try cmd.endPass();
        try cmd.beginPass(.{ .depth = .{ .texture = shadow_cube, .layer = 4 } });
        try cmd.endPass();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = atlas }, .mip_level = 2 } });
        try cmd.endPass();
        try cmd.generateMips(atlas);
        try device.submit();
    }

    // No attachments at all, and colours with nothing to hang them on.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{});
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "nothing to draw into") != null);
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = shadow_map }, .extra_colors = &.{.{ .target = .{ .texture = atlas } }} });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
    }

    // A level or a face the texture does not have.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = atlas }, .mip_level = 5 } });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "mip level") != null);
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = shadow_cube, .layer = 6 } });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
    }

    // Attachments of different sizes: a depth buffer for the wrong level.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = atlas }, .mip_level = 1 }, .depth = .{ .texture = shadow_map } });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "different sizes") != null);
    }
    {
        // ...and the right one goes through: level one of a 16 texture is 8.
        const small_depth = try device.createTexture(.{ .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .render_target = true } });
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = atlas }, .mip_level = 1 }, .depth = .{ .texture = small_depth } });
        try cmd.endPass();
        try device.submit();
    }
}

test "a multisampled pass resolves, and a pipeline has to agree about the samples" {
    var device = try nothing();
    defer device.deinit();

    const surface = try device.createSurface(.{ .width = 16, .height = 16 });
    const shader = try device.createShader(.{});
    const single = try device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{} });
    const multi = try device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{}, .samples = 4 });
    try testing.expectError(error.InvalidArgument, device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{}, .samples = 3 }));
    try testing.expectError(error.InvalidArgument, device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{}, .color_format = null }));

    const msaa = try device.createTexture(.{ .width = 16, .height = 16, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const resolved = try device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .render_target = true } });
    const other_size = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const other_format = try device.createTexture(.{ .width = 16, .height = 16, .format = .rgba16_float, .usage = .{ .render_target = true } });

    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .resolve = .{ .texture = resolved } } });
        try cmd.setPipeline(multi);
        try cmd.draw(.{ .vertex_count = 3 });
        try cmd.endPass();
        // The same into the window.
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .resolve = .{ .surface = surface } } });
        try cmd.endPass();
        try device.submit();
    }
    // A pipeline for one sample in a pass of four is a draw that would be wrong.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa } } });
        try cmd.setPipeline(single);
        try cmd.draw(.{ .vertex_count = 3 });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "samples") != null);
    }
    // Resolving what is not multisampled, or into what does not match.
    const bad = [_]types.ColorAttachment{
        .{ .target = .{ .texture = resolved }, .resolve = .{ .texture = other_size } },
        .{ .target = .{ .surface = surface }, .resolve = .{ .texture = resolved } },
        .{ .target = .{ .texture = msaa }, .resolve = .{ .texture = other_size } },
        .{ .target = .{ .texture = msaa }, .resolve = .{ .texture = other_format } },
        .{ .target = .{ .texture = msaa }, .resolve = .{ .texture = msaa } },
    };
    for (bad) |attachment| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = attachment });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
    }
}

test "generateMips is a command, outside a pass, on a texture with something to fill" {
    var device = try modest();
    defer device.deinit();

    const chain = try device.createTexture(.{ .width = 16, .height = 16, .mip_levels = 0 });
    const flat = try device.createTexture(.{ .width = 16, .height = 16 });
    const depth = try device.createTexture(.{ .width = 16, .height = 16, .format = .depth32_float, .mip_levels = 0, .usage = .{ .render_target = true } });
    const target = try device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .render_target = true } });

    {
        const cmd = device.begin();
        try cmd.generateMips(chain);
        try device.submit();
    }
    {
        const cmd = device.begin();
        try cmd.generateMips(flat);
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "nothing below it") != null);
    }
    {
        const cmd = device.begin();
        try cmd.generateMips(depth);
        try testing.expectError(error.InvalidArgument, device.submit());
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
        try cmd.generateMips(chain);
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "inside a pass") != null);
    }
    {
        const cmd = device.begin();
        try cmd.generateMips(.none);
        try testing.expectError(error.InvalidArgument, device.submit());
    }
}

test "a device with fewer colour attachments refuses a pass with more" {
    var device = try modest();
    defer device.deinit();

    const a = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const b = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const c = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });

    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = a } }, .extra_colors = &.{.{ .target = .{ .texture = b } }} });
        try cmd.endPass();
        try device.submit();
    }
    {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .texture = a } },
            .extra_colors = &.{ .{ .target = .{ .texture = b } }, .{ .target = .{ .texture = c } } },
        });
        try cmd.endPass();
        try testing.expectError(error.InvalidArgument, device.submit());
        try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "max_color_attachments") != null);
    }
}

// -------------------------------------------------------------------------
// Tests - choosing a backend
// -------------------------------------------------------------------------

test "every backend this build brings has an opener with its own name" {
    for (available()) |tag| {
        const how = opener(tag) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(@tagName(tag), how.name);
        try testing.expectEqual(tag, how.tag);
        try testing.expectEqual(tag.clip(), how.clip);
    }
    // `.vulkan` and `.d3d12` are both in `available()` above now, and so
    // already covered by the loop; checked again here so this test still
    // says something if either is ever pulled back out of it. `other` is
    // whatever a caller makes - never built in.
    try testing.expect(opener(.vulkan) != null);
    if (builtin.os.tag == .windows) try testing.expect(opener(.d3d12) != null);
    try testing.expect(opener(.other) == null);
}

test "probing the Vulkan loader is safe whether or not it is installed" {
    // A machine without Vulkan is a supported deployment state, so this is
    // deliberately only a no-crash probe.
    _ = vulkanLoaderAvailable();
}

test "a built-in opener opens the same device as init does" {
    var by_init = try Device.init(testing.allocator, .{ .backend = .none });
    defer by_init.deinit();
    var by_opener = try Device.initWith(testing.allocator, .{}, opener(.none).?);
    defer by_opener.deinit();

    try testing.expectEqual(by_init.backendTag(), by_opener.backendTag());
    try testing.expectEqual(types.Backend.none, by_opener.backendTag());
    try testing.expectEqualStrings("none", by_opener.info().name);
    try testing.expectEqualStrings(by_init.info().renderer, by_opener.info().renderer);
    try testing.expectEqual(by_init.clip(), by_opener.clip());
}

test "initWith opens a backend the caller supplies, under its own name and clip space" {
    const flipped: math.Clip = .{ .depth = .zero_to_one, .flip_y = true };
    const mine: backend.Opener = .{ .name = "mine", .clip = flipped, .open = none_backend.open };

    var device = try Device.initWith(testing.allocator, .{}, mine);
    defer device.deinit();

    try testing.expectEqual(types.Backend.other, device.backendTag());
    try testing.expectEqualStrings("mine", device.info().name);
    try testing.expectEqual(types.Backend.other, device.info().backend);
    try testing.expectEqual(flipped, device.clip());

    // it is a working device, not a label
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = 16 });
    device.destroyBuffer(buffer);
    try testing.expectError(error.InvalidHandle, device.updateBuffer(buffer, 0, &.{ 1, 2, 3, 4 }));

    // and it prints under its name
    var text: [64]u8 = undefined;
    var w: Io.Writer = .fixed(&text);
    try w.print("{f}", .{device.info()});
    try testing.expectEqualStrings("mine: nothing at all", w.buffered());
}

fn refuses(_: Allocator, _: types.DeviceDesc) Error!backend.Opened {
    return error.NoDevice;
}

test "an opener that refuses leaves nothing behind" {
    const how: backend.Opener = .{ .name = "absent", .clip = .gl, .open = refuses };
    try testing.expectError(error.NoDevice, Device.initWith(testing.allocator, .{}, how));
}
