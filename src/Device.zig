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

const none_backend = @import("backend/none.zig");
const gl_backend = @import("backend/gl.zig");
const d3d11_backend = if (builtin.os.tag == .windows) @import("backend/d3d11.zig") else void;

const Device = @This();

pub const Error = types.Error;

gpa: Allocator,
impl: backend.Impl,
vtable: *const backend.Vtable,
tag: types.Backend,

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
pub fn available() []const types.Backend {
    return if (builtin.os.tag == .windows)
        &.{ .d3d11, .gl, .none }
    else
        &.{ .gl, .none };
}

pub fn init(gpa: Allocator, desc: types.DeviceDesc) Error!Device {
    const chosen: types.Backend = switch (desc.backend) {
        .none => .none,
        .gl => .gl,
        .d3d11 => .d3d11,
        .auto => if (desc.gl != null) .gl else if (builtin.os.tag == .windows) .d3d11 else return error.Unsupported,
    };

    const opened = switch (chosen) {
        .none => try none_backend.open(gpa, desc),
        .gl => try gl_backend.open(gpa, desc),
        .d3d11 => if (builtin.os.tag == .windows) try d3d11_backend.open(gpa, desc) else return error.Unsupported,
    };

    return .{
        .gpa = gpa,
        .impl = opened[0],
        .vtable = opened[1],
        .tag = chosen,
        .list = .init(gpa),
    };
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

pub fn backendTag(self: *const Device) types.Backend {
    return self.tag;
}

pub fn info(self: *const Device) types.Info {
    return self.vtable.info(self.impl);
}

/// Which clip space a projection for this device is built for. Hand it to
/// `fluxion-math`'s `perspective` and `orthographic`.
pub fn clip(self: *const Device) math.Clip {
    return self.tag.clip();
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
    if (desc.width == 0 or desc.height == 0) return self.refuse("createTexture: a texture with no area");
    if (desc.format.isDepth() and desc.data != null) return self.refuse("createTexture: a depth texture cannot be filled from memory");
    if (desc.data) |data| {
        const needed = desc.effectiveRowPitch() * (desc.height - 1) + desc.tightRowPitch();
        if (data.len < needed) return self.refuse("createTexture: fewer bytes than the size and pitch need");
    }
    const native = try self.vtable.createTexture(self.impl, desc);
    errdefer self.vtable.destroyTexture(self.impl, native);
    return self.textures.add(self.gpa, .{
        .native = native,
        .width = desc.width,
        .height = desc.height,
        .format = desc.format,
        .usage = desc.usage,
    });
}

pub fn destroyTexture(self: *Device, h: types.Texture) void {
    if (self.textures.remove(h)) |entry| self.vtable.destroyTexture(self.impl, entry.native);
}

/// Replace every texel. `row_pitch` of zero means tightly packed.
pub fn updateTexture(self: *Device, h: types.Texture, bytes: []const u8, row_pitch: usize) Error!void {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    if (entry.format.isDepth()) return self.refuse("updateTexture: a depth texture cannot be filled from memory");
    const tight = @as(usize, entry.width) * entry.format.bytesPerPixel();
    const pitch = if (row_pitch == 0) tight else row_pitch;
    if (pitch < tight) return self.refuse("updateTexture: a row pitch narrower than a row");
    if (bytes.len < pitch * (entry.height - 1) + tight) return self.refuse("updateTexture: fewer bytes than the texture needs");
    try self.vtable.updateTexture(self.impl, entry.native, bytes, pitch);
}

/// The texels as RGBA, eight bits a channel, top row first, tightly packed:
/// `width * height * 4` bytes the caller owns. The one way to look at what
/// was drawn without a window, and what every test here does.
pub fn readTexture(self: *Device, h: types.Texture, gpa: Allocator) Error![]u8 {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    if (entry.format.isDepth()) return self.refuse("readTexture: a depth texture cannot be read as colour");
    return self.vtable.readTexture(self.impl, entry.native, gpa);
}

pub fn textureSize(self: *Device, h: types.Texture) Error!types.Extent {
    const entry = self.textures.get(h) orelse return error.InvalidHandle;
    return .{ .width = entry.width, .height = entry.height };
}

pub fn createSampler(self: *Device, desc: types.SamplerDesc) Error!types.Sampler {
    const native = try self.vtable.createSampler(self.impl, desc);
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
    if (desc.color_format.isDepth()) return self.refuse("createPipeline: color_format is a depth format");

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
    return self.surfaces.add(self.gpa, .{ .native = native, .vsync = desc.vsync });
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

pub fn setVsync(self: *Device, h: types.Surface, vsync: bool) Error!void {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    entry.vsync = vsync;
}

/// Show what the last submit drew into the surface.
pub fn present(self: *Device, h: types.Surface) Error!void {
    const entry = self.surfaces.get(h) orelse return error.InvalidHandle;
    try self.vtable.present(self.impl, entry.native, entry.vsync);
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
    var pipeline: ?*resources.PipelineEntry = null;
    var bound_slots: u32 = 0; // a bit per vertex buffer slot
    var index_bound = false;

    for (list, 0..) |command, at| {
        switch (command) {
            .begin_pass => |pass| {
                if (in_pass) return self.refuseAt(at, "beginPass inside a pass");
                switch (pass.color.target) {
                    .surface => |h| if (!self.surfaces.contains(h)) return self.refuseAt(at, "beginPass: the surface is not alive"),
                    .texture => |h| {
                        const texture = self.textures.get(h) orelse return self.refuseAt(at, "beginPass: the colour texture is not alive");
                        if (!texture.usage.render_target) return self.refuseAt(at, "beginPass: the colour texture was not made with usage.render_target");
                        if (texture.format.isDepth()) return self.refuseAt(at, "beginPass: a depth texture as the colour attachment");
                    },
                }
                if (pass.depth) |depth| {
                    const texture = self.textures.get(depth.texture) orelse return self.refuseAt(at, "beginPass: the depth texture is not alive");
                    if (!texture.usage.render_target) return self.refuseAt(at, "beginPass: the depth texture was not made with usage.render_target");
                    if (!texture.format.isDepth()) return self.refuseAt(at, "beginPass: the depth attachment is not a depth format");
                }
                in_pass = true;
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
                const current = try self.drawable(at, in_pass, pipeline, bound_slots);
                _ = current;
                if (draw.vertex_count == 0 or draw.instance_count == 0) return self.refuseAt(at, "draw: nothing to draw");
            },
            .draw_indexed => |draw| {
                _ = try self.drawable(at, in_pass, pipeline, bound_slots);
                if (!index_bound) return self.refuseAt(at, "drawIndexed: no index buffer bound in this pass");
                if (draw.index_count == 0 or draw.instance_count == 0) return self.refuseAt(at, "drawIndexed: nothing to draw");
            },
        }
    }
    if (in_pass) return self.refuse("submit: a pass was begun and never ended");
}

fn drawable(self: *Device, at: usize, in_pass: bool, pipeline: ?*resources.PipelineEntry, bound_slots: u32) Error!*resources.PipelineEntry {
    if (!in_pass) return self.refuseAt(at, "draw outside a pass");
    const current = pipeline orelse return self.refuseAt(at, "draw with no pipeline set in this pass");
    const needed: u32 = if (current.buffer_slots == 0) 0 else (@as(u32, 1) << @intCast(current.buffer_slots)) - 1;
    if (bound_slots & needed != needed) return self.refuseAt(at, "draw: the pipeline reads a vertex buffer slot nothing is bound to");
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
