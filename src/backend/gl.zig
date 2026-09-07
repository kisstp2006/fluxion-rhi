// SPDX-License-Identifier: BSD-2-Clause

//! The OpenGL 3.3 core backend.
//!
//! The context is not made here and never will be: it comes through
//! `GlHooks` from whoever opened the window, and this file starts where
//! `getProcAddress` does, with a `fluxion-gl` table filled from it.
//!
//! **Three things OpenGL does differently, absorbed here.**
//!
//! The origin is the bottom left, so every viewport and scissor rectangle is
//! flipped against the current attachment's height, and every readback has
//! its rows reversed - a program sees the top-left world the other backends
//! have.
//!
//! A vertex array object captures buffer bindings, so there is one per
//! pipeline and the attribute pointers are re-specified whenever a vertex
//! buffer binding changes, at the draw that uses it. That costs a handful of
//! calls per draw that changes buffers and nothing otherwise.
//!
//! OpenGL 3.3 cannot instance and offset by a base vertex in the same draw -
//! `glDrawElementsInstancedBaseVertex` is 4.2 - so a draw that asks for both
//! is refused with `error.Unsupported`. Everything else in `Command` maps to
//! one call.
//!
//! **Uniform blocks and samplers are bound by name once**, when the pipeline
//! is made, from `PipelineDesc.uniform_blocks` and `PipelineDesc.textures`.
//! GLSL 330 has no `layout(binding = n)`; this is the seam that lets the same
//! `setUniformBuffer(slot, ...)` mean the same thing on Direct3D.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const opengl = @import("fluxion_gl");
const c = opengl.enums;
const gt = opengl.types;

const types = @import("../types.zig");
const backend = @import("../backend.zig");
const commands = @import("../commands.zig");
const Device = @import("../Device.zig");

const Error = backend.Error;

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const Gl = struct {
    gpa: Allocator,
    hooks: types.GlHooks,
    api: opengl.Gl,
    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,
    /// What the driver's debug output last complained about, kept rather
    /// than printed: `submit` turns it into `error.Failed`, and a test that
    /// wants to read it can.
    last_error: [256]u8 = undefined,
    last_error_len: usize = 0,
    had_error: bool = false,
    /// The one surface a context has. Made once, handed back on every
    /// `createSurface`.
    surface: SurfaceRes,

    // Per-submit state. Reset at every pass.
    pipeline: ?*PipelineRes = null,
    target_height: u32 = 0,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
    index: ?IndexBinding = null,
};

const max_vertex_slots = 8;

const VertexBinding = struct {
    name: gt.Uint = 0,
    offset: u32 = 0,
};

const IndexBinding = struct {
    name: gt.Uint,
    kind: gt.Enum,
    size: u32,
};

/// The shape `fluxion-dyn` wants a resolver in: something with a `get`.
const Resolver = struct {
    hooks: types.GlHooks,
    pub fn get(self: Resolver, name: [*:0]const u8) ?opengl.Proc {
        return self.hooks.get_proc_address(self.hooks.context, name);
    }
};

// -------------------------------------------------------------------------
// Resources
// -------------------------------------------------------------------------

const BufferRes = struct {
    name: gt.Uint,
    target: gt.Enum,
    size: usize,
};

const TextureRes = struct {
    name: gt.Uint,
    /// Made the first time the texture is drawn into or read back.
    framebuffer: gt.Uint = 0,
    width: u32,
    height: u32,
    format: types.Format,
};

const SamplerRes = struct {
    name: gt.Uint,
};

const ShaderRes = struct {
    program: gt.Uint,
};

const PipelineRes = struct {
    program: gt.Uint,
    vao: gt.Uint,
    attributes: []types.VertexAttribute,
    buffers: []types.VertexBufferLayout,
    topology: gt.Enum,
    blend: types.BlendState,
    depth: types.DepthState,
    cull: types.CullMode,
    front_face: types.FrontFace,
};

const SurfaceRes = struct {
    /// A surface is the default framebuffer; there is nothing to store but
    /// the fact that it exists.
    claimed: bool = false,
};

// -------------------------------------------------------------------------
// Opening
// -------------------------------------------------------------------------

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!struct { backend.Impl, *const backend.Vtable } {
    const hooks = desc.gl orelse return error.NoDevice;

    const self = try gpa.create(Gl);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .hooks = hooks,
        .api = undefined,
        .debug = desc.debug,
        .surface = .{},
    };

    // The whole 3.3 table, or the name of what was missing - which is what a
    // 2.1 context or a context that is not current looks like.
    self.api.load(Resolver{ .hooks = hooks }) catch return error.NoDevice;

    const version = self.api.version() catch return error.NoDevice;
    if (!version.atLeast(3, 3)) return error.NoDevice;

    if (self.api.string(c.renderer)) |name| {
        const n = @min(name.len, self.renderer.len);
        @memcpy(self.renderer[0..n], name[0..n]);
        self.renderer_len = n;
    }

    // State that never changes under this backend.
    self.api.pixelStorei(c.pack_alignment, 1);
    self.api.pixelStorei(c.unpack_alignment, 1);
    self.api.enable(c.scissor_test);

    if (desc.debug) {
        if (self.api.debugMessageCallback) |set| {
            self.api.enable(c.debug_output);
            self.api.enable(c.debug_output_synchronous);
            set(debugMessage, self);
        }
    }

    return .{ self, &vtable };
}

fn debugMessage(
    source: gt.Enum,
    kind: gt.Enum,
    id: gt.Uint,
    severity: gt.Enum,
    length: gt.Sizei,
    message: [*:0]const gt.Char,
    user_param: ?*const anyopaque,
) callconv(.c) void {
    _ = source;
    _ = id;
    _ = severity;
    // Only errors. The rest is the driver narrating buffer placement and
    // shader recompiles, which is worth reading in a graphics debugger and
    // not worth a line in a program's log.
    if (kind != c.debug_type_error) return;
    const self: *Gl = @ptrCast(@alignCast(@constCast(user_param orelse return)));
    const text = message[0..@intCast(length)];
    const n = @min(text.len, self.last_error.len);
    @memcpy(self.last_error[0..n], text[0..n]);
    self.last_error_len = n;
    self.had_error = true;
}

/// The last thing the driver's debug output called an error, or empty.
pub fn lastError(impl: backend.Impl) []const u8 {
    const self = cast(impl);
    return self.last_error[0..self.last_error_len];
}

const vtable: backend.Vtable = .{
    .deinit = deinit,
    .info = info,
    .createBuffer = createBuffer,
    .destroyBuffer = destroyBuffer,
    .updateBuffer = updateBuffer,
    .createTexture = createTexture,
    .destroyTexture = destroyTexture,
    .updateTexture = updateTexture,
    .readTexture = readTexture,
    .createSampler = createSampler,
    .destroySampler = destroySampler,
    .createShader = createShader,
    .destroyShader = destroyShader,
    .createPipeline = createPipeline,
    .destroyPipeline = destroyPipeline,
    .createSurface = createSurface,
    .destroySurface = destroySurface,
    .resizeSurface = resizeSurface,
    .surfaceSize = surfaceSize,
    .present = present,
    .submit = submit,
};

fn cast(impl: backend.Impl) *Gl {
    return @ptrCast(@alignCast(impl));
}

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

fn deinit(impl: backend.Impl) void {
    const self = cast(impl);
    self.gpa.destroy(self);
}

fn info(impl: backend.Impl) types.Info {
    const self = cast(impl);
    return .{ .backend = .gl, .renderer = self.renderer[0..self.renderer_len] };
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const res = try self.gpa.create(BufferRes);
    errdefer self.gpa.destroy(res);

    const target: gt.Enum = switch (desc.kind) {
        .vertex => c.array_buffer,
        .index => c.element_array_buffer,
        .uniform => c.uniform_buffer,
    };
    // A uniform buffer is rounded up to sixteen, as the std140 rules and
    // Direct3D both want; the size a program sees stays what it asked for.
    const size = if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 16) else desc.size;

    var name: gt.Uint = 0;
    api.genBuffers(1, @ptrCast(&name));
    api.bindBuffer(target, name);
    const usage: gt.Enum = if (desc.dynamic or desc.kind == .uniform) c.dynamic_draw else c.static_draw;
    if (desc.data) |data| {
        if (data.len == size) {
            api.bufferData(target, @intCast(size), data.ptr, usage);
        } else {
            api.bufferData(target, @intCast(size), null, usage);
            api.bufferSubData(target, 0, @intCast(data.len), data.ptr);
        }
    } else {
        api.bufferData(target, @intCast(size), null, usage);
    }

    res.* = .{ .name = name, .target = target, .size = size };
    return res;
}

fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    self.api.deleteBuffers(1, @ptrCast(&res.name));
    self.gpa.destroy(res);
}

fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) Error!void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    self.api.bindBuffer(res.target, res.name);
    self.api.bufferSubData(res.target, @intCast(offset), @intCast(bytes.len), bytes.ptr);
    // A vertex array remembers the element buffer it last saw; a bind here
    // for an update must not leave it pointing at a stray one.
    if (res.target == c.element_array_buffer) self.index = null;
    self.bindings_dirty = true;
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

const GlFormat = struct {
    internal: gt.Int,
    format: gt.Enum,
    kind: gt.Enum,
};

fn glFormat(format: types.Format) GlFormat {
    return switch (format) {
        .rgba8_unorm => .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte },
        .rgba8_unorm_srgb => .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte },
        .bgra8_unorm => .{ .internal = c.rgba8, .format = c.bgra, .kind = c.unsigned_byte },
        .r8_unorm => .{ .internal = c.r8, .format = c.red, .kind = c.unsigned_byte },
        .rgba16_float => .{ .internal = c.rgba16f, .format = c.rgba, .kind = c.half_float },
        .rgba32_float => .{ .internal = c.rgba32f, .format = c.rgba, .kind = c.float },
        .depth24_stencil8 => .{ .internal = c.depth24_stencil8, .format = c.depth_stencil, .kind = c.unsigned_int_24_8 },
        .depth32_float => .{ .internal = c.depth_component32f, .format = c.depth_component, .kind = c.float },
    };
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    var name: gt.Uint = 0;
    api.genTextures(1, @ptrCast(&name));
    api.bindTexture(c.texture_2d, name);

    const f = glFormat(desc.format);
    if (desc.data) |data| {
        api.pixelStorei(c.unpack_row_length, @intCast(desc.effectiveRowPitch() / desc.format.bytesPerPixel()));
        api.texImage2D(c.texture_2d, 0, f.internal, @intCast(desc.width), @intCast(desc.height), 0, f.format, f.kind, data.ptr);
        api.pixelStorei(c.unpack_row_length, 0);
    } else {
        api.texImage2D(c.texture_2d, 0, f.internal, @intCast(desc.width), @intCast(desc.height), 0, f.format, f.kind, null);
    }
    // One level and no mipmaps, so say so: the default minification filter
    // wants a complete chain and samples black without one.
    api.texParameteri(c.texture_2d, c.texture_max_level, 0);
    api.texParameteri(c.texture_2d, c.texture_min_filter, @intCast(c.linear));
    api.texParameteri(c.texture_2d, c.texture_mag_filter, @intCast(c.linear));

    res.* = .{ .name = name, .width = desc.width, .height = desc.height, .format = desc.format };
    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (res.framebuffer != 0) self.api.deleteFramebuffers(1, @ptrCast(&res.framebuffer));
    self.api.deleteTextures(1, @ptrCast(&res.name));
    self.gpa.destroy(res);
}

fn updateTexture(impl: backend.Impl, native: backend.Native, bytes: []const u8, row_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const f = glFormat(res.format);
    self.api.bindTexture(c.texture_2d, res.name);
    self.api.pixelStorei(c.unpack_row_length, @intCast(row_pitch / res.format.bytesPerPixel()));
    self.api.texSubImage2D(c.texture_2d, 0, 0, 0, @intCast(res.width), @intCast(res.height), f.format, f.kind, bytes.ptr);
    self.api.pixelStorei(c.unpack_row_length, 0);
}

/// The framebuffer that draws into or reads from this texture, made on first
/// use. Depth is attached per pass, not here.
fn framebufferOf(self: *Gl, res: *TextureRes) gt.Uint {
    if (res.framebuffer == 0) {
        self.api.genFramebuffers(1, @ptrCast(&res.framebuffer));
        self.api.bindFramebuffer(c.framebuffer, res.framebuffer);
        self.api.framebufferTexture2D(c.framebuffer, c.color_attachment0, c.texture_2d, res.name, 0);
    }
    return res.framebuffer;
}

fn readTexture(impl: backend.Impl, native: backend.Native, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const api = &self.api;

    const row = @as(usize, res.width) * 4;
    const pixels = try gpa.alloc(u8, row * res.height);
    errdefer gpa.free(pixels);

    api.bindFramebuffer(c.framebuffer, framebufferOf(self, res));
    api.readPixels(0, 0, @intCast(res.width), @intCast(res.height), c.rgba, c.unsigned_byte, pixels.ptr);

    // Bottom row first is what came back; top row first is the contract.
    var top: usize = 0;
    var bottom: usize = res.height - 1;
    while (top < bottom) : ({
        top += 1;
        bottom -= 1;
    }) {
        const a = pixels[top * row ..][0..row];
        const b = pixels[bottom * row ..][0..row];
        for (a, b) |*x, *y| std.mem.swap(u8, x, y);
    }
    return pixels;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    var name: gt.Uint = 0;
    api.genSamplers(1, @ptrCast(&name));
    api.samplerParameteri(name, c.texture_min_filter, @intCast(filterEnum(desc.min_filter)));
    api.samplerParameteri(name, c.texture_mag_filter, @intCast(filterEnum(desc.mag_filter)));
    api.samplerParameteri(name, c.texture_wrap_s, @intCast(wrapEnum(desc.wrap_u)));
    api.samplerParameteri(name, c.texture_wrap_t, @intCast(wrapEnum(desc.wrap_v)));

    res.* = .{ .name = name };
    return res;
}

fn filterEnum(filter: types.Filter) gt.Enum {
    return switch (filter) {
        .nearest => c.nearest,
        .linear => c.linear,
    };
}

fn wrapEnum(wrap: types.Wrap) gt.Enum {
    return switch (wrap) {
        .repeat => c.repeat,
        .clamp_to_edge => c.clamp_to_edge,
        .mirror => c.mirrored_repeat,
    };
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    self.api.deleteSamplers(1, @ptrCast(&res.name));
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Shaders and pipelines
// -------------------------------------------------------------------------

fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const sources = desc.glsl orelse {
        log.writeAll("fluxion-rhi: the OpenGL backend needs `ShaderDesc.glsl`, and none was given") catch {};
        return error.ShaderFailed;
    };

    const vertex = try compileStage(api, c.vertex_shader, sources.vertex, log);
    defer api.deleteShader(vertex);
    const fragment = try compileStage(api, c.fragment_shader, sources.fragment, log);
    defer api.deleteShader(fragment);

    const program = api.createProgram();
    errdefer api.deleteProgram(program);
    api.attachShader(program, vertex);
    api.attachShader(program, fragment);
    api.linkProgram(program);

    var linked: [1]gt.Int = .{0};
    api.getProgramiv(program, c.link_status, &linked);
    if (linked[0] == 0) {
        var text: [4096]gt.Char = undefined;
        var length: gt.Sizei = 0;
        api.getProgramInfoLog(program, text.len, &length, &text);
        log.print("the program did not link:\n{s}", .{text[0..@intCast(length)]}) catch {};
        return error.ShaderFailed;
    }

    const res = try self.gpa.create(ShaderRes);
    res.* = .{ .program = program };
    return res;
}

fn compileStage(api: *const opengl.Gl, kind: gt.Enum, source: [:0]const u8, log: *Io.Writer) Error!gt.Uint {
    const name = api.createShader(kind);
    errdefer api.deleteShader(name);

    api.shaderSource(name, 1, &.{source.ptr}, null);
    api.compileShader(name);

    var compiled: [1]gt.Int = .{0};
    api.getShaderiv(name, c.compile_status, &compiled);
    if (compiled[0] == 0) {
        var text: [4096]gt.Char = undefined;
        var length: gt.Sizei = 0;
        api.getShaderInfoLog(name, text.len, &length, &text);
        log.print("the {s} shader did not compile:\n{s}", .{
            if (kind == c.vertex_shader) "vertex" else "fragment",
            text[0..@intCast(length)],
        }) catch {};
        return error.ShaderFailed;
    }
    return name;
}

fn destroyShader(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(ShaderRes, native);
    self.api.deleteProgram(res.program);
    self.gpa.destroy(res);
}

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;
    const program = as(ShaderRes, shader).program;

    const res = try self.gpa.create(PipelineRes);
    errdefer self.gpa.destroy(res);
    const attributes = try self.gpa.dupe(types.VertexAttribute, desc.attributes);
    errdefer self.gpa.free(attributes);
    const buffers = try self.gpa.dupe(types.VertexBufferLayout, desc.buffers);
    errdefer self.gpa.free(buffers);

    if (buffers.len > max_vertex_slots) {
        log.print("fluxion-rhi: the OpenGL backend binds at most {d} vertex buffers", .{max_vertex_slots}) catch {};
        return error.PipelineFailed;
    }

    // The bindings GLSL 330 cannot state in the source, stated here once.
    api.useProgram(program);
    for (desc.uniform_blocks, 0..) |name, slot| {
        const index = api.getUniformBlockIndex(program, name.ptr);
        if (index == invalid_index) {
            log.print("uniform block `{s}` is not in the shader (or nothing reads it)", .{name}) catch {};
            return error.PipelineFailed;
        }
        api.uniformBlockBinding(program, index, @intCast(slot));
    }
    for (desc.textures, 0..) |name, slot| {
        const location = api.getUniformLocation(program, name.ptr);
        if (location < 0) {
            log.print("sampler `{s}` is not in the shader (or nothing reads it)", .{name}) catch {};
            return error.PipelineFailed;
        }
        api.uniform1i(location, @intCast(slot));
    }

    var vao: gt.Uint = 0;
    api.genVertexArrays(1, @ptrCast(&vao));

    res.* = .{
        .program = program,
        .vao = vao,
        .attributes = attributes,
        .buffers = buffers,
        .topology = switch (desc.topology) {
            .triangles => c.triangles,
            .triangle_strip => c.triangle_strip,
            .lines => c.lines,
            .line_strip => c.line_strip,
            .points => c.points,
        },
        .blend = desc.blend,
        .depth = desc.depth,
        .cull = desc.cull,
        .front_face = desc.front_face,
    };
    return res;
}

/// `GL_INVALID_INDEX`.
const invalid_index: gt.Uint = 0xFFFFFFFF;

fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(PipelineRes, native);
    self.api.deleteVertexArrays(1, @ptrCast(&res.vao));
    self.gpa.free(res.attributes);
    self.gpa.free(res.buffers);
    if (self.pipeline == res) self.pipeline = null;
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Surfaces
// -------------------------------------------------------------------------

fn createSurface(impl: backend.Impl, desc: types.SurfaceDesc) Error!backend.Native {
    _ = desc;
    const self = cast(impl);
    // The default framebuffer. A second one would be a second context, which
    // is a thing the hooks do not describe.
    if (self.surface.claimed) return error.Unsupported;
    self.surface.claimed = true;
    return &self.surface;
}

fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    _ = native;
    cast(impl).surface.claimed = false;
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    // The window system resized the default framebuffer already; the hooks
    // report the new size.
    _ = impl;
    _ = native;
    _ = width;
    _ = height;
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = native;
    const self = cast(impl);
    return self.hooks.framebuffer_size(self.hooks.context);
}

fn present(impl: backend.Impl, native: backend.Native, vsync: bool) Error!void {
    _ = native;
    _ = vsync; // the swap interval is the context's; see `Window.setSwapInterval`
    const self = cast(impl);
    self.hooks.swap_buffers(self.hooks.context);
}

// -------------------------------------------------------------------------
// Submitting
// -------------------------------------------------------------------------

fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) Error!void {
    const self = cast(impl);
    const api = &self.api;

    for (list) |command| {
        switch (command) {
            .begin_pass => |pass| try beginPass(self, device, pass),
            .end_pass => {
                api.bindFramebuffer(c.framebuffer, 0);
                self.pipeline = null;
            },
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                bindPipeline(self, res);
            },
            .set_viewport => |v| {
                // Flipped: the top-left rectangle, measured from the bottom.
                const y = @as(f32, @floatFromInt(self.target_height)) - v.y - v.height;
                api.viewport(@intFromFloat(v.x), @intFromFloat(y), @intFromFloat(v.width), @intFromFloat(v.height));
                api.depthRange(v.min_depth, v.max_depth);
            },
            .set_scissor => |maybe| if (maybe) |r| {
                const y = @as(i32, @intCast(self.target_height)) - r.y - @as(i32, @intCast(r.height));
                api.scissor(r.x, y, @intCast(r.width), @intCast(r.height));
            } else {
                api.scissor(0, 0, std.math.maxInt(gt.Sizei), std.math.maxInt(gt.Sizei));
            },
            .set_vertex_buffer => |b| {
                if (b.slot >= max_vertex_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.vertex_bindings[b.slot] = .{ .name = res.name, .offset = b.offset };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.index = .{
                    .name = res.name,
                    .kind = if (b.format == .u16) c.unsigned_short else c.unsigned_int,
                    .size = b.format.size(),
                };
                self.bindings_dirty = true;
            },
            .set_uniform_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                api.bindBufferBase(c.uniform_buffer, b.slot, res.name);
            },
            .set_texture => |b| {
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                api.activeTexture(c.texture0 + b.slot);
                api.bindTexture(c.texture_2d, texture.name);
                api.bindSampler(b.slot, sampler.name);
            },
            .draw => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                flushBindings(self, pipeline);
                api.drawArraysInstanced(pipeline.topology, @intCast(d.first_vertex), @intCast(d.vertex_count), @intCast(d.instance_count));
            },
            .draw_indexed => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                flushBindings(self, pipeline);
                const index = self.index orelse return error.InvalidArgument;
                const offset = opengl.offset(@as(usize, d.first_index) * index.size);
                if (d.base_vertex == 0) {
                    api.drawElementsInstanced(pipeline.topology, @intCast(d.index_count), index.kind, offset, @intCast(d.instance_count));
                } else if (d.instance_count == 1) {
                    api.drawElementsBaseVertex(pipeline.topology, @intCast(d.index_count), index.kind, offset, d.base_vertex);
                } else {
                    // `glDrawElementsInstancedBaseVertex` is OpenGL 4.2.
                    return error.Unsupported;
                }
            },
        }
    }

    if (self.debug) {
        if (self.had_error) {
            self.had_error = false;
            return error.Failed;
        }
        if (api.checkError() != null) return error.Failed;
    }
}

fn beginPass(self: *Gl, device: *Device, pass: types.RenderPassDesc) Error!void {
    const api = &self.api;

    var width: u32 = 0;
    var height: u32 = 0;
    switch (pass.color.target) {
        .surface => {
            api.bindFramebuffer(c.framebuffer, 0);
            const size = self.hooks.framebuffer_size(self.hooks.context);
            width = size[0];
            height = size[1];
        },
        .texture => |h| {
            const res = as(TextureRes, device.textures.get(h).?.native);
            api.bindFramebuffer(c.framebuffer, framebufferOf(self, res));
            width = res.width;
            height = res.height;
            // Depth is attached to the colour texture's framebuffer for this
            // pass, and detached after, so two passes into the same texture
            // with different depth buffers do not see each other's.
            if (pass.depth) |depth| {
                const d = as(TextureRes, device.textures.get(depth.texture).?.native);
                const attachment: gt.Enum = if (d.format.hasStencil()) c.depth_stencil_attachment else c.depth_attachment;
                api.framebufferTexture2D(c.framebuffer, attachment, c.texture_2d, d.name, 0);
            } else {
                api.framebufferTexture2D(c.framebuffer, c.depth_stencil_attachment, c.texture_2d, 0, 0);
            }
            if (api.checkFramebufferStatus(c.framebuffer) != c.framebuffer_complete) return error.PipelineFailed;
        },
    }
    self.target_height = height;
    self.pipeline = null;
    self.bindings_dirty = true;

    api.viewport(0, 0, @intCast(width), @intCast(height));
    api.scissor(0, 0, @intCast(width), @intCast(height));

    // A clear goes through the write masks, so they are opened first and the
    // pipeline that follows sets them back.
    var mask: gt.Bitfield = 0;
    if (pass.color.load == .clear) {
        api.colorMask(gt.gl_true, gt.gl_true, gt.gl_true, gt.gl_true);
        const col = pass.color.clear_color;
        api.clearColor(col[0], col[1], col[2], col[3]);
        mask |= c.color_buffer_bit;
    }
    if (pass.depth) |depth| if (depth.load == .clear) {
        api.depthMask(gt.gl_true);
        api.clearDepth(depth.clear_depth);
        api.clearStencil(depth.clear_stencil);
        mask |= c.depth_buffer_bit | c.stencil_buffer_bit;
    };
    if (mask != 0) api.clear(mask);
}

fn bindPipeline(self: *Gl, res: *PipelineRes) void {
    const api = &self.api;
    self.pipeline = res;
    self.bindings_dirty = true;

    api.useProgram(res.program);
    api.bindVertexArray(res.vao);

    if (res.blend.enabled) {
        api.enable(c.blend);
        api.blendFuncSeparate(factor(res.blend.src_rgb), factor(res.blend.dst_rgb), factor(res.blend.src_alpha), factor(res.blend.dst_alpha));
        api.blendEquationSeparate(equation(res.blend.op_rgb), equation(res.blend.op_alpha));
    } else {
        api.disable(c.blend);
    }

    if (res.depth.test_enabled) {
        api.enable(c.depth_test);
        api.depthFunc(compare(res.depth.compare));
    } else {
        api.disable(c.depth_test);
    }
    api.depthMask(if (res.depth.write) gt.gl_true else gt.gl_false);

    switch (res.cull) {
        .none => api.disable(c.cull_face),
        .back => {
            api.enable(c.cull_face);
            api.cullFace(c.back);
        },
        .front => {
            api.enable(c.cull_face);
            api.cullFace(c.front);
        },
    }
    api.frontFace(if (res.front_face == .ccw) c.ccw else c.cw);
}

/// Point the pipeline's attributes at whatever buffers are bound now. Done at
/// the draw, because a vertex array object remembers pointers and not slots.
fn flushBindings(self: *Gl, pipeline: *PipelineRes) void {
    if (!self.bindings_dirty) return;
    self.bindings_dirty = false;
    const api = &self.api;

    for (pipeline.attributes) |attribute| {
        const binding = self.vertex_bindings[attribute.buffer];
        const layout = pipeline.buffers[attribute.buffer];
        api.bindBuffer(c.array_buffer, binding.name);
        const pointer = opengl.offset(@as(usize, binding.offset) + attribute.offset);
        const comps: gt.Int = @intCast(attribute.format.components());
        switch (attribute.format) {
            .float, .float2, .float3, .float4 => api.vertexAttribPointer(attribute.location, comps, c.float, gt.gl_false, @intCast(layout.stride), pointer),
            .ubyte4_norm => api.vertexAttribPointer(attribute.location, comps, c.unsigned_byte, gt.gl_true, @intCast(layout.stride), pointer),
            .ubyte4 => api.vertexAttribIPointer(attribute.location, comps, c.unsigned_byte, @intCast(layout.stride), pointer),
            .uint => api.vertexAttribIPointer(attribute.location, comps, c.unsigned_int, @intCast(layout.stride), pointer),
            .int => api.vertexAttribIPointer(attribute.location, comps, c.int, @intCast(layout.stride), pointer),
        }
        api.enableVertexAttribArray(attribute.location);
        api.vertexAttribDivisor(attribute.location, if (layout.step == .instance) 1 else 0);
    }
    if (self.index) |index| api.bindBuffer(c.element_array_buffer, index.name);
}

fn factor(f: types.BlendFactor) gt.Enum {
    return switch (f) {
        .zero => c.zero,
        .one => c.one,
        .src_color => c.src_color,
        .one_minus_src_color => c.one_minus_src_color,
        .src_alpha => c.src_alpha,
        .one_minus_src_alpha => c.one_minus_src_alpha,
        .dst_color => c.dst_color,
        .one_minus_dst_color => c.one_minus_dst_color,
        .dst_alpha => c.dst_alpha,
        .one_minus_dst_alpha => c.one_minus_dst_alpha,
    };
}

fn equation(op: types.BlendOp) gt.Enum {
    return switch (op) {
        .add => c.func_add,
        .subtract => c.func_subtract,
        .reverse_subtract => c.func_reverse_subtract,
        .min => c.min,
        .max => c.max,
    };
}

fn compare(f: types.CompareFn) gt.Enum {
    return switch (f) {
        .never => c.never,
        .less => c.less,
        .equal => c.equal,
        .less_equal => c.lequal,
        .greater => c.greater,
        .not_equal => c.notequal,
        .greater_equal => c.gequal,
        .always => c.always,
    };
}

// -------------------------------------------------------------------------
// Tests. Anything that needs a context lives in the examples, which have a
// window; what is here is the arithmetic.
// -------------------------------------------------------------------------

test "every format has a GL spelling" {
    for (std.enums.values(types.Format)) |format| {
        const f = glFormat(format);
        try std.testing.expect(f.internal != 0);
        try std.testing.expect(f.format != 0);
    }
}

test "a device with no hooks is refused, not opened" {
    try std.testing.expectError(error.NoDevice, open(std.testing.allocator, .{ .backend = .gl }));
}
