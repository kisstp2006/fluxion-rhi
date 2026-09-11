// SPDX-License-Identifier: BSD-2-Clause

//! The WebGL 2 backend.
//!
//! OpenGL ES 3.0 in a browser, through `fluxion-webgl`. The model is the
//! OpenGL backend's - a vertex array per pipeline, blocks and samplers bound
//! by name once when the pipeline is made, every rectangle flipped to the top
//! left - and most of this file is that one with the calls spelled the way
//! WebGL spells them. What is here rather than there is what WebGL leaves out.
//!
//! **There is no context to make and no hooks to take.** The page made the
//! canvas and its context before the module ran, and `fluxion-webgl`'s
//! imports are that context. So `DeviceDesc.gl` means nothing here, and the
//! one surface there can be is the canvas: its size is the drawing buffer's,
//! the page resizes it, and presenting is returning - the browser shows what
//! was drawn when the frame callback that drew it ends.
//!
//! **No base vertex.** WebGL 2 has no `drawElementsBaseVertex` at all, so an
//! indexed draw with one moves the per-vertex attribute pointers that many
//! vertices on instead. The pointers are specified at the draw in any case,
//! which makes it free while the base vertex stays put - and unlike OpenGL
//! 3.3, it combines with instancing.
//!
//! **No BGRA.** WebGL takes no `bgra` upload, so a `bgra8_unorm` texture is
//! stored as RGBA with red and blue swapped on the way in. A readback is RGBA
//! whatever the texture was, which is the contract.
//!
//! **No debug output.** There is no callback to hand the driver; with
//! `DeviceDesc.debug` the error queue is drained after every submit instead.
//!
//! It builds for every target - against `fluxion-webgl`'s stub off wasm - and
//! `Device` opens it off wasm only under test. The suite at the bottom checks
//! the bookkeeping on a machine with no browser: what was bound where, what
//! was given back, which way each rectangle was flipped. Whether the picture
//! is right is `examples/web.zig`'s question, asked of a real browser.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const webgl = @import("fluxion_webgl");
const c = webgl.enums;
const Enum = webgl.types.Enum;

const types = @import("../types.zig");
const backend = @import("../backend.zig");
const commands = @import("../commands.zig");
const Device = @import("../Device.zig");

const Error = backend.Error;

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const WebGl = struct {
    gpa: Allocator,
    gl: webgl.Context,
    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,
    /// The one surface there is, the canvas. Made once, handed back on
    /// every `createSurface`.
    surface: SurfaceRes = .{},

    // Per-submit state. Reset at every pass.
    pipeline: ?*PipelineRes = null,
    target_width: u32 = 0,
    target_height: u32 = 0,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
    /// The base vertex the attribute pointers were last specified for.
    flushed_base_vertex: i32 = 0,
    index: ?IndexBinding = null,
};

const max_vertex_slots = 8;

const VertexBinding = struct {
    buffer: webgl.Buffer = .none,
    offset: u32 = 0,
};

const IndexBinding = struct {
    buffer: webgl.Buffer,
    kind: Enum,
    size: u32,
};

// -------------------------------------------------------------------------
// Resources
// -------------------------------------------------------------------------

const BufferRes = struct {
    buffer: webgl.Buffer,
    target: Enum,
    size: usize,
};

const TextureRes = struct {
    texture: webgl.Texture,
    /// Made the first time the texture is drawn into or read back.
    framebuffer: webgl.Framebuffer = .none,
    width: u32,
    height: u32,
    format: types.Format,
};

const SamplerRes = struct {
    sampler: webgl.Sampler,
};

const ShaderRes = struct {
    program: webgl.Program,
};

const PipelineRes = struct {
    program: webgl.Program,
    vao: webgl.VertexArray,
    attributes: []types.VertexAttribute,
    buffers: []types.VertexBufferLayout,
    topology: Enum,
    blend: types.BlendState,
    depth: types.DepthState,
    cull: types.CullMode,
    front_face: types.FrontFace,
};

const SurfaceRes = struct {
    /// A surface is the canvas's drawing buffer; there is nothing to store
    /// but the fact that it has been handed out.
    claimed: bool = false,
};

// -------------------------------------------------------------------------
// Opening
// -------------------------------------------------------------------------

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!struct { backend.Impl, *const backend.Vtable } {
    const gl: webgl.Context = .init();
    // Everything below is WebGL 2: vertex arrays, instancing, uniform blocks
    // and samplers are all core there, and the last two are in WebGL 1 in no
    // form at all. A page that got a WebGL 1 context has no device here.
    if (!gl.version.atLeast(.webgl2)) return error.NoDevice;

    const self = try gpa.create(WebGl);
    self.* = .{ .gpa = gpa, .gl = gl, .debug = desc.debug };
    self.renderer_len = gl.string(c.renderer, &self.renderer).len;

    // State that never changes under this backend.
    gl.pixelStorei(c.pack_alignment, 1);
    gl.pixelStorei(c.unpack_alignment, 1);
    gl.enable(c.scissor_test);

    return .{ self, &vtable };
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

fn cast(impl: backend.Impl) *WebGl {
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
    return .{ .backend = .webgl, .renderer = self.renderer[0..self.renderer_len] };
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) Error!backend.Native {
    const self = cast(impl);
    const gl = self.gl;

    const res = try self.gpa.create(BufferRes);
    errdefer self.gpa.destroy(res);

    const target: Enum = switch (desc.kind) {
        .vertex => c.array_buffer,
        .index => c.element_array_buffer,
        .uniform => c.uniform_buffer,
    };
    // A uniform buffer is rounded up to sixteen, as the std140 rules and
    // Direct3D both want; the size a program sees stays what it asked for.
    const size = if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 16) else desc.size;

    const buffer = gl.createBuffer() catch return error.OutOfMemory;
    // An element buffer bound while a vertex array is bound becomes that
    // array's, so none is bound for the making of one.
    if (target == c.element_array_buffer) gl.bindVertexArray(.none);
    gl.bindBuffer(target, buffer);
    const usage: Enum = if (desc.dynamic or desc.kind == .uniform) c.dynamic_draw else c.static_draw;
    if (desc.data) |data| {
        if (data.len == size) {
            gl.bufferData(target, data, usage);
        } else {
            gl.bufferDataSize(target, size, usage);
            gl.bufferSubData(target, 0, data);
        }
    } else {
        gl.bufferDataSize(target, size, usage);
    }

    res.* = .{ .buffer = buffer, .target = target, .size = size };
    return res;
}

fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    self.gl.deleteBuffer(res.buffer);
    self.gpa.destroy(res);
}

fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) Error!void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    // As in `createBuffer`: an update must not leave a pipeline's vertex
    // array pointing at the element buffer it happened to be bound for.
    if (res.target == c.element_array_buffer) self.gl.bindVertexArray(.none);
    self.gl.bindBuffer(res.target, res.buffer);
    self.gl.bufferSubData(res.target, offset, bytes);
    self.bindings_dirty = true;
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

const GlFormat = struct {
    internal: Enum,
    format: Enum,
    kind: Enum,
};

fn glFormat(format: types.Format) GlFormat {
    return switch (format) {
        .rgba8_unorm => .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte },
        .rgba8_unorm_srgb => .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte },
        // Stored as RGBA: WebGL has no BGRA upload. See `writeTexels`.
        .bgra8_unorm => .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte },
        .r8_unorm => .{ .internal = c.r8, .format = c.red, .kind = c.unsigned_byte },
        .rgba16_float => .{ .internal = c.rgba16f, .format = c.rgba, .kind = c.half_float },
        .rgba32_float => .{ .internal = c.rgba32f, .format = c.rgba, .kind = c.float },
        .depth24_stencil8 => .{ .internal = c.depth24_stencil8, .format = c.depth_stencil, .kind = c.unsigned_int_24_8 },
        .depth32_float => .{ .internal = c.depth_component32f, .format = c.depth_component, .kind = c.float },
    };
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    const gl = self.gl;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const texture = gl.createTexture() catch return error.OutOfMemory;
    errdefer gl.deleteTexture(texture);
    gl.bindTexture(c.texture_2d, texture);

    res.* = .{ .texture = texture, .width = desc.width, .height = desc.height, .format = desc.format };
    try writeTexels(self, res, desc.data, desc.effectiveRowPitch(), .allocate);

    // One level and no mipmaps, so say so: the default minification filter
    // wants a complete chain and samples black without one.
    gl.texParameteri(c.texture_2d, c.texture_max_level, 0);
    gl.texParameteri(c.texture_2d, c.texture_min_filter, @intCast(c.linear));
    gl.texParameteri(c.texture_2d, c.texture_mag_filter, @intCast(c.linear));
    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (res.framebuffer != .none) self.gl.deleteFramebuffer(res.framebuffer);
    self.gl.deleteTexture(res.texture);
    self.gpa.destroy(res);
}

fn updateTexture(impl: backend.Impl, native: backend.Native, bytes: []const u8, row_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    self.gl.bindTexture(c.texture_2d, res.texture);
    try writeTexels(self, res, bytes, row_pitch, .replace);
}

/// Put the whole of a texture's texels in, from `data` with its rows
/// `row_pitch` bytes apart - or, with no data, make the storage and leave it
/// empty, which is what a render target wants. The texture is bound.
fn writeTexels(
    self: *WebGl,
    res: *TextureRes,
    data: ?[]const u8,
    row_pitch: usize,
    how: enum { allocate, replace },
) Error!void {
    const gl = self.gl;
    const f = glFormat(res.format);
    var image: webgl.Context.Image = .{
        .internal_format = f.internal,
        .width = @intCast(res.width),
        .height = @intCast(res.height),
        .format = f.format,
        .kind = f.kind,
    };

    var swapped: ?[]u8 = null;
    defer if (swapped) |bytes| self.gpa.free(bytes);
    var row_length: i32 = 0;

    if (data) |bytes| {
        if (res.format == .bgra8_unorm) {
            // Swapped into a copy, tightly packed, so no row length either.
            swapped = try bgraToRgba(self.gpa, bytes, res.width, res.height, row_pitch);
            image.pixels = swapped;
        } else {
            row_length = @intCast(row_pitch / res.format.bytesPerPixel());
            image.pixels = bytes;
        }
    }

    gl.pixelStorei(c.unpack_row_length, row_length);
    defer gl.pixelStorei(c.unpack_row_length, 0);
    switch (how) {
        .allocate => gl.texImage2D(image),
        .replace => gl.texSubImage2D(0, 0, image),
    }
}

/// BGRA rows, `row_pitch` bytes apart, as tightly packed RGBA: red and blue
/// change places in every texel. The copy is the caller's.
fn bgraToRgba(gpa: Allocator, bytes: []const u8, width: u32, height: u32, row_pitch: usize) Allocator.Error![]u8 {
    const row = @as(usize, width) * 4;
    const out = try gpa.alloc(u8, row * height);
    for (0..height) |y| {
        const from = bytes[y * row_pitch ..][0..row];
        const to = out[y * row ..][0..row];
        var x: usize = 0;
        while (x < row) : (x += 4) {
            to[x + 0] = from[x + 2];
            to[x + 1] = from[x + 1];
            to[x + 2] = from[x + 0];
            to[x + 3] = from[x + 3];
        }
    }
    return out;
}

/// The framebuffer that draws into or reads from this texture, made on first
/// use. Depth is attached per pass, not here.
fn framebufferOf(self: *WebGl, res: *TextureRes) Error!webgl.Framebuffer {
    if (res.framebuffer == .none) {
        res.framebuffer = self.gl.createFramebuffer() catch return error.OutOfMemory;
        self.gl.bindFramebuffer(c.framebuffer, res.framebuffer);
        self.gl.framebufferTexture2D(c.framebuffer, c.color_attachment0, c.texture_2d, res.texture, 0);
    }
    return res.framebuffer;
}

fn readTexture(impl: backend.Impl, native: backend.Native, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const gl = self.gl;

    const row = @as(usize, res.width) * 4;
    const pixels = try gpa.alloc(u8, row * res.height);
    errdefer gpa.free(pixels);

    gl.bindFramebuffer(c.framebuffer, try framebufferOf(self, res));
    gl.readPixels(0, 0, @intCast(res.width), @intCast(res.height), c.rgba, c.unsigned_byte, pixels);
    gl.bindFramebuffer(c.framebuffer, .none);

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
    const gl = self.gl;

    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    const sampler = gl.createSampler() catch return error.OutOfMemory;
    gl.samplerParameteri(sampler, c.texture_min_filter, @intCast(filterEnum(desc.min_filter)));
    gl.samplerParameteri(sampler, c.texture_mag_filter, @intCast(filterEnum(desc.mag_filter)));
    gl.samplerParameteri(sampler, c.texture_wrap_s, @intCast(wrapEnum(desc.wrap_u)));
    gl.samplerParameteri(sampler, c.texture_wrap_t, @intCast(wrapEnum(desc.wrap_v)));

    res.* = .{ .sampler = sampler };
    return res;
}

fn filterEnum(filter: types.Filter) Enum {
    return switch (filter) {
        .nearest => c.nearest,
        .linear => c.linear,
    };
}

fn wrapEnum(wrap: types.Wrap) Enum {
    return switch (wrap) {
        .repeat => c.repeat,
        .clamp_to_edge => c.clamp_to_edge,
        .mirror => c.mirrored_repeat,
    };
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    self.gl.deleteSampler(res.sampler);
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Shaders and pipelines
// -------------------------------------------------------------------------

fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);

    const sources = desc.glsl_es orelse {
        log.writeAll("fluxion-rhi: the WebGL backend needs `ShaderDesc.glsl_es`, and none was given") catch {};
        return error.ShaderFailed;
    };

    const res = try self.gpa.create(ShaderRes);
    errdefer self.gpa.destroy(res);

    // Both stages and the link, with the driver's own words in `log` and the
    // stage they came from after them.
    const program = self.gl.buildProgram(sources.vertex, sources.fragment, log) catch |err| return switch (err) {
        error.OutOfObjects => error.OutOfMemory,
        else => error.ShaderFailed,
    };

    res.* = .{ .program = program };
    return res;
}

fn destroyShader(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(ShaderRes, native);
    self.gl.deleteProgram(res.program);
    self.gpa.destroy(res);
}

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const gl = self.gl;
    const program = as(ShaderRes, shader).program;

    if (desc.buffers.len > max_vertex_slots) {
        log.print("fluxion-rhi: the WebGL backend binds at most {d} vertex buffers", .{max_vertex_slots}) catch {};
        return error.PipelineFailed;
    }

    const res = try self.gpa.create(PipelineRes);
    errdefer self.gpa.destroy(res);
    const attributes = try self.gpa.dupe(types.VertexAttribute, desc.attributes);
    errdefer self.gpa.free(attributes);
    const buffers = try self.gpa.dupe(types.VertexBufferLayout, desc.buffers);
    errdefer self.gpa.free(buffers);

    // The bindings GLSL ES 3.00 cannot state in the source, stated here once.
    // The program remembers them, so a pipeline sharing the shader restates
    // the same ones.
    gl.useProgram(program);
    for (desc.uniform_blocks, 0..) |name, slot| {
        const index = gl.uniformBlockIndex(program, name) orelse {
            log.print("uniform block `{s}` is not in the shader (or nothing reads it)", .{name}) catch {};
            return error.PipelineFailed;
        };
        gl.uniformBlockBinding(program, index, @intCast(slot));
    }
    for (desc.textures, 0..) |name, slot| {
        const location = gl.uniformLocation(program, name);
        if (!location.valid()) {
            log.print("sampler `{s}` is not in the shader (or nothing reads it)", .{name}) catch {};
            return error.PipelineFailed;
        }
        gl.uniform1i(location, @intCast(slot));
    }

    const vao = gl.createVertexArray() catch return error.OutOfMemory;

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

fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(PipelineRes, native);
    self.gl.deleteVertexArray(res.vao);
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
    // The canvas the page made the context on. A second one would be a
    // second context, and the page made one.
    if (self.surface.claimed) return error.Unsupported;
    self.surface.claimed = true;
    return &self.surface;
}

fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    _ = native;
    cast(impl).surface.claimed = false;
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    // The page resized the canvas already - the glue matches it to the
    // element before every frame - and `canvasSize` reports what it did.
    _ = impl;
    _ = native;
    _ = width;
    _ = height;
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    _ = native;
    return canvasSize();
}

fn canvasSize() [2]u32 {
    const size = webgl.canvasSize();
    return .{ @intCast(@max(size.width, 0)), @intCast(@max(size.height, 0)) };
}

fn present(impl: backend.Impl, native: backend.Native, vsync: bool) Error!void {
    // Nothing to do. The browser shows the drawing buffer when the frame
    // callback that drew into it returns, at the display's own rate - which
    // is the only vsync a page has.
    _ = impl;
    _ = native;
    _ = vsync;
}

// -------------------------------------------------------------------------
// Submitting
// -------------------------------------------------------------------------

fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) Error!void {
    const self = cast(impl);
    const gl = self.gl;

    for (list) |command| {
        switch (command) {
            .begin_pass => |pass| try beginPass(self, device, pass),
            .end_pass => {
                gl.bindFramebuffer(c.framebuffer, .none);
                self.pipeline = null;
            },
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                bindPipeline(self, res);
            },
            .set_viewport => |v| {
                // Flipped: the top-left rectangle, measured from the bottom.
                const y = @as(f32, @floatFromInt(self.target_height)) - v.y - v.height;
                gl.viewport(@intFromFloat(v.x), @intFromFloat(y), @intFromFloat(v.width), @intFromFloat(v.height));
                gl.depthRange(v.min_depth, v.max_depth);
            },
            .set_scissor => |maybe| if (maybe) |r| {
                const y = @as(i32, @intCast(self.target_height)) - r.y - @as(i32, @intCast(r.height));
                gl.scissor(r.x, y, @intCast(r.width), @intCast(r.height));
            } else {
                gl.scissor(0, 0, @intCast(self.target_width), @intCast(self.target_height));
            },
            .set_vertex_buffer => |b| {
                if (b.slot >= max_vertex_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.vertex_bindings[b.slot] = .{ .buffer = res.buffer, .offset = b.offset };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.index = .{
                    .buffer = res.buffer,
                    .kind = if (b.format == .u16) c.unsigned_short else c.unsigned_int,
                    .size = b.format.size(),
                };
                self.bindings_dirty = true;
            },
            .set_uniform_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                gl.bindBufferBase(c.uniform_buffer, b.slot, res.buffer);
            },
            .set_texture => |b| {
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                gl.activeTexture(c.textureUnit(b.slot));
                gl.bindTexture(c.texture_2d, texture.texture);
                gl.bindSampler(b.slot, sampler.sampler);
            },
            .draw => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                try flushBindings(self, pipeline, 0);
                gl.drawArraysInstanced(pipeline.topology, @intCast(d.first_vertex), @intCast(d.vertex_count), @intCast(d.instance_count));
            },
            .draw_indexed => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                try flushBindings(self, pipeline, d.base_vertex);
                const index = self.index orelse return error.InvalidArgument;
                // In bytes, which is how WebGL counts an offset into an
                // element buffer.
                const offset: i32 = @intCast(@as(usize, d.first_index) * index.size);
                gl.drawElementsInstanced(pipeline.topology, @intCast(d.index_count), index.kind, offset, @intCast(d.instance_count));
            },
        }
    }

    if (self.debug and gl.checkError() != null) return error.Failed;
}

fn beginPass(self: *WebGl, device: *Device, pass: types.RenderPassDesc) Error!void {
    const gl = self.gl;

    var size: [2]u32 = undefined;
    switch (pass.color.target) {
        .surface => {
            gl.bindFramebuffer(c.framebuffer, .none);
            size = canvasSize();
        },
        .texture => |h| {
            const res = as(TextureRes, device.textures.get(h).?.native);
            gl.bindFramebuffer(c.framebuffer, try framebufferOf(self, res));
            size = .{ res.width, res.height };
            // Depth is attached to the colour texture's framebuffer for this
            // pass, and detached after, so two passes into the same texture
            // with different depth buffers do not see each other's.
            if (pass.depth) |depth| {
                const d = as(TextureRes, device.textures.get(depth.texture).?.native);
                const attachment: Enum = if (d.format.hasStencil()) c.depth_stencil_attachment else c.depth_attachment;
                gl.framebufferTexture2D(c.framebuffer, attachment, c.texture_2d, d.texture, 0);
            } else {
                gl.framebufferTexture2D(c.framebuffer, c.depth_stencil_attachment, c.texture_2d, .none, 0);
            }
            gl.checkFramebuffer(c.framebuffer) catch return error.PipelineFailed;
        },
    }
    self.target_width = size[0];
    self.target_height = size[1];
    self.pipeline = null;
    self.bindings_dirty = true;

    // The whole attachment, at the whole depth range: the state a pass
    // begins in, whatever the last one left.
    gl.viewport(0, 0, @intCast(size[0]), @intCast(size[1]));
    gl.depthRange(0, 1);
    gl.scissor(0, 0, @intCast(size[0]), @intCast(size[1]));

    // A clear goes through the write masks, so they are opened first and the
    // pipeline that follows sets them back.
    var mask: u32 = 0;
    if (pass.color.load == .clear) {
        gl.colorMask(true, true, true, true);
        const col = pass.color.clear_color;
        gl.clearColor(col[0], col[1], col[2], col[3]);
        mask |= c.color_buffer_bit;
    }
    if (pass.depth) |depth| if (depth.load == .clear) {
        gl.depthMask(true);
        gl.clearDepth(depth.clear_depth);
        gl.clearStencil(depth.clear_stencil);
        mask |= c.depth_buffer_bit | c.stencil_buffer_bit;
    };
    if (mask != 0) gl.clear(mask);
}

fn bindPipeline(self: *WebGl, res: *PipelineRes) void {
    const gl = self.gl;
    self.pipeline = res;
    self.bindings_dirty = true;

    gl.useProgram(res.program);
    gl.bindVertexArray(res.vao);

    if (res.blend.enabled) {
        gl.enable(c.blend);
        gl.blendFuncSeparate(factor(res.blend.src_rgb), factor(res.blend.dst_rgb), factor(res.blend.src_alpha), factor(res.blend.dst_alpha));
        gl.blendEquationSeparate(equation(res.blend.op_rgb), equation(res.blend.op_alpha));
    } else {
        gl.disable(c.blend);
    }

    if (res.depth.test_enabled) {
        gl.enable(c.depth_test);
        gl.depthFunc(compare(res.depth.compare));
    } else {
        gl.disable(c.depth_test);
    }
    gl.depthMask(res.depth.write);

    switch (res.cull) {
        .none => gl.disable(c.cull_face),
        .back => {
            gl.enable(c.cull_face);
            gl.cullFace(c.back);
        },
        .front => {
            gl.enable(c.cull_face);
            gl.cullFace(c.front);
        },
    }
    gl.frontFace(if (res.front_face == .ccw) c.ccw else c.cw);
}

/// Point the pipeline's attributes at whatever buffers are bound now - and,
/// for the ones that step per vertex, `base_vertex` vertices further on.
///
/// Done at the draw, because a vertex array object remembers pointers and
/// not slots. Which is also what makes the missing base vertex cheap: the
/// pointer is being written anyway, so moving it is an addition.
fn flushBindings(self: *WebGl, pipeline: *PipelineRes, base_vertex: i32) Error!void {
    if (!self.bindings_dirty and base_vertex == self.flushed_base_vertex) return;
    const gl = self.gl;

    for (pipeline.attributes) |attribute| {
        const binding = self.vertex_bindings[attribute.buffer];
        const layout = pipeline.buffers[attribute.buffer];

        const moved: i64 = if (layout.step == .vertex) @as(i64, base_vertex) * layout.stride else 0;
        const at: i64 = @as(i64, binding.offset) + attribute.offset + moved;
        // A base vertex that reaches back before the start of the buffer
        // has no pointer to become.
        if (at < 0 or at > std.math.maxInt(i32)) return error.Unsupported;
        const offset: i32 = @intCast(at);
        const stride: i32 = @intCast(layout.stride);
        const comps: i32 = @intCast(attribute.format.components());

        gl.bindBuffer(c.array_buffer, binding.buffer);
        switch (attribute.format) {
            .float, .float2, .float3, .float4 => gl.vertexAttribPointer(attribute.location, comps, c.float, false, stride, offset),
            .ubyte4_norm => gl.vertexAttribPointer(attribute.location, comps, c.unsigned_byte, true, stride, offset),
            .ubyte4 => gl.vertexAttribIPointer(attribute.location, comps, c.unsigned_byte, stride, offset),
            .uint => gl.vertexAttribIPointer(attribute.location, comps, c.unsigned_int, stride, offset),
            .int => gl.vertexAttribIPointer(attribute.location, comps, c.int, stride, offset),
        }
        gl.enableVertexAttribArray(attribute.location);
        gl.vertexAttribDivisor(attribute.location, if (layout.step == .instance) 1 else 0);
    }
    if (self.index) |index| gl.bindBuffer(c.element_array_buffer, index.buffer);

    self.bindings_dirty = false;
    self.flushed_base_vertex = base_vertex;
}

fn factor(f: types.BlendFactor) Enum {
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

fn equation(op: types.BlendOp) Enum {
    return switch (op) {
        .add => c.func_add,
        .subtract => c.func_subtract,
        .reverse_subtract => c.func_reverse_subtract,
        .min => c.min,
        .max => c.max,
    };
}

fn compare(f: types.CompareFn) Enum {
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
// Tests - against `fluxion-webgl`'s stub, which counts what it is told and
// draws none of it. Whether the picture is right is `examples/web.zig`'s to
// find out, in a browser.
// -------------------------------------------------------------------------

const testing = std.testing;
const stub = webgl.stub;

/// What the stub compiles, which is anything; these are here so a pipeline
/// has a program to be made from.
const shader_desc: types.ShaderDesc = .{ .glsl_es = .{
    .vertex = "#version 300 es\nvoid main() {}",
    .fragment = "#version 300 es\nprecision highp float;\nvoid main() {}",
} };

fn openDevice() !Device {
    stub.reset();
    return Device.init(testing.allocator, .{ .backend = .webgl });
}

fn bufferName(device: *Device, h: types.Buffer) u32 {
    return as(BufferRes, device.buffers.get(h).?.native).buffer.index();
}

fn samplerName(device: *Device, h: types.Sampler) u32 {
    return as(SamplerRes, device.samplers.get(h).?.native).sampler.index();
}

test "a device opens on a WebGL 2 context and says what it is" {
    var device = try openDevice();
    defer device.deinit();

    try testing.expectEqual(types.Backend.webgl, device.info().backend);
    try testing.expectEqualStrings("Fluxion WebGL stub", device.info().renderer);
    // OpenGL ES keeps OpenGL's clip space.
    try testing.expectEqual(types.Backend.gl.clip(), device.clip());

    // The one surface is the canvas, at the canvas's size.
    const surface = try device.createSurface(.{});
    const size = try device.surfaceSize(surface);
    try testing.expectEqual(@as(u32, 800), size.width);
    try testing.expectEqual(@as(u32, 600), size.height);
    try testing.expectError(error.Unsupported, device.createSurface(.{}));
}

test "everything made is given back" {
    var device = try openDevice();

    _ = try device.createBuffer(.{ .kind = .vertex, .size = 32, .data = &@as([32]u8, @splat(1)) });
    _ = try device.createBuffer(.{ .kind = .index, .size = 12 });
    _ = try device.createBuffer(.{ .kind = .uniform, .size = 20 });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    _ = try device.createTexture(.{ .width = 2, .height = 2, .format = .r8_unorm, .data = "abcd" });
    _ = try device.createSampler(.nearest);
    const shader = try device.createShader(shader_desc);
    _ = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .uniform_blocks = &.{"Frame"},
        .textures = &.{"atlas"},
    });
    _ = try device.createSurface(.{});

    // A texture that has been drawn into has a framebuffer as well.
    const pixels = try device.readTexture(target, testing.allocator);
    testing.allocator.free(pixels);

    try testing.expect(stub.state.live_objects > 0);
    device.deinit();
    try testing.expectEqual(0, stub.state.live_objects);
}

test "a buffer made without data is given its size and nothing else" {
    var device = try openDevice();
    defer device.deinit();

    // Rounded up to sixteen, as a uniform buffer is everywhere.
    _ = try device.createBuffer(.{ .kind = .uniform, .size = 20 });
    try testing.expectEqual(32, stub.state.last_buffer_size);
    try testing.expectEqual(0, stub.state.last_upload_len);

    // And one with all of its data is one upload.
    _ = try device.createBuffer(.{ .kind = .vertex, .size = 8, .data = "12345678" });
    try testing.expectEqual(8, stub.state.last_upload_len);
}

test "a BGRA texture goes in as RGBA" {
    var device = try openDevice();
    defer device.deinit();

    // Blue, written the BGRA way: blue first.
    _ = try device.createTexture(.{ .width = 1, .height = 1, .format = .bgra8_unorm, .data = &.{ 255, 0, 0, 255 } });
    try testing.expectEqual(c.rgba, stub.state.last_image.format);
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, stub.state.last_image.first);

    // Every other format goes as it is.
    _ = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 255, 0, 0, 255 } });
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, stub.state.last_image.first);
    _ = try device.createTexture(.{ .width = 1, .height = 1, .format = .r8_unorm, .data = "x" });
    try testing.expectEqual(c.red, stub.state.last_image.format);
}

test "a shader needs GLSL ES, and says so" {
    var device = try openDevice();
    defer device.deinit();

    try testing.expectError(error.ShaderFailed, device.createShader(.{
        .glsl = .{ .vertex = "#version 330 core\n", .fragment = "#version 330 core\n" },
    }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "glsl_es") != null);
}

test "a shader that does not compile is the driver's words, and leaves nothing" {
    var device = try openDevice();
    defer device.deinit();

    stub.state.fail_compile = true;
    try testing.expectError(error.ShaderFailed, device.createShader(shader_desc));
    try testing.expect(std.mem.startsWith(u8, device.diagnostics(), "ERROR:"));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "(in the vertex shader)") != null);
    try testing.expectEqual(0, stub.state.live_objects);
}

test "blocks and samplers are bound by name when the pipeline is made" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    _ = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .uniform_blocks = &.{ "Frame", "Light" },
        .textures = &.{"atlas"},
    });
    // The second block went to slot one.
    try testing.expectEqual(1, stub.state.last_block_binding.binding);

    // A name the linker removed is a failure here, and not zeros later.
    try testing.expectError(error.PipelineFailed, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .uniform_blocks = &.{"_gone"},
    }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "_gone") != null);
    try testing.expectError(error.PipelineFailed, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .textures = &.{"_gone"},
    }));
}

test "the rectangles count from the top left" {
    var device = try openDevice();
    defer device.deinit();

    const target = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setViewport(.{ .x = 0, .y = 0, .width = 32, .height = 16 });
    try cmd.setScissor(.{ .x = 8, .y = 4, .width = 16, .height = 8 });
    try cmd.endPass();
    try device.submit();

    // Sixteen high at the top of sixty-four is forty-eight up from the
    // bottom, which is where WebGL measures from.
    try testing.expectEqual(.{ 0, 48, 32, 16 }, stub.state.last_viewport);
    try testing.expectEqual(.{ 8, 52, 16, 8 }, stub.state.last_scissor);

    // And no scissor is the whole attachment again.
    const again = device.begin();
    try again.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try again.setScissor(null);
    try again.endPass();
    try device.submit();
    try testing.expectEqual(.{ 0, 0, 64, 64 }, stub.state.last_scissor);
}

test "a base vertex moves the pointers that step per vertex, and not the others" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    const corners = try device.createBuffer(.{ .kind = .vertex, .size = 256 });
    const placements = try device.createBuffer(.{ .kind = .vertex, .size = 256 });
    const indices = try device.createBuffer(.{ .kind = .index, .size = 64 });

    // The per-vertex attribute last, so it is the one the stub saw last.
    const moved = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 1, .format = .float4, .offset = 0, .buffer = 1 },
            .{ .location = 0, .format = .float2, .offset = 0, .buffer = 0 },
        },
        .buffers = &.{ .{ .stride = 8 }, .{ .stride = 16, .step = .instance } },
    });
    // And the per-instance one last.
    const kept = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0, .buffer = 0 },
            .{ .location = 1, .format = .float4, .offset = 0, .buffer = 1 },
        },
        .buffers = &.{ .{ .stride = 8 }, .{ .stride = 16, .step = .instance } },
    });

    const surface = try device.createSurface(.{});
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.setVertexBuffer(0, corners, 0);
    try cmd.setVertexBuffer(1, placements, 0);
    try cmd.setIndexBuffer(indices, .u16);
    try cmd.setPipeline(moved);
    try cmd.drawIndexed(.{ .index_count = 6, .first_index = 3, .base_vertex = 10, .instance_count = 2 });
    try cmd.endPass();
    try device.submit();

    // Ten vertices of eight bytes on, with instancing, which OpenGL 3.3
    // cannot do at all.
    try testing.expectEqual(80, stub.state.last_attribute.offset);
    try testing.expectEqual(2, stub.state.last_draw.instances);
    // And the first index is counted in bytes: three of two bytes each.
    try testing.expectEqual(6, stub.state.last_draw.offset);

    const again = device.begin();
    try again.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try again.setVertexBuffer(0, corners, 0);
    try again.setVertexBuffer(1, placements, 0);
    try again.setIndexBuffer(indices, .u16);
    try again.setPipeline(kept);
    try again.drawIndexed(.{ .index_count = 6, .base_vertex = 10 });
    try again.endPass();
    try device.submit();
    try testing.expectEqual(0, stub.state.last_attribute.offset);

    // A base vertex that reaches back before the buffer has no pointer to be.
    const back = device.begin();
    try back.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try back.setVertexBuffer(0, corners, 0);
    try back.setVertexBuffer(1, placements, 0);
    try back.setIndexBuffer(indices, .u16);
    try back.setPipeline(moved);
    try back.drawIndexed(.{ .index_count = 6, .base_vertex = -1 });
    try back.endPass();
    try testing.expectError(error.Unsupported, device.submit());
}

test "blending gives colour and alpha a rule each" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .blend = .alpha,
    });
    const surface = try device.createSurface(.{});
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.setPipeline(pipeline);
    try cmd.endPass();
    try device.submit();

    // Straight alpha: the colour weighted by the source's alpha, and the
    // alpha itself accumulated, so drawing onto opaque stays opaque.
    try testing.expectEqual(
        .{ c.src_alpha, c.one_minus_src_alpha, c.one, c.one_minus_src_alpha },
        stub.state.last_blend_func,
    );
}

test "a buffer and a sampler land in the slot they were set to" {
    var device = try openDevice();
    defer device.deinit();

    const frame = try device.createBuffer(.{ .kind = .uniform, .size = 64 });
    const texture = try device.createTexture(.{ .width = 4, .height = 4 });
    const sampler = try device.createSampler(.linear);
    const surface = try device.createSurface(.{});

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.setUniformBuffer(1, frame);
    try cmd.setTexture(2, texture, sampler);
    try cmd.endPass();
    try device.submit();

    try testing.expectEqual(bufferName(&device, frame), stub.state.uniform_buffers[1]);
    try testing.expectEqual(samplerName(&device, sampler), stub.state.samplers[2]);
}

test "an integer attribute is read as integers" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 2, .format = .uint, .offset = 0 }},
        .buffers = &.{.{ .stride = 4, .step = .instance }},
    });
    const cells = try device.createBuffer(.{ .kind = .vertex, .size = 16 });
    const surface = try device.createSurface(.{});

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, cells, 0);
    try cmd.draw(.{ .vertex_count = 6, .instance_count = 4 });
    try cmd.endPass();
    try device.submit();

    try testing.expect(stub.state.last_attribute.integer);
    try testing.expectEqual(c.unsigned_int, stub.state.last_attribute.kind);
    try testing.expectEqual(1, stub.state.draw_calls);
}

test "a readback is what the pass cleared to, top row first" {
    var device = try openDevice();
    defer device.deinit();

    const target = try device.createTexture(.{ .width = 4, .height = 2, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 1, 0, 0.5, 1 } } });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(@as(usize, 4 * 2 * 4), pixels.len);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 128, 255 }, pixels[0..4]);
}

test "debug drains the error queue after a submit" {
    stub.reset();
    var device = try Device.init(testing.allocator, .{ .backend = .webgl, .debug = true });
    defer device.deinit();

    const surface = try device.createSurface(.{});
    stub.state.pending_error = c.invalid_operation;
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.endPass();
    try testing.expectError(error.Failed, device.submit());
}
