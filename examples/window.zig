// SPDX-License-Identifier: BSD-2-Clause

//! A window for a device to draw into, and the device itself.
//!
//! Nothing here is part of the library. The window is `fluxion-platform`'s;
//! what this file adds is the two seams a device needs from one - a
//! `GlHooks` for the OpenGL backend, an `HWND` for Direct3D - and the shape
//! the examples want: one struct with a `pump`, and the Escape key closing it.
//!
//! ```zig
//! var window = try Window.open(.{ .backend = .gl });
//! defer window.close();
//! var device = try window.openDevice(gpa, .{});
//! defer device.deinit();
//! const surface = try window.createSurface(&device);
//! ```
//!
//! `fluxion_platform` is a lazy dependency of the examples alone; the library
//! never opens a window.

const std = @import("std");
const builtin = @import("builtin");

const rhi = @import("fluxion_rhi");
const platform = @import("fluxion_platform");

/// The backend a machine most likely has: Direct3D on Windows, OpenGL
/// elsewhere.
pub fn defaultBackend() rhi.Backend {
    return if (builtin.os.tag == .windows) .d3d11 else .gl;
}

pub fn parseBackend(text: []const u8) ?rhi.Backend {
    return std.meta.stringToEnum(rhi.Backend, text);
}

pub const Window = struct {
    inner: *Inner,
    backend: rhi.Backend,
    /// The framebuffer, in pixels.
    width: u32,
    height: u32,
    resized: bool = false,

    /// Boxed, because a `platform.Window` points at its context and the
    /// hooks point at this - neither may move.
    const Inner = struct {
        ctx: platform.Context,
        win: platform.Window,
    };

    pub const Error = platform.Error;

    pub const Options = struct {
        /// Decides whether the window comes with an OpenGL context. Has to be
        /// known here: no platform lets a window change its mind afterwards.
        backend: rhi.Backend,
        title: []const u8 = "Fluxion RHI",
        width: u32 = 960,
        height: u32 = 540,
        visible: bool = true,
    };

    pub fn open(options: Options) Error!Window {
        const gpa = std.heap.smp_allocator;

        const inner = try gpa.create(Inner);
        errdefer gpa.destroy(inner);
        inner.ctx = try platform.Context.init(gpa, .{});
        errdefer inner.ctx.deinit();

        inner.win = try inner.ctx.createWindow(.{
            .title = options.title,
            .width = options.width,
            .height = options.height,
            .visible = options.visible,
            .gl = if (options.backend == .gl) .{ .major = 3, .minor = 3, .profile = .core } else null,
        });
        errdefer inner.win.destroy();

        if (options.backend == .gl) {
            try inner.win.makeContextCurrent();
            inner.win.setSwapInterval(.vsync) catch {};
        }

        const size = inner.win.framebufferSize();
        return .{ .inner = inner, .backend = options.backend, .width = size[0], .height = size[1] };
    }

    /// Is this error the machine's answer rather than the program's fault?
    pub fn isAbsent(err: Error) bool {
        return switch (err) {
            error.Unsupported, error.NoDisplay, error.ConnectionFailed, error.WindowCreationFailed, error.Unavailable => true,
            error.OutOfMemory => false,
        };
    }

    /// What the OpenGL backend needs: the context, as four callbacks.
    pub fn hooks(self: Window) rhi.GlHooks {
        return .{
            .context = self.inner,
            .get_proc_address = getProcAddress,
            .swap_buffers = swapBuffers,
            .framebuffer_size = framebufferSize,
        };
    }

    fn getProcAddress(context: *anyopaque, name: [*:0]const u8) ?rhi.GlProc {
        const inner: *Inner = @ptrCast(@alignCast(context));
        return inner.win.getProcAddress(name);
    }

    fn swapBuffers(context: *anyopaque) void {
        const inner: *Inner = @ptrCast(@alignCast(context));
        inner.win.swapBuffers() catch {};
    }

    fn framebufferSize(context: *anyopaque) [2]u32 {
        const inner: *Inner = @ptrCast(@alignCast(context));
        return inner.win.framebufferSize();
    }

    /// The platform's own handle: an `HWND` on Windows.
    pub fn nativeHandle(self: Window) usize {
        return self.inner.win.native();
    }

    pub const DeviceOptions = struct {
        debug: bool = false,
        software: bool = false,
    };

    /// A device on this window's backend.
    pub fn openDevice(self: Window, gpa: std.mem.Allocator, options: DeviceOptions) rhi.Error!rhi.Device {
        return rhi.Device.init(gpa, .{
            .backend = switch (self.backend) {
                .gl => .gl,
                .d3d11 => .d3d11,
                .d3d12 => .d3d12,
                .none => .none,
                // A browser's, and this is a desktop window: `Device` says
                // `Unsupported`, which is the truth.
                .webgl => .webgl,
                .vulkan => .vulkan,
                // Made by `Device.initWith`, and this glue knows no such backend.
                .other => return error.Unsupported,
            },
            .gl = if (self.backend == .gl) self.hooks() else null,
            .debug = options.debug,
            .software = options.software,
        });
    }

    /// The device's surface for this window. On `.vulkan`, the `VkSurfaceKHR`
    /// is made here - `fluxion-rhi` has no windowing code of its own to make
    /// one with - from the device's own instance and this window's own
    /// `createVulkanSurface`, then handed in as `SurfaceDesc.vulkan_surface`.
    pub fn createSurface(self: Window, device: *rhi.Device) rhi.Error!rhi.Surface {
        if (device.tag == .vulkan) {
            const handles = device.vulkanInstanceHandles() orelse return error.Unsupported;
            const surface = self.inner.win.createVulkanSurface(
                handles.instance,
                @ptrCast(@alignCast(handles.get_instance_proc_addr)),
                null,
            ) catch return error.Unsupported;
            return device.createSurface(.{
                .vulkan_surface = surface,
                .width = self.width,
                .height = self.height,
            });
        }
        return device.createSurface(.{
            .native_window = self.nativeHandle(),
            .width = self.width,
            .height = self.height,
        });
    }

    /// Drain the events and answer whether the window is still there.
    pub fn pump(self: *Window) bool {
        self.inner.ctx.pump() catch return false;
        while (self.inner.ctx.poll()) |ev| switch (ev) {
            .close => self.inner.win.setShouldClose(true),
            .key => |k| if (k.key == .escape and k.action == .press) self.inner.win.setShouldClose(true),
            .framebuffer_resize => |r| {
                self.width = r.width;
                self.height = r.height;
                self.resized = true;
            },
            else => {},
        };
        return !self.inner.win.shouldClose();
    }

    /// Whether the window changed size since this was last asked.
    pub fn takeResize(self: *Window) bool {
        defer self.resized = false;
        return self.resized;
    }

    pub fn minimised(self: Window) bool {
        return self.width == 0 or self.height == 0;
    }

    pub fn close(self: *Window) void {
        self.inner.win.destroy();
        self.inner.ctx.deinit();
        std.heap.smp_allocator.destroy(self.inner);
        self.* = undefined;
    }
};

/// A hidden window with a context for the tests, or a skip where there is
/// no display.
pub fn openForTest(backend: rhi.Backend) !Window {
    return Window.open(.{ .backend = backend, .width = 64, .height = 64, .visible = false }) catch |err|
        if (Window.isAbsent(err)) error.SkipZigTest else err;
}

/// A device for the tests: GL on a hidden window, Direct3D on WARP.
pub const TestDevice = struct {
    window: ?Window,
    device: rhi.Device,

    pub fn open(backend: rhi.Backend) !TestDevice {
        switch (backend) {
            .gl => {
                var window = try openForTest(.gl);
                errdefer window.close();
                const device = window.openDevice(std.testing.allocator, .{ .debug = true }) catch |err| switch (err) {
                    error.NoDevice, error.Unsupported => return error.SkipZigTest,
                    else => return err,
                };
                return .{ .window = window, .device = device };
            },
            .d3d11 => {
                const device = rhi.Device.init(std.testing.allocator, .{ .backend = .d3d11, .software = true }) catch |err| switch (err) {
                    error.NoDevice, error.Unsupported => return error.SkipZigTest,
                    else => return err,
                };
                return .{ .window = null, .device = device };
            },
            .none => return .{ .window = null, .device = try rhi.Device.init(std.testing.allocator, .{ .backend = .none }) },
            // There is no browser here to draw in. The backend's own suite
            // runs against the stub; the picture is `examples/web.zig`'s.
            .webgl => return error.SkipZigTest,
            // No Vulkan backend exists yet.
            .vulkan => return error.SkipZigTest,
            // No Direct3D 12 backend exists yet.
            .d3d12 => return error.SkipZigTest,
            // A backend the caller supplies has no glue here.
            .other => return error.SkipZigTest,
        }
    }

    pub fn close(self: *TestDevice) void {
        self.device.deinit();
        if (self.window) |*w| w.close();
        self.* = undefined;
    }
};

// -------------------------------------------------------------------------
// Tests - the OpenGL backend, which needs a context and so lives here
// -------------------------------------------------------------------------

const testing = std.testing;

const flat_vs = @embedFile("shaders/flat.vert.glsl");
const flat_fs = @embedFile("shaders/flat.frag.glsl");

const Vertex = extern struct { position: [2]f32, colour: [4]f32 };

fn at(pixels: []const u8, width: usize, x: usize, y: usize) [4]u8 {
    return pixels[(y * width + x) * 4 ..][0..4].*;
}

test "a triangle through the OpenGL backend lands top-left up" {
    var fixture = try TestDevice.open(.gl);
    defer fixture.close();
    var device = &fixture.device;
    try testing.expectEqual(rhi.Backend.gl, device.info().backend);

    const target = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });
    const shader = device.createShader(.{ .glsl = .{ .vertex = flat_vs, .fragment = flat_fs } }) catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };
    const pipeline = device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0 },
            .{ .location = 1, .format = .float4, .offset = 8 },
        },
        .buffers = &.{.{ .stride = @sizeOf(Vertex) }},
        .uniform_blocks = &.{"Frame"},
    }) catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };

    // Red, pointing up, over the middle of a blue field: the same triangle
    // the Direct3D test draws, so the two backends are held to one answer.
    const vertices = [_]Vertex{
        .{ .position = .{ -0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.0, 0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
    const tint = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    try device.updateBuffer(tint, 0, std.mem.asBytes(&[4]f32{ 1, 1, 1, 1 }));

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.setUniformBuffer(0, tint);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);

    // Top-left origin, whatever GL thinks: the point is up, the base is down.
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, at(pixels, 64, 32, 20));
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, at(pixels, 64, 32, 40));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 64, 2, 2));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 64, 61, 61));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 64, 2, 61));
}

test "a scissor rectangle counts from the top left on OpenGL" {
    var fixture = try TestDevice.open(.gl);
    defer fixture.close();
    var device = &fixture.device;

    const target = try device.createTexture(.{ .width = 32, .height = 32, .usage = .{ .render_target = true } });
    const shader = try device.createShader(.{ .glsl = .{ .vertex = flat_vs, .fragment = flat_fs } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0 },
            .{ .location = 1, .format = .float4, .offset = 8 },
        },
        .buffers = &.{.{ .stride = @sizeOf(Vertex) }},
        .topology = .triangle_strip,
        .uniform_blocks = &.{"Frame"},
    });
    const vertices = [_]Vertex{
        .{ .position = .{ -1, -1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ -1, 1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, -1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, 1 }, .colour = .{ 1, 1, 1, 1 } },
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
    const tint = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 1, 1, 1, 1 }) });

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 0, 1 } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.setUniformBuffer(0, tint);
    // The top-left quarter only.
    try cmd.setScissor(.{ .x = 0, .y = 0, .width = 16, .height = 16 });
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual([4]u8{ 255, 255, 255, 255 }, at(pixels, 32, 4, 4));
    try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, at(pixels, 32, 4, 28));
    try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, at(pixels, 32, 28, 4));
    try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, at(pixels, 32, 28, 28));
}

test "a window's surface is where a frame ends up" {
    var fixture = try TestDevice.open(.gl);
    defer fixture.close();
    var device = &fixture.device;

    const surface = try fixture.window.?.createSurface(device);
    const size = try device.surfaceSize(surface);
    try testing.expect(size.width >= 64 and size.height >= 64);

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface }, .clear_color = .{ 0.2, 0.3, 0.4, 1 } } });
    try cmd.endPass();
    try device.submit();
    try device.present(surface);

    // A second surface is a second context, which the hooks do not describe.
    try testing.expectError(error.Unsupported, device.createSurface(.{}));
}

// -------------------------------------------------------------------------
// Tests - the OpenGL backend in three dimensions: mip chains, cubes, volumes,
// arrays, multisampling, depth, compression, and what `caps` promises.
//
// Nothing here is checked against the picture a person would see: each is a
// texture written or drawn, and read back. What a test needs that this
// machine's driver lacks it skips, by asking `caps` first, and `caps` is
// itself held to what it says by the last test of the group.
// -------------------------------------------------------------------------

/// `count` texels of one colour.
fn solid(count: usize, colour: [4]u8) ![]u8 {
    const bytes = try testing.allocator.alloc(u8, count * 4);
    for (0..count) |i| @memcpy(bytes[i * 4 ..][0..4], &colour);
    return bytes;
}

fn near(got: u8, want: u8, tolerance: u8) bool {
    return (if (got > want) got - want else want - got) <= tolerance;
}

/// Every texel of `pixels` is `colour`, give or take.
fn expectSolid(pixels: []const u8, colour: [4]u8, tolerance: u8) !void {
    try testing.expect(pixels.len % 4 == 0);
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) {
        for (colour, pixels[i..][0..4]) |want, got| {
            if (near(got, want, tolerance)) continue;
            std.debug.print("texel {d} is {any}, wanted {any}\n", .{ i / 4, pixels[i..][0..4].*, colour });
            return error.TestExpectedEqual;
        }
    }
}

fn expectTexel(pixels: []const u8, width: usize, x: usize, y: usize, colour: [4]u8, tolerance: u8) !void {
    const got = at(pixels, width, x, y);
    for (colour, got) |want, have| {
        if (near(have, want, tolerance)) continue;
        std.debug.print("texel ({d}, {d}) is {any}, wanted {any}\n", .{ x, y, got, colour });
        return error.TestExpectedEqual;
    }
}

/// One image of a texture is a `width` by `height` field of one colour.
fn expectImage(device: *rhi.Device, texture: rhi.Texture, sub: rhi.types.Subresource, width: usize, height: usize, colour: [4]u8, tolerance: u8) !void {
    const pixels = try device.readSubresource(texture, sub, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(width * height * 4, pixels.len);
    expectSolid(pixels, colour, tolerance) catch |err| {
        std.debug.print("in level {d}, layer {d}\n", .{ sub.mip, sub.layer });
        return err;
    };
}

/// A device on a hidden window, or a skip.
fn openGl() !TestDevice {
    return TestDevice.open(.gl);
}

/// The flat colour shader and what it takes: a white tint and a quad
/// that covers the target, in whatever colour is asked for.
const Flat = struct {
    shader: rhi.Shader,
    tint: rhi.Buffer,

    const Options = struct {
        color_format: ?rhi.Format = .rgba8_unorm,
        depth_format: ?rhi.Format = null,
        depth: rhi.DepthState = .none,
        samples: u32 = 1,
    };

    fn init(device: *rhi.Device) !Flat {
        return .{
            .shader = try device.createShader(.{ .glsl = .{ .vertex = flat_vs, .fragment = flat_fs } }),
            .tint = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 1, 1, 1, 1 }) }),
        };
    }

    fn pipeline(self: Flat, device: *rhi.Device, options: Options) !rhi.Pipeline {
        return device.createPipeline(.{
            .shader = self.shader,
            .attributes = &.{
                .{ .location = 0, .format = .float2, .offset = 0 },
                .{ .location = 1, .format = .float4, .offset = 8 },
            },
            .buffers = &.{.{ .stride = @sizeOf(Vertex) }},
            .topology = .triangle_strip,
            .uniform_blocks = &.{"Frame"},
            .color_format = options.color_format,
            .depth_format = options.depth_format,
            .depth = options.depth,
            .samples = options.samples,
        });
    }

    fn quad(device: *rhi.Device, colour: [4]f32) !rhi.Buffer {
        const vertices = [_]Vertex{
            .{ .position = .{ -1, -1 }, .colour = colour },
            .{ .position = .{ -1, 1 }, .colour = colour },
            .{ .position = .{ 1, -1 }, .colour = colour },
            .{ .position = .{ 1, 1 }, .colour = colour },
        };
        return device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
    }

    /// The draw itself: what a pass into an attachment does with a pipeline.
    fn draw(self: Flat, cmd: *rhi.CommandList, pipe: rhi.Pipeline, vertices: rhi.Buffer, count: u32) !void {
        try cmd.setPipeline(pipe);
        try cmd.setVertexBuffer(0, vertices, 0);
        try cmd.setUniformBuffer(0, self.tint);
        try cmd.draw(.{ .vertex_count = count });
    }
};

/// One shader that reads a texture at a coordinate from a uniform block and
/// writes what it found, for whichever kind of texture a test wants read.
const Sampling = struct {
    pipeline: rhi.Pipeline,
    quad: rhi.Buffer,
    params: rhi.Buffer,

    const vertex = "#version 330 core\nlayout(location = 0) in vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }\n";

    fn fragment(comptime sampler: []const u8, comptime lookup: []const u8) [:0]const u8 {
        return "#version 330 core\nuniform " ++ sampler ++ " tex;\nlayout(std140) uniform Params { vec4 coord; };\nout vec4 o_colour;\nvoid main() { o_colour = " ++ lookup ++ "; }\n";
    }

    fn init(device: *rhi.Device, source: [:0]const u8) !Sampling {
        const shader = device.createShader(.{ .glsl = .{ .vertex = vertex, .fragment = source } }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };
        const corners = [_][2]f32{ .{ -1, -1 }, .{ -1, 1 }, .{ 1, -1 }, .{ 1, 1 } };
        return .{
            .pipeline = try device.createPipeline(.{
                .shader = shader,
                .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
                .buffers = &.{.{ .stride = 8 }},
                .topology = .triangle_strip,
                .uniform_blocks = &.{"Params"},
                .textures = &.{"tex"},
            }),
            .quad = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) }),
            .params = try device.createBuffer(.{ .kind = .uniform, .size = 16 }),
        };
    }

    /// What the shader reads from `texture` at `coord`, as a 4 by 4 field of it.
    fn read(self: Sampling, device: *rhi.Device, texture: rhi.Texture, sampler: rhi.Sampler, coord: [4]f32) ![]u8 {
        try device.updateBuffer(self.params, 0, std.mem.asBytes(&coord));
        const target = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
        defer device.destroyTexture(target);
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0.5, 0.5, 0.5, 0.5 } } });
        try cmd.setPipeline(self.pipeline);
        try cmd.setVertexBuffer(0, self.quad, 0);
        try cmd.setUniformBuffer(0, self.params);
        try cmd.setTexture(0, texture, sampler);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        return device.readTexture(target, testing.allocator);
    }

    fn expect(self: Sampling, device: *rhi.Device, texture: rhi.Texture, sampler: rhi.Sampler, coord: [4]f32, colour: [4]u8, tolerance: u8) !void {
        const pixels = try self.read(device, texture, sampler, coord);
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, colour, tolerance);
    }
};

const plain_2d = Sampling.fragment("sampler2D", "texture(tex, coord.xy)");
/// The coordinate moves from one pixel of the 4 by 4 target to the next by
/// `coord.zw`, so that there is a footprint to pick a level from: with a
/// constant coordinate it is none, and the level is whatever the driver makes of zero.
const stepping_2d = Sampling.fragment("sampler2D", "texture(tex, coord.xy + gl_FragCoord.xy * coord.zw)");

test "every mip level of a texture is an image of its own" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const colours = [_][4]u8{ .{ 255, 0, 0, 255 }, .{ 0, 255, 0, 255 }, .{ 0, 0, 255, 255 }, .{ 255, 255, 0, 255 } };
    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0 });
    try testing.expectEqual(@as(u32, 4), (try device.textureInfo(texture)).mip_levels);

    for (colours, 0..) |colour, level| {
        const side = @as(usize, 8) >> @intCast(level);
        const bytes = try solid(side * side, colour);
        defer testing.allocator.free(bytes);
        try device.writeTexture(texture, .{ .mip = @intCast(level) }, bytes, 0, 0);
    }
    for (colours, 0..) |colour, level| {
        const side = @as(usize, 8) >> @intCast(level);
        try expectImage(device, texture, .{ .mip = @intCast(level) }, side, side, colour, 0);
    }

    // A box inside a level, written on its own: rows one and two of eight, which
    // read back where they were put, with what was there above and below.
    const white = try solid(4 * 2, .{ 255, 255, 255, 255 });
    defer testing.allocator.free(white);
    try device.writeTexture(texture, .{ .x = 2, .y = 1, .width = 4, .height = 2 }, white, 0, 0);
    const level0 = try device.readTexture(texture, testing.allocator);
    defer testing.allocator.free(level0);
    try expectTexel(level0, 8, 2, 1, .{ 255, 255, 255, 255 }, 0);
    try expectTexel(level0, 8, 5, 2, .{ 255, 255, 255, 255 }, 0);
    try expectTexel(level0, 8, 1, 1, colours[0], 0);
    try expectTexel(level0, 8, 6, 2, colours[0], 0);
    try expectTexel(level0, 8, 3, 0, colours[0], 0);
    try expectTexel(level0, 8, 3, 3, colours[0], 0);
    // And the other levels did not notice.
    try expectImage(device, texture, .{ .mip = 1 }, 4, 4, colours[1], 0);
}

test "rows with padding, of any pitch, land where they belong" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const width = 4;
    const height = 4;
    // A whole number of texels past the row (a row length OpenGL can be told),
    // one that is not (gathered first), and the tight one.
    for ([_]usize{ 16, 20, 18, 32 }) |pitch| {
        const bytes = try testing.allocator.alloc(u8, pitch * height);
        defer testing.allocator.free(bytes);
        @memset(bytes, 0xEE);
        for (0..height) |y| {
            for (0..width) |x| {
                // Different along every row and down every column, so that a
                // shear or a turn is seen, and each texel is what was written.
                bytes[y * pitch + x * 4 ..][0..4].* = .{ @intCast(x * 50 + 5), @intCast(y * 70 + 9), 33, 255 };
            }
        }

        const written = try device.createTexture(.{ .width = width, .height = height });
        try device.writeTexture(written, .{}, bytes, pitch, 0);
        const made = try device.createTexture(.{ .width = width, .height = height, .data = bytes, .row_pitch = pitch });
        for ([_]rhi.Texture{ written, made }) |texture| {
            const pixels = try device.readTexture(texture, testing.allocator);
            defer testing.allocator.free(pixels);
            for (0..height) |y| {
                for (0..width) |x| {
                    testing.expectEqual([4]u8{ @intCast(x * 50 + 5), @intCast(y * 70 + 9), 33, 255 }, at(pixels, width, x, y)) catch |err| {
                        std.debug.print("pitch {d}, texel ({d}, {d})\n", .{ pitch, x, y });
                        return err;
                    };
                }
            }
        }
    }
}

test "formats come back as bytes, whatever they were" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const floats = [4]f32{ 0.5, 0.25, 1, 0.75 };
    const one_float = [1]f32{0.25};
    const two_floats = [2]f32{ 0.5, 0.25 };
    const Case = struct { format: rhi.Format, bytes: []const u8, want: [4]u8, tolerance: u8 = 0 };
    const cases = [_]Case{
        // One channel is grey, and two have no blue.
        .{ .format = .r8_unorm, .bytes = &.{77}, .want = .{ 77, 77, 77, 255 } },
        .{ .format = .rg8_unorm, .bytes = &.{ 10, 20 }, .want = .{ 10, 20, 0, 255 } },
        .{ .format = .rgba8_unorm_srgb, .bytes = &.{ 10, 20, 30, 40 }, .want = .{ 10, 20, 30, 40 } },
        // Written blue first, read red first.
        .{ .format = .bgra8_unorm, .bytes = &.{ 10, 20, 30, 40 }, .want = .{ 30, 20, 10, 40 } },
        .{ .format = .bgra8_unorm_srgb, .bytes = &.{ 10, 20, 30, 40 }, .want = .{ 30, 20, 10, 40 } },
        // Halves and floats are clamped to bytes.
        .{ .format = .r16_float, .bytes = &.{ 0x00, 0x38 }, .want = .{ 128, 128, 128, 255 } },
        .{ .format = .rg16_float, .bytes = &.{ 0x00, 0x38, 0x00, 0x34 }, .want = .{ 128, 64, 0, 255 } },
        .{ .format = .rgba16_float, .bytes = &.{ 0x00, 0x38, 0x00, 0x34, 0x00, 0x3C, 0x00, 0x00 }, .want = .{ 128, 64, 255, 0 } },
        .{ .format = .r32_float, .bytes = std.mem.asBytes(&one_float), .want = .{ 64, 64, 64, 255 } },
        .{ .format = .rg32_float, .bytes = std.mem.asBytes(&two_floats), .want = .{ 128, 64, 0, 255 } },
        .{ .format = .rgba32_float, .bytes = std.mem.asBytes(&floats), .want = .{ 128, 64, 255, 191 } },
        // 1023 in red and 3 in alpha; and 1.0 in the red of eleven-eleven-ten.
        .{ .format = .rgb10a2_unorm, .bytes = &.{ 0xFF, 0x03, 0x00, 0xC0 }, .want = .{ 255, 0, 0, 255 } },
        .{ .format = .rg11b10_float, .bytes = &.{ 0xC0, 0x03, 0x00, 0x00 }, .want = .{ 255, 0, 0, 255 } },
    };
    for (cases) |case| {
        if (!device.caps().formatSupport(case.format).sampled) continue;
        errdefer std.debug.print("format {t}\n", .{case.format});

        const made = try device.createTexture(.{ .width = 1, .height = 1, .format = case.format, .data = case.bytes });
        try expectImage(device, made, .{}, 1, 1, case.want, case.tolerance);
        // And the same through a write, which is the other way in.
        const written = try device.createTexture(.{ .width = 1, .height = 1, .format = case.format });
        try device.writeTexture(written, .{}, case.bytes, 0, 0);
        try expectImage(device, written, .{}, 1, 1, case.want, case.tolerance);
    }
}

test "a target in an sRGB format is written encoded" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const target = try device.createTexture(.{ .width = 4, .height = 4, .format = .rgba8_unorm_srgb, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0.5, 0.5, 0.5, 1 } } });
    try cmd.endPass();
    try device.submit();
    // Half as much light is more than half a byte: 188, the way Direct3D
    // stores it, and not the 128 of a target that took the number as it came.
    try expectImage(device, target, .{}, 4, 4, .{ 188, 188, 188, 255 }, 2);
}

test "generateMips averages down to a single texel" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const texture = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0 });
    var checker: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| {
        const white = (i % 4 + i / 4) % 2 == 0;
        checker[i * 4 ..][0..4].* = if (white) .{ 255, 255, 255, 255 } else .{ 0, 0, 0, 255 };
    }
    try device.writeTexture(texture, .{}, &checker, 0, 0);

    const cmd = device.begin();
    try cmd.generateMips(texture);
    try device.submit();

    // Every two by two of a checkerboard is half white, and so is the whole.
    try expectImage(device, texture, .{ .mip = 1 }, 2, 2, .{ 128, 128, 128, 255 }, 2);
    try expectImage(device, texture, .{ .mip = 2 }, 1, 1, .{ 128, 128, 128, 255 }, 2);
}

test "a cube is six faces, written and read a face at a time" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const faces = [_][4]u8{
        .{ 255, 0, 0, 255 },   .{ 0, 255, 0, 255 },   .{ 0, 0, 255, 255 },
        .{ 255, 255, 0, 255 }, .{ 0, 255, 255, 255 }, .{ 255, 0, 255, 255 },
    };
    const lower = [_][4]u8{
        .{ 10, 0, 0, 255 },  .{ 0, 20, 0, 255 },  .{ 0, 0, 30, 255 },
        .{ 40, 40, 0, 255 }, .{ 0, 50, 50, 255 }, .{ 60, 0, 60, 255 },
    };

    // Written a face at a time, to both levels.
    const written = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .mip_levels = 2 });
    for (faces, lower, 0..) |colour, small, face| {
        const big = try solid(16, colour);
        defer testing.allocator.free(big);
        try device.writeTexture(written, .{ .z = @intCast(face), .depth = 1 }, big, 0, 0);
        const little = try solid(4, small);
        defer testing.allocator.free(little);
        try device.writeTexture(written, .{ .mip = 1, .z = @intCast(face), .depth = 1 }, little, 0, 0);
    }
    for (faces, lower, 0..) |colour, small, face| {
        try expectImage(device, written, .{ .layer = @intCast(face) }, 4, 4, colour, 0);
        try expectImage(device, written, .{ .mip = 1, .layer = @intCast(face) }, 2, 2, small, 0);
    }

    // And made from six faces of bytes in a row, then one of them replaced.
    var all: [6 * 16 * 4]u8 = undefined;
    for (faces, 0..) |colour, face| {
        for (0..16) |i| all[(face * 16 + i) * 4 ..][0..4].* = colour;
    }
    const made = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &all });
    for (faces, 0..) |colour, face| try expectImage(device, made, .{ .layer = @intCast(face) }, 4, 4, colour, 0);
    const replacement = try solid(16, .{ 9, 9, 9, 255 });
    defer testing.allocator.free(replacement);
    try device.writeTexture(made, .{ .z = 2, .depth = 1 }, replacement, 0, 0);
    try expectImage(device, made, .{ .layer = 2 }, 4, 4, .{ 9, 9, 9, 255 }, 0);
    try expectImage(device, made, .{ .layer = 1 }, 4, 4, faces[1], 0);
    try expectImage(device, made, .{ .layer = 3 }, 4, 4, faces[3], 0);
}

test "a volume is slices that shrink with its levels" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    // Eight by eight by four: four levels, with four slices, two, one and one.
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 4, .mip_levels = 0 });
    try testing.expectEqual(@as(u32, 4), (try device.textureInfo(volume)).mip_levels);

    // Level zero: all four slices in one write, then one of them again.
    var stack: [4 * 64 * 4]u8 = undefined;
    for (0..4) |slice| {
        for (0..64) |i| stack[(slice * 64 + i) * 4 ..][0..4].* = .{ @intCast(slice * 60 + 20), 100, 200, 255 };
    }
    try device.writeTexture(volume, .{}, &stack, 0, 0);
    for (0..4) |slice| try expectImage(device, volume, .{ .layer = @intCast(slice) }, 8, 8, .{ @intCast(slice * 60 + 20), 100, 200, 255 }, 0);
    const again = try solid(64, .{ 1, 2, 3, 255 });
    defer testing.allocator.free(again);
    try device.writeTexture(volume, .{ .z = 3, .depth = 1 }, again, 0, 0);
    try expectImage(device, volume, .{ .layer = 3 }, 8, 8, .{ 1, 2, 3, 255 }, 0);
    try expectImage(device, volume, .{ .layer = 2 }, 8, 8, .{ 140, 100, 200, 255 }, 0);

    // Level one is four by four by two, with its own pitches: padded rows,
    // and a slice that does not follow its neighbour at once.
    const row_pitch = 4 * 4 + 8;
    const slice_pitch = row_pitch * 4 + 16;
    var padded: [slice_pitch * 2]u8 = @splat(0xEE);
    for (0..2) |slice| {
        for (0..4) |y| {
            for (0..4) |x| {
                padded[slice * slice_pitch + y * row_pitch + x * 4 ..][0..4].* = .{ @intCast(slice * 100 + 50), 7, 8, 255 };
            }
        }
    }
    try device.writeTexture(volume, .{ .mip = 1 }, &padded, row_pitch, slice_pitch);
    try expectImage(device, volume, .{ .mip = 1, .layer = 0 }, 4, 4, .{ 50, 7, 8, 255 }, 0);
    try expectImage(device, volume, .{ .mip = 1, .layer = 1 }, 4, 4, .{ 150, 7, 8, 255 }, 0);

    // And with a pitch that is not a whole number of texels, which is gathered
    // first: two slices at once, with an odd gap between them.
    const odd_row = 4 * 4 + 2;
    const odd_slice = odd_row * 4 + 3;
    var odd: [odd_slice * 2]u8 = @splat(0xEE);
    for (0..2) |slice| {
        for (0..4) |y| {
            for (0..4) |x| {
                odd[slice * odd_slice + y * odd_row + x * 4 ..][0..4].* = .{ @intCast(slice * 90 + 30), 1, 2, 255 };
            }
        }
    }
    try device.writeTexture(volume, .{ .mip = 1 }, &odd, odd_row, odd_slice);
    try expectImage(device, volume, .{ .mip = 1, .layer = 0 }, 4, 4, .{ 30, 1, 2, 255 }, 0);
    try expectImage(device, volume, .{ .mip = 1, .layer = 1 }, 4, 4, .{ 120, 1, 2, 255 }, 0);

    // The last two levels are one slice.
    const single = try solid(1, .{ 77, 78, 79, 255 });
    defer testing.allocator.free(single);
    try device.writeTexture(volume, .{ .mip = 3 }, single, 0, 0);
    try expectImage(device, volume, .{ .mip = 3 }, 1, 1, .{ 77, 78, 79, 255 }, 0);
    try testing.expectError(error.InvalidArgument, device.readSubresource(volume, .{ .mip = 2, .layer = 1 }, testing.allocator));
}

test "a pass draws into a layer, a level, a face and a slice" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{});
    const red = try Flat.quad(device, .{ 1, 0, 0, 1 });

    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 16, .height = 16, .depth_or_layers = 3, .mip_levels = 0, .usage = .{ .render_target = true } });
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 4, .mip_levels = 2, .usage = .{ .render_target = true } });

    const cmd = device.begin();
    // Layer zero cleared, layer one drawn into, and a level of layer two.
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = array }, .layer = 0, .clear_color = .{ 0, 0, 1, 1 } } });
    try cmd.endPass();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = array }, .layer = 1, .clear_color = .{ 0, 0, 0, 1 } } });
    try flat.draw(cmd, pipeline, red, 4);
    try cmd.endPass();
    // Level one of a sixteen-pixel texture is eight, and a viewport that is
    // its top-left quarter is counted from the top of that, not of level zero.
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = array }, .mip_level = 1, .layer = 2, .clear_color = .{ 0, 1, 0, 1 } } });
    try cmd.setViewport(.{ .x = 0, .y = 0, .width = 4, .height = 4 });
    try flat.draw(cmd, pipeline, red, 4);
    try cmd.endPass();
    // A face of a cube, and a slice of a volume.
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = cube }, .layer = 3, .clear_color = .{ 1, 1, 0, 1 } } });
    try cmd.endPass();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = cube }, .layer = 0, .clear_color = .{ 0, 1, 1, 1 } } });
    try cmd.endPass();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = volume }, .layer = 2, .clear_color = .{ 1, 0, 1, 1 } } });
    try cmd.endPass();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = volume }, .mip_level = 1, .layer = 1, .clear_color = .{ 1, 1, 1, 1 } } });
    try cmd.endPass();
    try device.submit();

    try expectImage(device, array, .{ .layer = 0 }, 16, 16, .{ 0, 0, 255, 255 }, 0);
    try expectImage(device, array, .{ .layer = 1 }, 16, 16, .{ 255, 0, 0, 255 }, 0);
    const quarter = try device.readSubresource(array, .{ .mip = 1, .layer = 2 }, testing.allocator);
    defer testing.allocator.free(quarter);
    try testing.expectEqual(@as(usize, 8 * 8 * 4), quarter.len);
    try expectTexel(quarter, 8, 1, 1, .{ 255, 0, 0, 255 }, 0);
    try expectTexel(quarter, 8, 6, 6, .{ 0, 255, 0, 255 }, 0);
    try expectTexel(quarter, 8, 1, 6, .{ 0, 255, 0, 255 }, 0);
    try expectTexel(quarter, 8, 6, 1, .{ 0, 255, 0, 255 }, 0);
    try expectImage(device, cube, .{ .layer = 3 }, 8, 8, .{ 255, 255, 0, 255 }, 0);
    try expectImage(device, cube, .{ .layer = 0 }, 8, 8, .{ 0, 255, 255, 255 }, 0);
    try expectImage(device, volume, .{ .layer = 2 }, 8, 8, .{ 255, 0, 255, 255 }, 0);
    try expectImage(device, volume, .{ .mip = 1, .layer = 1 }, 4, 4, .{ 255, 255, 255, 255 }, 0);

    // Depth goes where colour goes: a face of a cube of depth, and a layer of an array of it,
    // which a framebuffer takes only if it can tell what is attached and how.
    const shadow_cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .render_target = true } });
    const shadow_array = try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 3, .mip_levels = 2, .format = .depth24_stencil8, .usage = .{ .render_target = true } });
    try cmd.beginPass(.{ .depth = .{ .texture = shadow_cube, .layer = 4 } });
    try cmd.endPass();
    try cmd.beginPass(.{ .depth = .{ .texture = shadow_array, .mip_level = 1, .layer = 2 } });
    try cmd.endPass();
    try device.submit();
}

test "more images than the framebuffers kept are drawn into all the same" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const layers = 24;
    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = layers, .usage = .{ .render_target = true } });
    // Twice, the second time backwards, so that the ones thrown away
    // to make room are asked for again.
    for ([_]bool{ false, true }) |backwards| {
        const cmd = device.begin();
        for (0..layers) |i| {
            const layer = if (backwards) layers - 1 - i else i;
            const shade: f32 = @as(f32, @floatFromInt(layer)) / 32;
            try cmd.beginPass(.{ .color = .{ .target = .{ .texture = array }, .layer = @intCast(layer), .clear_color = .{ shade, 1 - shade, 0, 1 } } });
            try cmd.endPass();
        }
        try device.submit();
        for (0..layers) |layer| {
            const shade: f32 = @as(f32, @floatFromInt(layer)) / 32;
            const want: [4]u8 = .{ @intFromFloat(@round(shade * 255)), @intFromFloat(@round((1 - shade) * 255)), 0, 255 };
            try expectImage(device, array, .{ .layer = @intCast(layer) }, 4, 4, want, 1);
        }
    }

    // A texture made after another was destroyed does not inherit its framebuffers.
    device.destroyTexture(array);
    const next = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = layers, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = next }, .layer = 5, .clear_color = .{ 1, 1, 1, 1 } } });
    try cmd.endPass();
    try device.submit();
    try expectImage(device, next, .{ .layer = 5 }, 4, 4, .{ 255, 255, 255, 255 }, 0);
}

test "a shader reads a cube, a volume and an array as what they are" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const nearest = try device.createSampler(.nearest);

    // Cube: a direction chooses the face, and the faces are the six of `Dimension`.
    const faces = [_][4]u8{
        .{ 255, 0, 0, 255 },   .{ 0, 255, 0, 255 },   .{ 0, 0, 255, 255 },
        .{ 255, 255, 0, 255 }, .{ 0, 255, 255, 255 }, .{ 255, 0, 255, 255 },
    };
    var cube_bytes: [6 * 4 * 4 * 4]u8 = undefined;
    for (faces, 0..) |colour, face| {
        for (0..16) |i| cube_bytes[(face * 16 + i) * 4 ..][0..4].* = colour;
    }
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &cube_bytes });
    const cube_reader = try Sampling.init(device, comptime Sampling.fragment("samplerCube", "texture(tex, coord.xyz)"));
    // +X, -X, +Y, -Y, +Z, -Z.
    const directions = [_][4]f32{ .{ 1, 0, 0, 0 }, .{ -1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, -1, 0, 0 }, .{ 0, 0, 1, 0 }, .{ 0, 0, -1, 0 } };
    for (directions, faces) |direction, colour| try cube_reader.expect(device, cube, nearest, direction, colour, 0);

    // Volume: the third coordinate is the depth.
    var volume_bytes: [4 * 4 * 4 * 4]u8 = undefined;
    for (0..4) |slice| {
        for (0..16) |i| volume_bytes[(slice * 16 + i) * 4 ..][0..4].* = .{ @intCast(slice * 80), 50, 60, 255 };
    }
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4, .data = &volume_bytes });
    const volume_reader = try Sampling.init(device, comptime Sampling.fragment("sampler3D", "texture(tex, coord.xyz)"));
    for (0..4) |slice| {
        const depth = (@as(f32, @floatFromInt(slice)) + 0.5) / 4;
        try volume_reader.expect(device, volume, nearest, .{ 0.5, 0.5, depth, 0 }, .{ @intCast(slice * 80), 50, 60, 255 }, 0);
    }

    // Array: the third coordinate is the layer, as a whole number.
    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = 3, .data = volume_bytes[0 .. 3 * 16 * 4] });
    const array_reader = try Sampling.init(device, comptime Sampling.fragment("sampler2DArray", "texture(tex, coord.xyz)"));
    for (0..3) |layer| {
        try array_reader.expect(device, array, nearest, .{ 0.5, 0.5, @floatFromInt(layer), 0 }, .{ @intCast(layer * 80), 50, 60, 255 }, 0);
    }
}

test "a triangle drawn with four samples is resolved into one" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    if (!device.caps().formatSupport(.rgba8_unorm).supportsSamples(4)) return error.SkipZigTest;

    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{ .samples = 4 });
    const vertices = [_]Vertex{
        .{ .position = .{ -0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.0, 0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });

    const many = try device.createTexture(.{ .width = 64, .height = 64, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const resolved = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });

    // Resolved into a texture, and - the same samples, kept - into the window.
    const surface = try fixture.window.?.createSurface(device);
    // A surface is resolved into at its own size, which is the window's.
    const window = try device.surfaceSize(surface);
    const many_window = try device.createTexture(.{ .width = window.width, .height = window.height, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = many }, .clear_color = .{ 0, 0, 1, 1 }, .resolve = .{ .texture = resolved } } });
    try flat.draw(cmd, pipeline, buffer, 3);
    try cmd.endPass();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = many_window }, .clear_color = .{ 0, 0, 1, 1 }, .resolve = .{ .surface = surface } } });
    try flat.draw(cmd, pipeline, buffer, 3);
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(resolved, testing.allocator);
    defer testing.allocator.free(pixels);
    try expectTexel(pixels, 64, 32, 20, .{ 255, 0, 0, 255 }, 0);
    try expectTexel(pixels, 64, 32, 40, .{ 255, 0, 0, 255 }, 0);
    try expectTexel(pixels, 64, 2, 2, .{ 0, 0, 255, 255 }, 0);
    try expectTexel(pixels, 64, 61, 61, .{ 0, 0, 255, 255 }, 0);
    // Wide at the base and narrow at the point, which is up: the resolved image is a drawn one.
    try expectTexel(pixels, 64, 20, 46, .{ 255, 0, 0, 255 }, 0);
    try expectTexel(pixels, 64, 20, 17, .{ 0, 0, 255, 255 }, 0);

    // What makes it multisampled and not merely bigger: along the edges,
    // pixels the triangle covers in part are a mix of the two colours.
    var mixed: usize = 0;
    for (0..64 * 64) |i| {
        const texel = pixels[i * 4 ..][0..4];
        if (texel[0] > 8 and texel[0] < 247 and texel[2] > 8 and texel[2] < 247) mixed += 1;
    }
    try testing.expect(mixed > 20);
}

test "a depth-only pass is what a later colour pass tests against" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);

    const depth = try device.createTexture(.{ .width = 64, .height = 64, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    const color = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });

    // Nothing is drawn in colour by a pipeline with no colour format.
    const depth_only = try flat.pipeline(device, .{ .color_format = null, .depth_format = .depth32_float, .depth = .standard });
    const tested = try flat.pipeline(device, .{ .depth_format = .depth32_float, .depth = .{ .test_enabled = true, .write = false, .compare = .less } });
    const near_quad = try Flat.quad(device, .{ 1, 0, 0, 1 });
    const far_quad = try Flat.quad(device, .{ 0, 1, 0, 1 });

    const cmd = device.begin();
    // The left half of the image, at depth a quarter.
    try cmd.beginPass(.{ .depth = .{ .texture = depth } });
    try cmd.setViewport(.{ .width = 64, .height = 64, .min_depth = 0.25, .max_depth = 0.25 });
    try cmd.setScissor(.{ .x = 0, .y = 0, .width = 32, .height = 64 });
    try flat.draw(cmd, depth_only, near_quad, 4);
    try cmd.endPass();
    // Then the whole of it, behind, in colour: what is in front is kept.
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = color }, .clear_color = .{ 0, 0, 0, 1 } }, .depth = .{ .texture = depth, .load = .load } });
    try cmd.setViewport(.{ .width = 64, .height = 64, .min_depth = 0.75, .max_depth = 0.75 });
    try flat.draw(cmd, tested, far_quad, 4);
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(color, testing.allocator);
    defer testing.allocator.free(pixels);
    try expectTexel(pixels, 64, 8, 32, .{ 0, 0, 0, 255 }, 0);
    try expectTexel(pixels, 64, 30, 5, .{ 0, 0, 0, 255 }, 0);
    try expectTexel(pixels, 64, 40, 32, .{ 0, 255, 0, 255 }, 0);
    try expectTexel(pixels, 64, 60, 60, .{ 0, 255, 0, 255 }, 0);

    // The same image of depth, sampled: as it is, and compared. A sampler that
    // does not compare must read the depth itself even after one that did.
    const depth_reader = try Sampling.init(device, comptime Sampling.fragment("sampler2D", "vec4(vec3(texture(tex, coord.xy).r), 1.0)"));
    const shadow_reader = try Sampling.init(device, comptime Sampling.fragment("sampler2DShadow", "vec4(vec3(texture(tex, coord.xyz)), 1.0)"));
    const plain = try device.createSampler(.nearest);
    const comparing = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .compare = .less_equal });
    // Left of the middle is a quarter: 64 of 255. Right of it is the far plane.
    try depth_reader.expect(device, depth, plain, .{ 0.25, 0.5, 0, 0 }, .{ 64, 64, 64, 255 }, 1);
    try depth_reader.expect(device, depth, plain, .{ 0.75, 0.5, 0, 0 }, .{ 255, 255, 255, 255 }, 0);
    // A reference of a tenth is in front of a quarter; a half is behind it.
    try shadow_reader.expect(device, depth, comparing, .{ 0.25, 0.5, 0.1, 0 }, .{ 255, 255, 255, 255 }, 0);
    try shadow_reader.expect(device, depth, comparing, .{ 0.25, 0.5, 0.5, 0 }, .{ 0, 0, 0, 255 }, 0);
    try depth_reader.expect(device, depth, plain, .{ 0.25, 0.5, 0, 0 }, .{ 64, 64, 64, 255 }, 1);
}

test "a sampler chooses its levels, its border and its bias" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const reader = try Sampling.init(device, plain_2d);
    const stepping = try Sampling.init(device, stepping_2d);
    const features = device.caps().features;

    // Three levels, one colour each.
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0 });
    const colours = [_][4]u8{ .{ 255, 0, 0, 255 }, .{ 0, 255, 0, 255 }, .{ 0, 0, 255, 255 } };
    for (colours, 0..) |colour, level| {
        const side = @as(usize, 4) >> @intCast(level);
        const bytes = try solid(side * side, colour);
        defer testing.allocator.free(bytes);
        try device.writeTexture(texture, .{ .mip = @intCast(level) }, bytes, 0, 0);
    }
    const middle: [4]f32 = .{ 0.5, 0.5, 0, 0 };
    // One texel of level zero a pixel: its own level, and a level of bias from the next.
    const walking: [4]f32 = .{ 0, 0, 0.25, 0.25 };

    // Level zero only, whatever is asked of the levels below it.
    const flat_chain = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .lod_min = 1, .lod_max = 1 });
    try stepping.expect(device, texture, flat_chain, walking, colours[0], 0);
    // The range of levels is the level: one and one is level one, two and two is two.
    const level_one = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 1, .lod_max = 1 });
    try stepping.expect(device, texture, level_one, walking, colours[1], 0);
    const level_two = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 2, .lod_max = 2 });
    try stepping.expect(device, texture, level_two, walking, colours[2], 0);
    // More taps than the hardware has are clamped, not refused.
    _ = try device.createSampler(.{ .mip_filter = .linear, .max_anisotropy = 16 });

    if (features.sampler_lod_bias) {
        // A level of bias moves a base level to the next.
        const biased = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_bias = 1 });
        try stepping.expect(device, texture, biased, walking, colours[1], 0);
    }

    if (features.sampler_border) {
        const solid_red = try solid(4, colours[0]);
        defer testing.allocator.free(solid_red);
        const small = try device.createTexture(.{ .width = 2, .height = 2, .data = solid_red });
        const outside: [4]f32 = .{ 1.5, 0.5, 0, 0 };
        const white = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .wrap_u = .border, .wrap_v = .border, .border = .opaque_white });
        try reader.expect(device, small, white, outside, .{ 255, 255, 255, 255 }, 0);
        const black = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .wrap_u = .border, .wrap_v = .border, .border = .opaque_black });
        try reader.expect(device, small, black, outside, .{ 0, 0, 0, 255 }, 0);
        const clear = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .wrap_u = .border, .wrap_v = .border });
        try reader.expect(device, small, clear, outside, .{ 0, 0, 0, 0 }, 0);
        // Inside it is the texture; and a repeat brings the far side round.
        try reader.expect(device, small, white, middle, colours[0], 0);
        const repeat = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .wrap_u = .repeat, .wrap_v = .repeat });
        try reader.expect(device, small, repeat, outside, colours[0], 0);
    }
}

/// A block of texels of one colour, in each compressed format this test can
/// write one of by hand.
const Block = struct { format: rhi.Format, red: []const u8, blue: ?[]const u8 = null };
const blocks = [_]Block{
    // Colour 0 alone, as 565: red, or blue.
    .{ .format = .bc1_rgba_unorm, .red = &.{ 0x00, 0xF8, 0, 0, 0, 0, 0, 0 }, .blue = &.{ 0x1F, 0x00, 0, 0, 0, 0, 0, 0 } },
    // ETC2's individual mode with one colour and the smallest table.
    .{ .format = .etc2_rgb8_unorm, .red = &.{ 0xFF, 0x00, 0x00, 0x00, 0, 0, 0, 0 } },
    // ASTC's void extent: one colour, in sixteen bits a channel.
    .{ .format = .astc_4x4_unorm, .red = &.{ 0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF } },
};

test "a compressed texture is written in blocks and sampled as colour" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const reader = try Sampling.init(device, plain_2d);
    const stepping = try Sampling.init(device, stepping_2d);
    const nearest = try device.createSampler(.nearest);

    var tried: usize = 0;
    for (blocks) |block| {
        if (!device.caps().formatSupport(block.format).sampled) continue;
        tried += 1;
        errdefer std.debug.print("format {t}\n", .{block.format});
        const size = block.red.len;

        // Eight by eight is four blocks; made with them, and with a chain
        // whose last levels are smaller than a block.
        const four = try testing.allocator.alloc(u8, size * 4);
        defer testing.allocator.free(four);
        for (0..4) |i| @memcpy(four[i * size ..][0..size], block.red);
        const made = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .format = block.format, .data = four });
        try reader.expect(device, made, nearest, .{ 0.25, 0.25, 0, 0 }, .{ 255, 0, 0, 255 }, 12);
        try reader.expect(device, made, nearest, .{ 0.75, 0.75, 0, 0 }, .{ 255, 0, 0, 255 }, 12);

        // And empty, then written: one level at a time, down to the last, which is one block.
        const written = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .format = block.format });
        for (0..4) |level| {
            const side = @as(u32, 8) >> @intCast(level);
            const count = @max(1, side / 4) * @max(1, side / 4);
            const level_bytes = try testing.allocator.alloc(u8, size * count);
            defer testing.allocator.free(level_bytes);
            for (0..count) |i| @memcpy(level_bytes[i * size ..][0..size], block.red);
            try device.writeTexture(written, .{ .mip = @intCast(level) }, level_bytes, 0, 0);
        }
        try reader.expect(device, written, nearest, .{ 0.25, 0.75, 0, 0 }, .{ 255, 0, 0, 255 }, 12);
        const chain = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 3, .lod_max = 3 });
        try stepping.expect(device, written, chain, .{ 0, 0, 0.25, 0.25 }, .{ 255, 0, 0, 255 }, 12);

        // A region of whole blocks, on its own.
        if (block.blue) |blue| {
            try device.writeTexture(written, .{ .x = 4, .y = 4, .width = 4, .height = 4 }, blue, 0, 0);
            try reader.expect(device, written, nearest, .{ 0.75, 0.75, 0, 0 }, .{ 0, 0, 255, 255 }, 12);
            try reader.expect(device, written, nearest, .{ 0.25, 0.25, 0, 0 }, .{ 255, 0, 0, 255 }, 12);

            // A stack of layers: a block each, and the second one another colour.
            var two: [16]u8 = undefined;
            @memcpy(two[0..8], blue);
            @memcpy(two[8..16], block.red);
            const layers = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = 2, .format = block.format, .data = &two });
            const array_reader = try Sampling.init(device, comptime Sampling.fragment("sampler2DArray", "texture(tex, coord.xyz)"));
            try array_reader.expect(device, layers, nearest, .{ 0.5, 0.5, 0, 0 }, .{ 0, 0, 255, 255 }, 12);
            try array_reader.expect(device, layers, nearest, .{ 0.5, 0.5, 1, 0 }, .{ 255, 0, 0, 255 }, 12);
        }

        // A volume of blocks is not something OpenGL can promise for every one of them.
        try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4, .format = block.format }));
    }
    if (tried == 0) return error.SkipZigTest;
}

test "what the OpenGL caps claim is what the device does" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const caps = device.caps();

    // Limits that are numbers at all, and at least what the API guarantees.
    try testing.expect(caps.limits.max_texture_2d >= 1024);
    try testing.expect(caps.limits.max_texture_3d >= 64);
    try testing.expect(caps.limits.max_texture_cube >= 1024);
    try testing.expect(caps.limits.max_texture_layers >= 256);
    try testing.expect(caps.limits.max_color_attachments >= 4);
    try testing.expect(caps.limits.max_anisotropy >= 1);

    for (std.enums.values(rhi.Format)) |format| {
        const support = caps.formatSupport(format);
        errdefer std.debug.print("format {t}: {any}\n", .{ format, support });
        // Each shape it says it comes in is made, and every other is refused as not this device's.
        for (std.enums.values(rhi.types.Dimension)) |shape| {
            const desc: rhi.TextureDesc = .{ .dimension = shape, .width = 8, .height = 8, .depth_or_layers = if (shape == .d2 or shape == .cube) 1 else 2, .format = format };
            if (support.dimensions.contains(shape)) {
                _ = device.createTexture(desc) catch |err| {
                    std.debug.print("shape {t}\n", .{shape});
                    return err;
                };
            } else {
                try testing.expectError(error.Unsupported, device.createTexture(desc));
            }
        }
        if (!support.sampled) {
            try testing.expectEqual(@as(usize, 0), support.dimensions.count());
            // Nothing else is claimed of what cannot be sampled, and it is refused as such.
            try testing.expect(!support.render_target and !support.filterable and !support.generate_mips and !support.blendable);
            try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .format = format }));
            continue;
        }

        // Sampled: made, and with a whole chain.
        const chain = try device.createTexture(.{ .width = 16, .height = 16, .mip_levels = 0, .format = format });
        if (support.generate_mips) {
            const cmd = device.begin();
            try cmd.generateMips(chain);
            try device.submit();
        }
        if (!support.render_target) continue;

        // And drawn into, in each of them, at the last of its layers, faces or slices.
        for (std.enums.values(rhi.types.Dimension)) |shape| {
            if (!support.dimensions.contains(shape)) continue;
            errdefer std.debug.print("shape {t}\n", .{shape});
            const layers: u32 = if (shape == .d2 or shape == .cube) 1 else 2;
            const layer: u32 = switch (shape) {
                .d2 => 0,
                .cube => 5,
                .d3, .d2_array => 1,
            };
            const shaped = try device.createTexture(.{ .dimension = shape, .width = 8, .height = 8, .depth_or_layers = layers, .format = format, .usage = .{ .render_target = true } });
            const cmd = device.begin();
            if (format.isDepth()) {
                try cmd.beginPass(.{ .depth = .{ .texture = shaped, .layer = layer } });
            } else {
                try cmd.beginPass(.{ .color = .{ .target = .{ .texture = shaped }, .layer = layer, .clear_color = .{ 0.25, 0.5, 0.75, 1 } } });
            }
            try cmd.endPass();
            try device.submit();
            if (!format.isDepth()) {
                const pixels = try device.readSubresource(shaped, .{ .layer = layer }, testing.allocator);
                defer testing.allocator.free(pixels);
                try testing.expectEqual(@as(usize, 8 * 8 * 4), pixels.len);
            }
        }

        // Drawn into, cleared at every sample count it claims, and read back where it can be.
        const depth = format.isDepth();
        var samples: u32 = 1;
        while (samples <= 128) : (samples *= 2) {
            if (!support.supportsSamples(samples)) continue;
            const target = try device.createTexture(.{
                .width = 16,
                .height = 16,
                .format = format,
                .samples = samples,
                .usage = .{ .sampled = samples == 1, .render_target = true },
            });
            const cmd = device.begin();
            const resolved = if (samples == 1 or depth) null else try device.createTexture(.{ .width = 16, .height = 16, .format = format, .usage = .{ .render_target = true } });
            if (depth) {
                try cmd.beginPass(.{ .depth = .{ .texture = target } });
            } else {
                try cmd.beginPass(.{ .color = .{
                    .target = .{ .texture = target },
                    .clear_color = .{ 0.25, 0.5, 0.75, 1 },
                    .resolve = if (resolved) |r| .{ .texture = r } else null,
                } });
            }
            try cmd.endPass();
            try device.submit();
            if (!depth) {
                const pixels = try device.readTexture(resolved orelse target, testing.allocator);
                defer testing.allocator.free(pixels);
                try testing.expectEqual(@as(usize, 16 * 16 * 4), pixels.len);
            }
        }
    }
}

test "a pass with two colours writes both, and clears each to its own" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    const shader = try device.createShader(.{ .glsl = .{
        .vertex = Sampling.vertex,
        .fragment = "#version 330 core\nlayout(location = 0) out vec4 first;\nlayout(location = 1) out vec4 second;\nvoid main() { first = vec4(1.0, 0.0, 0.0, 1.0); second = vec4(0.0, 1.0, 0.0, 1.0); }\n",
    } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
        .extra_color_formats = &.{.rgba8_unorm},
    });
    const corners = [_][2]f32{ .{ -1, -1 }, .{ -1, 1 }, .{ 1, -1 }, .{ 1, 1 } };
    const quad = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });

    // The second colour is a layer of an array, which is no different to a pass.
    const first = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const second = try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 2, .usage = .{ .render_target = true } });

    const cmd = device.begin();
    try cmd.beginPass(.{
        .color = .{ .target = .{ .texture = first }, .clear_color = .{ 0, 0, 0, 1 } },
        .extra_colors = &.{.{ .target = .{ .texture = second }, .layer = 1, .clear_color = .{ 1, 1, 1, 1 } }},
    });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, quad, 0);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();
    try expectImage(device, first, .{}, 8, 8, .{ 255, 0, 0, 255 }, 0);
    try expectImage(device, second, .{ .layer = 1 }, 8, 8, .{ 0, 255, 0, 255 }, 0);

    // Nothing drawn: what is kept is kept, and each colour that is cleared is cleared as itself.
    try cmd.beginPass(.{
        .color = .{ .target = .{ .texture = first }, .load = .load },
        .extra_colors = &.{.{ .target = .{ .texture = second }, .layer = 1, .clear_color = .{ 0, 0, 1, 1 } }},
    });
    try cmd.endPass();
    try device.submit();
    try expectImage(device, first, .{}, 8, 8, .{ 255, 0, 0, 255 }, 0);
    try expectImage(device, second, .{ .layer = 1 }, 8, 8, .{ 0, 0, 255, 255 }, 0);
}

test "a texture whose rows are not a multiple of four bytes is not sheared" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    // Three texels of one and of two bytes: 3 and 6 bytes a row, each a different colour along it.
    var one: [3 * 3]u8 = undefined;
    var two: [3 * 3 * 2]u8 = undefined;
    for (0..3) |y| {
        for (0..3) |x| {
            one[y * 3 + x] = @intCast(x * 40 + 10);
            two[(y * 3 + x) * 2 ..][0..2].* = .{ @intCast(x * 40 + 10), @intCast(x * 20 + 5) };
        }
    }
    const grey = try device.createTexture(.{ .width = 3, .height = 3, .format = .r8_unorm, .data = &one });
    const pair = try device.createTexture(.{ .width = 3, .height = 3, .format = .rg8_unorm });
    try device.writeTexture(pair, .{}, &two, 0, 0);
    for ([_]rhi.Texture{ grey, pair }, 0..) |texture, which| {
        const pixels = try device.readTexture(texture, testing.allocator);
        defer testing.allocator.free(pixels);
        for (0..3) |y| {
            for (0..3) |x| {
                const r: u8 = @intCast(x * 40 + 10);
                const want: [4]u8 = if (which == 0) .{ r, r, r, 255 } else .{ r, @intCast(x * 20 + 5), 0, 255 };
                try testing.expectEqual(want, at(pixels, 3, x, y));
            }
        }
    }
}

// -------------------------------------------------------------------------
// Tests - which way up an image reads back: the picture, not the memory
//
// OpenGL stores what a pass drew bottom row first and what was written from
// memory as it was given, and `readTexture` is to give the picture top row
// first for both. The test of that is a picture that is not the same upside
// down: red on top, blue below.
// -------------------------------------------------------------------------

const red_top: [4]u8 = .{ 255, 0, 0, 255 };
const blue_bottom: [4]u8 = .{ 0, 0, 255, 255 };

/// A `size` by `size` picture, top row first: `top` over its upper half and `bottom` over the rest.
fn halves(size: usize, top: [4]u8, bottom: [4]u8) ![]u8 {
    const bytes = try testing.allocator.alloc(u8, size * size * 4);
    for (0..size) |y| {
        for (0..size) |x| @memcpy(bytes[(y * size + x) * 4 ..][0..4], if (y < size / 2) &top else &bottom);
    }
    return bytes;
}

/// The picture of `halves`, from memory, into one image of a texture.
fn writeHalves(device: *rhi.Device, texture: rhi.Texture, sub: rhi.types.Subresource, size: usize, top: [4]u8, bottom: [4]u8) !void {
    const bytes = try halves(size, top, bottom);
    defer testing.allocator.free(bytes);
    try device.writeTexture(texture, .{ .mip = sub.mip, .z = sub.layer, .depth = 1 }, bytes, 0, 0);
}

fn unit(rgba: [4]u8) [4]f32 {
    var out: [4]f32 = undefined;
    for (&out, rgba) |*to, byte| to.* = @as(f32, @floatFromInt(byte)) / 255;
    return out;
}

/// The picture of `halves`, drawn: the bottom is the clear, and the top a quad under a scissor that is the upper half.
fn drawHalves(device: *rhi.Device, flat: Flat, pipeline: rhi.Pipeline, texture: rhi.Texture, sub: rhi.types.Subresource, size: u32, top: [4]u8, bottom: [4]u8) !void {
    const quad = try Flat.quad(device, unit(top));
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = texture }, .mip_level = sub.mip, .layer = sub.layer, .clear_color = unit(bottom) } });
    try cmd.setScissor(.{ .x = 0, .y = 0, .width = size, .height = size / 2 });
    try flat.draw(cmd, pipeline, quad, 4);
    try cmd.endPass();
    try device.submit();
}

/// One image reads back as `halves` of its size.
fn expectHalves(device: *rhi.Device, texture: rhi.Texture, sub: rhi.types.Subresource, size: usize, top: [4]u8, bottom: [4]u8) !void {
    const pixels = try device.readSubresource(texture, sub, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(size * size * 4, pixels.len);
    for (0..size) |y| {
        for (0..size) |x| {
            expectTexel(pixels, size, x, y, if (y < size / 2) top else bottom, 0) catch |err| {
                std.debug.print("in level {d}, layer {d}: the {s} half\n", .{ sub.mip, sub.layer, if (y < size / 2) "upper" else "lower" });
                return err;
            };
        }
    }
}

/// The side of an image of an eight-pixel texture.
fn levelSide(sub: rhi.types.Subresource) u32 {
    return @as(u32, 8) >> @intCast(sub.mip);
}

/// Each shape of texture, and a level and a layer of it that are not the first.
const Shape = struct { dimension: rhi.types.Dimension, layers: u32, subs: []const rhi.types.Subresource };
const shapes = [_]Shape{
    .{ .dimension = .d2, .layers = 1, .subs = &.{ .{ .mip = 0, .layer = 0 }, .{ .mip = 1, .layer = 0 } } },
    .{ .dimension = .d2_array, .layers = 3, .subs = &.{ .{ .mip = 0, .layer = 1 }, .{ .mip = 1, .layer = 2 } } },
    .{ .dimension = .cube, .layers = 1, .subs = &.{ .{ .mip = 0, .layer = 2 }, .{ .mip = 1, .layer = 5 } } },
    .{ .dimension = .d3, .layers = 4, .subs = &.{ .{ .mip = 0, .layer = 3 }, .{ .mip = 1, .layer = 1 } } },
};

fn shapeTexture(device: *rhi.Device, shape: Shape) !rhi.Texture {
    return device.createTexture(.{
        .dimension = shape.dimension,
        .width = 8,
        .height = 8,
        .depth_or_layers = shape.layers,
        .mip_levels = 2,
        .usage = .{ .render_target = true },
    });
}

test "an image written from memory reads back as it was written" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;

    for (shapes) |shape| {
        errdefer std.debug.print("shape {t}\n", .{shape.dimension});
        const texture = try shapeTexture(device, shape);
        for (shape.subs) |sub| try writeHalves(device, texture, sub, levelSide(sub), red_top, blue_bottom);
        for (shape.subs) |sub| try expectHalves(device, texture, sub, levelSide(sub), red_top, blue_bottom);
    }

    // Made with it, too: a plain image, and a stack of two whose second is the other way about.
    const one = try halves(4, red_top, blue_bottom);
    defer testing.allocator.free(one);
    const plain = try device.createTexture(.{ .width = 4, .height = 4, .data = one });
    try expectHalves(device, plain, .{}, 4, red_top, blue_bottom);
    const other = try halves(4, blue_bottom, red_top);
    defer testing.allocator.free(other);
    const both = try std.mem.concat(testing.allocator, u8, &.{ one, other });
    defer testing.allocator.free(both);
    const stack = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = 2, .data = both });
    try expectHalves(device, stack, .{ .layer = 0 }, 4, red_top, blue_bottom);
    try expectHalves(device, stack, .{ .layer = 1 }, 4, blue_bottom, red_top);
}

test "an image a pass drew into reads back with the top of what was drawn first" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{});

    for (shapes) |shape| {
        errdefer std.debug.print("shape {t}\n", .{shape.dimension});
        const texture = try shapeTexture(device, shape);
        for (shape.subs) |sub| try drawHalves(device, flat, pipeline, texture, sub, levelSide(sub), red_top, blue_bottom);
        for (shape.subs) |sub| try expectHalves(device, texture, sub, levelSide(sub), red_top, blue_bottom);
    }
}

test "written and drawn images of one texture are read each its own way" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{});

    const texture = try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 3, .mip_levels = 2, .usage = .{ .render_target = true } });
    const at_zero: rhi.types.Subresource = .{ .mip = 0, .layer = 0 };
    // Layer zero written and layer one drawn, in one level; the other way about in the next.
    try writeHalves(device, texture, .{ .mip = 0, .layer = 0 }, 8, red_top, blue_bottom);
    try drawHalves(device, flat, pipeline, texture, .{ .mip = 0, .layer = 1 }, 8, red_top, blue_bottom);
    try drawHalves(device, flat, pipeline, texture, .{ .mip = 1, .layer = 0 }, 4, red_top, blue_bottom);
    try writeHalves(device, texture, .{ .mip = 1, .layer = 1 }, 4, red_top, blue_bottom);
    // One that was drawn and then written, and one written and then drawn.
    try drawHalves(device, flat, pipeline, texture, .{ .mip = 0, .layer = 2 }, 8, blue_bottom, red_top);
    try writeHalves(device, texture, .{ .mip = 0, .layer = 2 }, 8, red_top, blue_bottom);
    try writeHalves(device, texture, .{ .mip = 1, .layer = 2 }, 4, blue_bottom, red_top);
    try drawHalves(device, flat, pipeline, texture, .{ .mip = 1, .layer = 2 }, 4, red_top, blue_bottom);

    for ([_]rhi.types.Subresource{ at_zero, .{ .mip = 0, .layer = 1 }, .{ .mip = 0, .layer = 2 } }) |sub| {
        try expectHalves(device, texture, sub, 8, red_top, blue_bottom);
    }
    for ([_]rhi.types.Subresource{ .{ .mip = 1, .layer = 0 }, .{ .mip = 1, .layer = 1 }, .{ .mip = 1, .layer = 2 } }) |sub| {
        try expectHalves(device, texture, sub, 4, red_top, blue_bottom);
    }
}

test "the levels a chain is generated into come out the way up their first was" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{});

    // Drawn and written, in the two layers of a stack.
    const drawn = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .render_target = true } });
    try drawHalves(device, flat, pipeline, drawn, .{}, 8, red_top, blue_bottom);
    const written = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .render_target = true } });
    try writeHalves(device, written, .{}, 8, red_top, blue_bottom);
    const stack = try device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 2, .mip_levels = 0, .usage = .{ .render_target = true } });
    try writeHalves(device, stack, .{ .layer = 0 }, 8, red_top, blue_bottom);
    try drawHalves(device, flat, pipeline, stack, .{ .layer = 1 }, 8, red_top, blue_bottom);

    const cmd = device.begin();
    try cmd.generateMips(drawn);
    try cmd.generateMips(written);
    try cmd.generateMips(stack);
    try device.submit();

    // Level one is four by four and level two is two by two; level three, a single texel, is both colours.
    for ([_]rhi.Texture{ drawn, written }) |texture| {
        try expectHalves(device, texture, .{ .mip = 0 }, 8, red_top, blue_bottom);
        try expectHalves(device, texture, .{ .mip = 1 }, 4, red_top, blue_bottom);
        try expectHalves(device, texture, .{ .mip = 2 }, 2, red_top, blue_bottom);
    }
    for (0..2) |layer| {
        try expectHalves(device, stack, .{ .mip = 1, .layer = @intCast(layer) }, 4, red_top, blue_bottom);
        try expectHalves(device, stack, .{ .mip = 2, .layer = @intCast(layer) }, 2, red_top, blue_bottom);
    }
}

test "a drawn image that is written over is a written one, and drawn again is a drawn one" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    const flat = try Flat.init(device);
    const pipeline = try flat.pipeline(device, .{});
    const green: [4]u8 = .{ 0, 255, 0, 255 };
    const yellow: [4]u8 = .{ 255, 255, 0, 255 };

    const texture = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    try drawHalves(device, flat, pipeline, texture, .{}, 8, red_top, blue_bottom);
    try expectHalves(device, texture, .{}, 8, red_top, blue_bottom);

    // All of it, from memory: as it was written, whatever it was before.
    try writeHalves(device, texture, .{}, 8, green, yellow);
    try expectHalves(device, texture, .{}, 8, green, yellow);

    // And drawn once more, it is the picture again.
    try drawHalves(device, flat, pipeline, texture, .{}, 8, red_top, blue_bottom);
    try expectHalves(device, texture, .{}, 8, red_top, blue_bottom);

    // A box of it from memory. The image is a written one from then on, whole: the
    // box is where it was put, and what was drawn round it - stored bottom row first - reads
    // back the other way up. That is the granularity, and the price of it.
    const box = try solid(4 * 2, green);
    defer testing.allocator.free(box);
    try device.writeTexture(texture, .{ .x = 2, .y = 0, .width = 4, .height = 2 }, box, 0, 0);
    const pixels = try device.readTexture(texture, testing.allocator);
    defer testing.allocator.free(pixels);
    try expectTexel(pixels, 8, 2, 0, green, 0);
    try expectTexel(pixels, 8, 5, 1, green, 0);
    try expectTexel(pixels, 8, 1, 0, blue_bottom, 0);
    try expectTexel(pixels, 8, 6, 1, blue_bottom, 0);
    try expectTexel(pixels, 8, 0, 7, red_top, 0);
    // Drawn into again it is a drawn one, and the box is gone with the rest.
    try drawHalves(device, flat, pipeline, texture, .{}, 8, red_top, blue_bottom);
    try expectHalves(device, texture, .{}, 8, red_top, blue_bottom);
}

test "a compressed texture need not be whole blocks" {
    var fixture = try openGl();
    defer fixture.close();
    const device = &fixture.device;
    try testing.expect(device.caps().features.compressed_partial_blocks);
    try testing.expect(device.caps().features.render_target_origin_bottom_left);
    const reader = try Sampling.init(device, plain_2d);
    const nearest = try device.createSampler(.nearest);

    var tried: usize = 0;
    for (blocks) |block| {
        if (!device.caps().formatSupport(block.format).sampled) continue;
        tried += 1;
        errdefer std.debug.print("format {t}\n", .{block.format});
        const size = block.red.len;

        // Five by seven is two blocks across and two down, the last of each partly outside; and every level down to a texel.
        const info = block.format.info();
        const across = std.math.divCeil(usize, 5, info.block_width) catch unreachable;
        const down = std.math.divCeil(usize, 7, info.block_height) catch unreachable;
        const level_zero = try testing.allocator.alloc(u8, size * across * down);
        defer testing.allocator.free(level_zero);
        for (0..across * down) |i| @memcpy(level_zero[i * size ..][0..size], block.red);

        const made = try device.createTexture(.{ .width = 5, .height = 7, .mip_levels = 0, .format = block.format, .data = level_zero });
        try reader.expect(device, made, nearest, .{ 0.5, 0.5, 0, 0 }, .{ 255, 0, 0, 255 }, 12);
        const written = try device.createTexture(.{ .width = 5, .height = 7, .mip_levels = 0, .format = block.format });
        try device.writeTexture(written, .{}, level_zero, 0, 0);
        try reader.expect(device, written, nearest, .{ 0.9, 0.9, 0, 0 }, .{ 255, 0, 0, 255 }, 12);
        // A stack and a cube of the same odd size are made the same way.
        const dims = device.caps().formatSupport(block.format).dimensions;
        if (dims.contains(.d2_array)) _ = try device.createTexture(.{ .dimension = .d2_array, .width = 5, .height = 7, .depth_or_layers = 2, .mip_levels = 0, .format = block.format });
        if (dims.contains(.cube)) _ = try device.createTexture(.{ .dimension = .cube, .width = 5, .height = 5, .mip_levels = 0, .format = block.format });
    }
    if (tried == 0) return error.SkipZigTest;
}
