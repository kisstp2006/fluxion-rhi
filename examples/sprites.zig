// SPDX-License-Identifier: BSD-2-Clause

//! Two dimensions: a batch of textured sprites bouncing in a window, in one
//! draw call, on whichever backend is asked for.
//!
//! Run it with `zig build example-sprites`. `-- --backend gl` or `--backend
//! d3d11` picks the API; the default is Direct3D on Windows and OpenGL
//! elsewhere. `-- --frames 240` has it close itself; Escape quits.
//!
//! `-- --capture sprites.png` skips the window: it steps the scene to
//! `--at SECONDS`, draws one frame into a texture and writes it out, which is
//! how this can be looked at on a machine with no display - and how the test
//! at the bottom checks that both backends draw the same picture.
//!
//! What a 2D renderer is made of, and what this library was built for first:
//!
//!   * one quad in a vertex buffer, and a second buffer stepping once per
//!     instance with where each sprite goes, what colour it is and which part
//!     of the atlas it shows - so a hundred sprites are one `draw`;
//!   * that instance buffer rewritten every frame, which is what `dynamic` is
//!     for;
//!   * an orthographic matrix with the origin at the top left, built with
//!     `device.clip()` so it is right for the backend that was opened;
//!   * alpha blending, so the transparent corners of the atlas are transparent;
//!   * one shader source, compiled by
//!     [Fluxion Shader](https://github.com/kisstp2006/fluxion-shader) into the
//!     language each backend takes - and the pipeline described out of what
//!     that shader says about itself, so a location, a slot and a name are
//!     never written down twice.

const std = @import("std");
const Io = std.Io;

const rhi = @import("fluxion_rhi");
const math = @import("fluxion_math");
const image = @import("fluxion_image");
const shader = @import("fluxion_shader");
const windowing = @import("window");
const Window = windowing.Window;

const background: rhi.Color = .{ 0.06, 0.07, 0.10, 1 };

/// How many bounce about. One draw call either way.
const count = 24;

// -------------------------------------------------------------------------
// The shader, in both languages
// -------------------------------------------------------------------------

/// One source, and the library that reads it writes both languages out. The
/// contract between them - `layout(location = n)` against `ATTRn`, a block
/// name against `register(bn)` - is what `fluxion-shader` exists to keep, so
/// none of it appears twice here.
const source =
    \\attribute vec2 corner : 0;
    \\attribute vec4 placement : 1;
    \\attribute vec4 tint : 2;
    \\attribute vec4 uv_rect : 3;
    \\
    \\varying vec2 uv;
    \\varying vec4 shade;
    \\
    \\uniform Frame : 0 {
    \\    mat4 projection;
    \\}
    \\
    \\texture2d atlas : 0;
    \\
    \\vertex {
    \\    // The quad is a unit square; each instance says where it goes.
    \\    vec2 world = placement.xy + corner * placement.zw;
    \\    uv = mix(uv_rect.xy, uv_rect.zw, corner);
    \\    shade = tint;
    \\    position = projection * vec4(world, 0.0, 1.0);
    \\}
    \\
    \\fragment {
    \\    target = sample(atlas, uv) * shade;
    \\}
;

/// Which vertex buffer each attribute is read from.
///
/// The one thing the shader does not know and cannot: a location and a format
/// belong to the shader, but how the vertices are packed into buffers is the
/// program's business. Everything else about the pipeline comes out of the
/// module.
fn bufferOf(name: []const u8) u32 {
    return if (std.mem.eql(u8, name, "corner")) 0 else 1;
}

fn vertexFormat(ty: shader.Type) !rhi.VertexFormat {
    return switch (ty) {
        .float => .float,
        .vec2 => .float2,
        .vec3 => .float3,
        .vec4 => .float4,
        else => error.NotAVertexFormat,
    };
}

// -------------------------------------------------------------------------
// The scene
// -------------------------------------------------------------------------

/// What the instance buffer holds for each sprite: forty-eight bytes, three
/// float4s, which is what the vertex shader reads at locations 1 to 3.
const Instance = extern struct {
    /// x, y, width, height, in pixels from the top left.
    placement: [4]f32,
    tint: [4]f32,
    /// u0, v0, u1, v1 into the atlas.
    uv_rect: [4]f32,
};

const Sprite = struct {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    size: f32,
    tint: [4]f32,
    /// Which of the atlas's four cells.
    cell: u2,
};

const Scene = struct {
    sprites: [count]Sprite,
    width: f32,
    height: f32,

    /// Deterministic on purpose: `--capture` has to produce the same picture
    /// every time it is asked for the same moment, on every backend.
    fn init(width: u32, height: u32) Scene {
        var random: std.Random.DefaultPrng = .init(0x5EED);
        const rand = random.random();
        var self: Scene = .{ .sprites = undefined, .width = @floatFromInt(width), .height = @floatFromInt(height) };
        for (&self.sprites, 0..) |*sprite, i| {
            const size = 40 + rand.float(f32) * 56;
            sprite.* = .{
                .x = rand.float(f32) * (self.width - size),
                .y = rand.float(f32) * (self.height - size),
                .dx = (rand.float(f32) - 0.5) * 260,
                .dy = (rand.float(f32) - 0.5) * 260,
                .size = size,
                .tint = hue(@as(f32, @floatFromInt(i)) / count, 0.9),
                .cell = @intCast(i % 4),
            };
        }
        return self;
    }

    fn resize(self: *Scene, width: u32, height: u32) void {
        self.width = @floatFromInt(width);
        self.height = @floatFromInt(height);
    }

    fn step(self: *Scene, seconds: f32) void {
        for (&self.sprites) |*s| {
            s.x += s.dx * seconds;
            s.y += s.dy * seconds;
            if (s.x < 0) {
                s.x = 0;
                s.dx = -s.dx;
            }
            if (s.y < 0) {
                s.y = 0;
                s.dy = -s.dy;
            }
            if (s.x + s.size > self.width) {
                s.x = self.width - s.size;
                s.dx = -s.dx;
            }
            if (s.y + s.size > self.height) {
                s.y = self.height - s.size;
                s.dy = -s.dy;
            }
        }
    }

    fn instances(self: Scene) [count]Instance {
        var out: [count]Instance = undefined;
        for (&out, self.sprites) |*instance, s| {
            const u: f32 = @floatFromInt(s.cell & 1);
            const v: f32 = @floatFromInt(s.cell >> 1);
            instance.* = .{
                .placement = .{ s.x, s.y, s.size, s.size },
                .tint = s.tint,
                .uv_rect = .{ u * 0.5, v * 0.5, u * 0.5 + 0.5, v * 0.5 + 0.5 },
            };
        }
        return out;
    }
};

fn hue(turn: f32, alpha: f32) [4]f32 {
    const angle = turn * std.math.tau;
    return .{
        0.5 + 0.45 * @cos(angle),
        0.5 + 0.45 * @cos(angle - std.math.tau / 3.0),
        0.5 + 0.45 * @cos(angle + std.math.tau / 3.0),
        alpha,
    };
}

/// A 64 by 64 atlas of four 32-pixel cells: a disc, a ring, a diamond and a
/// square, white on transparent, so the tint is the colour and the corners
/// show what is behind.
fn makeAtlas() [64 * 64 * 4]u8 {
    var pixels: [64 * 64 * 4]u8 = undefined;
    for (0..64) |y| {
        for (0..64) |x| {
            const cell: u2 = @intCast((x / 32) + (y / 32) * 2);
            const cx = @as(f32, @floatFromInt(x % 32)) + 0.5 - 16;
            const cy = @as(f32, @floatFromInt(y % 32)) + 0.5 - 16;
            const r = @sqrt(cx * cx + cy * cy);
            const inside = switch (cell) {
                0 => r < 15,
                1 => r < 15 and r > 9,
                2 => @abs(cx) + @abs(cy) < 15,
                3 => @abs(cx) < 13 and @abs(cy) < 13,
            };
            const edge = switch (cell) {
                0 => r > 12,
                1 => r > 13 or r < 11,
                2 => @abs(cx) + @abs(cy) > 12,
                3 => @abs(cx) > 10 or @abs(cy) > 10,
            };
            const level: u8 = if (!inside) 0 else if (edge) 255 else 190;
            const p = pixels[(y * 64 + x) * 4 ..][0..4];
            p[0] = level;
            p[1] = level;
            p[2] = level;
            p[3] = if (inside) 255 else 0;
        }
    }
    return pixels;
}

// -------------------------------------------------------------------------
// The renderer
// -------------------------------------------------------------------------

const Renderer = struct {
    device: *rhi.Device,
    /// Kept rather than thrown away: the names in it are what the pipeline
    /// was described with, and holding it means nothing has to reason about
    /// how long a driver looks at them.
    module: shader.Module,
    pipeline: rhi.Pipeline,
    quad: rhi.Buffer,
    instances: rhi.Buffer,
    frame: rhi.Buffer,
    atlas: rhi.Texture,
    sampler: rhi.Sampler,

    fn init(gpa: std.mem.Allocator, device: *rhi.Device) !Renderer {
        var log: Io.Writer.Allocating = .init(gpa);
        defer log.deinit();

        var module = shader.compile(gpa, source, &log.writer) catch |err| {
            std.debug.print("{s}\n", .{log.written()});
            return err;
        };
        errdefer module.deinit();

        const handle = device.createShader(.{
            .glsl = .{ .vertex = module.glsl.vertex, .fragment = module.glsl.fragment },
            .hlsl = .{ .vertex = module.hlsl.vertex, .fragment = module.hlsl.fragment },
            .label = "sprites",
        }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };

        // The locations and the formats are the shader's; the packing into
        // buffers is this program's. Nothing is written down twice.
        var attributes: [8]rhi.VertexAttribute = undefined;
        var strides: [2]u32 = @splat(0);
        for (module.attributes, 0..) |a, i| {
            const buffer = bufferOf(a.name);
            const format = try vertexFormat(a.ty);
            attributes[i] = .{
                .location = a.location,
                .format = format,
                .offset = strides[buffer],
                .buffer = buffer,
            };
            strides[buffer] += format.size();
        }

        const pipeline = device.createPipeline(.{
            .shader = handle,
            .attributes = attributes[0..module.attributes.len],
            .buffers = &.{
                .{ .stride = strides[0] },
                .{ .stride = strides[1], .step = .instance },
            },
            .topology = .triangle_strip,
            .blend = .alpha,
            // The two lists the shader itself wrote down, in slot order.
            .uniform_blocks = (try module.uniformBlockNames()) orelse return error.SlotsHaveHoles,
            .textures = (try module.textureNames()) orelse return error.SlotsHaveHoles,
            .label = "sprites",
        }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };

        // One quad, as a strip, in 0..1: the shader scales and moves it.
        const corners = [_]f32{ 0, 0, 1, 0, 0, 1, 1, 1 };
        const atlas = makeAtlas();
        const frame_block = module.block("Frame") orelse return error.NoFrameBlock;
        return .{
            .device = device,
            .module = module,
            .pipeline = pipeline,
            .quad = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) }),
            .instances = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf([count]Instance), .dynamic = true }),
            // The size the shader said the block was, not the size this
            // program guessed it would be.
            .frame = try device.createBuffer(.{ .kind = .uniform, .size = frame_block.size }),
            .atlas = try device.createTexture(.{ .width = 64, .height = 64, .data = &atlas }),
            .sampler = try device.createSampler(.linear),
        };
    }

    fn deinit(self: *Renderer) void {
        self.module.deinit();
        self.* = undefined;
    }

    /// One frame into `target`, which is `width` by `height`.
    fn draw(self: Renderer, target: rhi.RenderTarget, width: u32, height: u32, scene: Scene) !void {
        const device = self.device;

        // The origin at the top left, one unit a pixel, for whichever clip
        // space this device has.
        const projection = math.orthographic(.{
            .left = 0,
            .right = @floatFromInt(width),
            .bottom = @floatFromInt(height),
            .top = 0,
            .near = -1,
            .far = 1,
            .clip = device.clip(),
        });
        try device.updateBuffer(self.frame, 0, std.mem.asBytes(&projection));
        const instances = scene.instances();
        try device.updateBuffer(self.instances, 0, std.mem.asBytes(&instances));

        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = target, .clear_color = background } });
        try cmd.setPipeline(self.pipeline);
        try cmd.setVertexBuffer(0, self.quad, 0);
        try cmd.setVertexBuffer(1, self.instances, 0);
        try cmd.setUniformBuffer(0, self.frame);
        try cmd.setTexture(0, self.atlas, self.sampler);
        try cmd.draw(.{ .vertex_count = 4, .instance_count = count });
        try cmd.endPass();
        try device.submit();
    }
};

// -------------------------------------------------------------------------
// The program
// -------------------------------------------------------------------------

const Options = struct {
    backend: rhi.Backend,
    width: u32 = 960,
    height: u32 = 540,
    frames: ?u32 = null,
    capture: ?[]const u8 = null,
    at: f32 = 2.0,
    software: bool = false,
    debug: bool = false,

    fn fromArguments(init: std.process.Init, arena: std.mem.Allocator) !Options {
        var self: Options = .{ .backend = windowing.defaultBackend() };
        const arguments = try init.minimal.args.toSlice(arena);
        var i: usize = 1;
        while (i < arguments.len) : (i += 1) {
            const argument = arguments[i];
            const value = if (i + 1 < arguments.len) arguments[i + 1] else null;
            if (std.mem.eql(u8, argument, "--backend")) {
                self.backend = windowing.parseBackend(value orelse return error.MissingValue) orelse return error.UnknownBackend;
                i += 1;
            } else if (std.mem.eql(u8, argument, "--frames")) {
                self.frames = try std.fmt.parseInt(u32, value orelse return error.MissingValue, 10);
                i += 1;
            } else if (std.mem.eql(u8, argument, "--capture")) {
                self.capture = value orelse return error.MissingValue;
                i += 1;
            } else if (std.mem.eql(u8, argument, "--at")) {
                self.at = try std.fmt.parseFloat(f32, value orelse return error.MissingValue);
                i += 1;
            } else if (std.mem.eql(u8, argument, "--width")) {
                self.width = try std.fmt.parseInt(u32, value orelse return error.MissingValue, 10);
                i += 1;
            } else if (std.mem.eql(u8, argument, "--height")) {
                self.height = try std.fmt.parseInt(u32, value orelse return error.MissingValue, 10);
                i += 1;
            } else if (std.mem.eql(u8, argument, "--software")) {
                self.software = true;
            } else if (std.mem.eql(u8, argument, "--debug")) {
                self.debug = true;
            } else {
                return error.UnknownArgument;
            }
        }
        return self;
    }
};

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout.interface;
    const gpa = init.gpa;

    const options = try Options.fromArguments(init, init.arena.allocator());

    // A window either way: the OpenGL backend has no context without one, and
    // the capture path keeps it hidden.
    var window = Window.open(.{
        .backend = options.backend,
        .title = "Fluxion RHI - sprites",
        .width = options.width,
        .height = options.height,
        .visible = options.capture == null,
    }) catch |err| {
        try out.print("no window for {t}: {t}\n", .{ options.backend, err });
        try out.flush();
        return err;
    };
    defer window.close();

    var device = window.openDevice(gpa, .{ .debug = options.debug, .software = options.software }) catch |err| {
        try out.print("no {t} device on this machine: {t}\n", .{ options.backend, err });
        try out.flush();
        return err;
    };
    defer device.deinit();
    try out.print("{f}\n", .{device.info()});

    var renderer = Renderer.init(gpa, &device) catch |err| {
        try out.print("the shader did not become a pipeline: {t}\n", .{err});
        try out.flush();
        return err;
    };
    defer renderer.deinit();
    var scene: Scene = .init(options.width, options.height);

    if (options.capture) |path| {
        // The whole picture at one moment, into a texture, then to a file.
        scene.step(options.at);
        const target = try device.createTexture(.{ .width = options.width, .height = options.height, .usage = .{ .render_target = true } });
        try renderer.draw(.{ .texture = target }, options.width, options.height, scene);
        const pixels = try device.readTexture(target, gpa);
        defer gpa.free(pixels);
        try image.png.writeFile(gpa, init.io, path, .{
            .width = options.width,
            .height = options.height,
            .pixels = pixels,
            .row_pitch = @as(usize, options.width) * 4,
        }, .{});
        try out.print("wrote {s}, {d} by {d}, at {d:.2} seconds, {d} sprites in one draw\n", .{ path, options.width, options.height, options.at, count });
        return out.flush();
    }

    const surface = try window.createSurface(&device);
    try out.print("{d} sprites, one draw call a frame - escape or close the window to quit\n", .{count});
    try out.flush();

    const started = Io.Timestamp.now(init.io, .awake).nanoseconds;
    var last = started;
    var frames: u32 = 0;
    while (window.pump()) {
        if (window.minimised()) continue;
        if (window.takeResize()) {
            try device.resizeSurface(surface, window.width, window.height);
            scene.resize(window.width, window.height);
        }
        const now = Io.Timestamp.now(init.io, .awake).nanoseconds;
        scene.step(@as(f32, @floatFromInt(now - last)) / std.time.ns_per_s);
        last = now;

        try renderer.draw(.{ .surface = surface }, window.width, window.height, scene);
        try device.present(surface);

        frames += 1;
        if (options.frames) |limit| if (frames >= limit) break;
    }
    try out.print("{d} frames\n", .{frames});
    try out.flush();
}

// -------------------------------------------------------------------------
// Tests: the same frame on every backend this machine has
// -------------------------------------------------------------------------

const testing = std.testing;

/// Big enough that two dozen sprites leave most of it uncovered.
const test_width = 640;
const test_height = 360;

fn frameOn(backend: rhi.Backend, gpa: std.mem.Allocator) ![]u8 {
    var fixture = try windowing.TestDevice.open(backend);
    defer fixture.close();
    var device = &fixture.device;

    var renderer = try Renderer.init(gpa, device);
    defer renderer.deinit();
    var scene: Scene = .init(test_width, test_height);
    scene.step(1.0);

    const target = try device.createTexture(.{ .width = test_width, .height = test_height, .usage = .{ .render_target = true } });
    try renderer.draw(.{ .texture = target }, test_width, test_height, scene);
    return device.readTexture(target, gpa);
}

/// Is this pixel the background, give or take the rounding a rasteriser is
/// entitled to?
fn isBackground(p: *const [4]u8) bool {
    inline for (0..3) |ch| {
        const expected: i32 = @intFromFloat(@round(background[ch] * 255));
        if (@abs(@as(i32, p[ch]) - expected) > 1) return false;
    }
    return true;
}

fn checkFrame(pixels: []const u8, scene: Scene) !void {
    // A pixel inside every sprite's disc is not the background: the atlas is
    // white where it is anything, and the tint is never black.
    for (scene.sprites) |s| {
        const x: usize = @intFromFloat(s.x + s.size / 2);
        const y: usize = @intFromFloat(s.y + s.size / 2);
        if (s.cell == 1) continue; // the ring is empty in the middle
        try testing.expect(!isBackground(pixels[(y * test_width + x) * 4 ..][0..4]));
    }
    // And the picture has the background in it somewhere: sprites do not
    // cover everything, and the alpha at the corners lets it through.
    var background_pixels: usize = 0;
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) {
        if (isBackground(pixels[i..][0..4])) background_pixels += 1;
    }
    try testing.expect(background_pixels > test_width * test_height / 4);
}

test "the sprites on Direct3D 11" {
    const pixels = try frameOn(.d3d11, testing.allocator);
    defer testing.allocator.free(pixels);
    var scene: Scene = .init(test_width, test_height);
    scene.step(1.0);
    try checkFrame(pixels, scene);
}

test "the sprites on OpenGL" {
    const pixels = try frameOn(.gl, testing.allocator);
    defer testing.allocator.free(pixels);
    var scene: Scene = .init(test_width, test_height);
    scene.step(1.0);
    try checkFrame(pixels, scene);
}

test "both backends draw the same picture" {
    // Not pixel-identical - two rasterisers round differently at the edges of
    // a disc - but the same picture: the great majority of pixels agree to
    // within a little.
    const a = frameOn(.d3d11, testing.allocator) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer testing.allocator.free(a);
    const b = frameOn(.gl, testing.allocator) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer testing.allocator.free(b);

    var agree: usize = 0;
    var total: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 4) {
        total += 1;
        var close = true;
        for (0..3) |ch| {
            const d = @abs(@as(i32, a[i + ch]) - @as(i32, b[i + ch]));
            if (d > 8) close = false;
        }
        if (close) agree += 1;
    }
    try testing.expect(agree * 100 / total >= 97);
}
