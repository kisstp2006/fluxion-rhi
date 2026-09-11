// SPDX-License-Identifier: BSD-2-Clause

//! The WebGL backend in a browser: a check of its own work, then a frame on
//! the canvas every frame after.
//!
//! ```bash
//! zig build example-web
//! python -m http.server 8000 --directory zig-out/web
//! ```
//!
//! and open `http://localhost:8000`. A page cannot load a module from
//! `file://`, which is the only reason for the server.
//!
//! **The check is what a browser can answer and the stub cannot.** Before
//! anything reaches the canvas, `init` draws a pattern into a texture of 64
//! by 64 and reads it back:
//!
//!   * an orange cell in the top-left quarter and a blue one in the bottom
//!     right, in one instanced draw, from an atlas uploaded as **BGRA** - so
//!     red and blue swapped the wrong way round is the wrong colour;
//!   * an orange cell in the bottom-left quarter from an **indexed** draw
//!     whose square starts four vertices into its buffer - a **base
//!     vertex**, which WebGL has not got and the backend makes out of the
//!     attribute pointers;
//!   * blue in the top-right corner through a **scissor** rectangle, and
//!     black beside it - which is the check that the rectangle counts from
//!     the top.
//!
//! Every cell is chosen by an **integer** attribute, the target size and the
//! cell width come from a **uniform block**, and the atlas is read through a
//! **sampler**. What the check found goes to the console, and the page puts
//! the verdict on screen. Then `frame` draws the same pattern on the canvas
//! four times the size - the scissor with it - and sprites from the same
//! atlas crossing the rest of it, blended.

const std = @import("std");
const rhi = @import("fluxion_rhi");
const webgl = @import("fluxion_webgl");

/// Send `std.log` to the browser console.
pub const std_options: std.Options = .{ .logFn = webgl.host.logFn };

/// Say what happened before trapping - in a browser. On the host, where the
/// tests below run, the test runner's own handler is the useful one.
pub const panic = if (webgl.is_wasm)
    webgl.host.panic
else
    std.debug.FullPanic(std.debug.defaultPanic);

const gpa = if (webgl.is_wasm) std.heap.wasm_allocator else std.testing.allocator;

// -------------------------------------------------------------------------
// What the shader reads
// -------------------------------------------------------------------------

/// One cell on screen: where, how big, and which picture in the atlas.
/// `extern`, so the offsets in the pipeline are the ones written here.
const Instance = extern struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    cell: u32,
};

/// The uniform block, laid out `std140`: two-float fields on eight bytes,
/// the four-float one on sixteen.
const Frame = extern struct {
    target_size: [2]f32,
    /// How wide one cell of the atlas is, in texture coordinates.
    cell_scale: [2]f32,
    /// Where the pattern's pixel (0, 0) goes, and how big a pixel of it is.
    origin: [2]f32 = .{ 0, 0 },
    scale: [2]f32 = .{ 1, 1 },
    tint: [4]f32 = .{ 1, 1, 1, 1 },
};

const vertex_source =
    \\#version 300 es
    \\precision highp float;
    \\precision highp int;
    \\
    \\layout(location = 0) in vec2 corner;
    \\layout(location = 1) in vec4 rect;
    \\layout(location = 2) in uint cell;
    \\
    \\layout(std140) uniform Frame {
    \\    vec2 target_size;
    \\    vec2 cell_scale;
    \\    vec2 origin;
    \\    vec2 scale;
    \\    vec4 tint;
    \\};
    \\
    \\out vec2 uv;
    \\
    \\void main() {
    \\    vec2 pixel = origin + (rect.xy + corner * rect.zw) * scale;
    \\    vec2 clip = pixel / target_size * 2.0 - 1.0;
    \\    uv = (corner + vec2(float(cell), 0.0)) * cell_scale;
    \\    // Pixels count down from the top; clip space counts up.
    \\    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    \\}
;

const fragment_source =
    \\#version 300 es
    \\precision highp float;
    \\precision highp int;
    \\precision highp sampler2D;
    \\
    \\layout(std140) uniform Frame {
    \\    vec2 target_size;
    \\    vec2 cell_scale;
    \\    vec2 origin;
    \\    vec2 scale;
    \\    vec4 tint;
    \\};
    \\
    \\uniform sampler2D atlas;
    \\
    \\in vec2 uv;
    \\out vec4 colour;
    \\
    \\void main() {
    \\    colour = texture(atlas, uv) * tint;
    \\}
;

// -------------------------------------------------------------------------
// The pattern
// -------------------------------------------------------------------------

/// Two triangles over the unit square.
const quad = [_][2]f32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };

/// The same square as four corners, with four vertices of nothing in front of
/// them. Only a base vertex of four reaches the square; without one the six
/// indices draw two triangles with no area, and the cell stays black.
const offset_corners = [_][2]f32{
    .{ -9, -9 }, .{ -9, -9 }, .{ -9, -9 }, .{ -9, -9 },
    .{ 0, 0 },   .{ 1, 0 },   .{ 0, 1 },   .{ 1, 1 },
};
const corner_indices = [_]u16{ 0, 1, 2, 2, 1, 3 };

/// Orange and blue, one texel each, written the BGRA way: blue first.
const atlas_bgra = [_]u8{
    0,   128, 255, 255,
    255, 0,   0,   255,
};

const orange = [4]u8{ 255, 128, 0, 255 };
const blue = [4]u8{ 0, 0, 255, 255 };
const black = [4]u8{ 0, 0, 0, 255 };

/// In pixels of the 64-pixel target.
const pattern = [_]Instance{
    .{ .x = 0, .y = 0, .w = 32, .h = 32, .cell = 0 },
    .{ .x = 32, .y = 32, .w = 32, .h = 32, .cell = 1 },
    .{ .x = 0, .y = 32, .w = 32, .h = 32, .cell = 0 },
    .{ .x = 0, .y = 0, .w = 64, .h = 64, .cell = 1 },
};
const pattern_size = 64;
const pattern_scissor: rhi.Rect = .{ .x = 48, .y = 0, .width = 16, .height = 16 };

/// Where the check looks, what it should find, and what it means if not.
const Expectation = struct { x: usize, y: usize, rgba: [4]u8, what: []const u8 };

const expectations = [_]Expectation{
    .{ .x = 8, .y = 8, .rgba = orange, .what = "an instanced draw from a BGRA atlas, top left" },
    .{ .x = 48, .y = 48, .rgba = blue, .what = "the second instance, bottom right" },
    .{ .x = 8, .y = 48, .rgba = orange, .what = "an indexed draw with a base vertex" },
    .{ .x = 56, .y = 8, .rgba = blue, .what = "a draw through a scissor rectangle" },
    .{ .x = 40, .y = 8, .rgba = black, .what = "outside the scissor, so it counts from the top" },
};

// -------------------------------------------------------------------------
// The module's state and entry points
// -------------------------------------------------------------------------

const sprite_count = 12;

const State = struct {
    device: rhi.Device,
    surface: rhi.Surface,
    pipeline: rhi.Pipeline,
    atlas: rhi.Texture,
    nearest: rhi.Sampler,
    quad: rhi.Buffer,
    offset_corners: rhi.Buffer,
    indices: rhi.Buffer,
    pattern: rhi.Buffer,
    sprites: rhi.Buffer,
    check_frame: rhi.Buffer,
    pattern_frame: rhi.Buffer,
    sprite_frame: rhi.Buffer,
    checked: rhi.Texture,
    checks: u32 = 0,
    failures: u32 = 0,
};

/// There is no `main` and no stack that lasts between two calls from the
/// page, so what lasts is here.
var state: State = undefined;

/// Open the device, make everything, and run the check. False, with the
/// reason in the console, if any of it could not be done.
export fn init() bool {
    start() catch |err| {
        std.log.err("could not start: {s}", .{@errorName(err)});
        return false;
    };
    return true;
}

/// How many checks `init` ran, and how many of them came out wrong. The page
/// shows both.
export fn checks() u32 {
    return state.checks;
}

export fn failures() u32 {
    return state.failures;
}

/// One frame on the canvas. Called from `requestAnimationFrame`.
export fn frame() void {
    draw() catch |err| std.log.err("frame: {s}: {s}", .{ @errorName(err), state.device.diagnostics() });
}

/// Give everything back. The device destroys what is still alive in it.
export fn deinit() void {
    state.device.deinit();
}

fn start() !void {
    var device = try rhi.Device.init(gpa, .{ .backend = .webgl });
    errdefer device.deinit();
    std.log.info("{f}", .{device.info()});

    const shader = device.createShader(.{ .glsl_es = .{ .vertex = vertex_source, .fragment = fragment_source } }) catch |err| {
        std.log.err("{s}", .{device.diagnostics()});
        return err;
    };
    const pipeline = device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0, .buffer = 0 },
            .{ .location = 1, .format = .float4, .offset = @offsetOf(Instance, "x"), .buffer = 1 },
            .{ .location = 2, .format = .uint, .offset = @offsetOf(Instance, "cell"), .buffer = 1 },
        },
        .buffers = &.{ .{ .stride = @sizeOf([2]f32) }, .{ .stride = @sizeOf(Instance), .step = .instance } },
        .blend = .alpha,
        .uniform_blocks = &.{"Frame"},
        .textures = &.{"atlas"},
    }) catch |err| {
        std.log.err("{s}", .{device.diagnostics()});
        return err;
    };

    const check_frame: Frame = .{ .target_size = .{ pattern_size, pattern_size }, .cell_scale = .{ 0.5, 1 } };

    state = .{
        .surface = try device.createSurface(.{}),
        .pipeline = pipeline,
        .atlas = try device.createTexture(.{ .width = 2, .height = 1, .format = .bgra8_unorm, .data = &atlas_bgra }),
        .nearest = try device.createSampler(.nearest),
        .quad = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(quad)), .data = std.mem.asBytes(&quad) }),
        .offset_corners = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(offset_corners)), .data = std.mem.asBytes(&offset_corners) }),
        .indices = try device.createBuffer(.{ .kind = .index, .size = @sizeOf(@TypeOf(corner_indices)), .data = std.mem.asBytes(&corner_indices) }),
        .pattern = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(pattern)), .data = std.mem.asBytes(&pattern) }),
        .sprites = try device.createBuffer(.{ .kind = .vertex, .size = sprite_count * @sizeOf(Instance), .dynamic = true }),
        .check_frame = try device.createBuffer(.{ .kind = .uniform, .size = @sizeOf(Frame), .data = std.mem.asBytes(&check_frame) }),
        .pattern_frame = try device.createBuffer(.{ .kind = .uniform, .size = @sizeOf(Frame) }),
        .sprite_frame = try device.createBuffer(.{ .kind = .uniform, .size = @sizeOf(Frame) }),
        .checked = try device.createTexture(.{ .width = pattern_size, .height = pattern_size, .usage = .{ .render_target = true } }),
        .device = device,
    };

    // A picture needs a GPU to be right or wrong, and the stub the host
    // tests run against has none.
    if (webgl.is_wasm) try check(&state);
}

/// Draw the pattern into the texture, read it back, and look.
fn check(s: *State) !void {
    const cmd = s.device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = s.checked }, .clear_color = .{ 0, 0, 0, 1 } } });
    try drawPattern(cmd, s, s.check_frame, pattern_scissor);
    try cmd.endPass();
    try s.device.submit();

    const pixels = try s.device.readTexture(s.checked, gpa);
    defer gpa.free(pixels);

    for (expectations) |e| {
        const at = (e.y * pattern_size + e.x) * 4;
        const got = pixels[at..][0..4].*;
        s.checks += 1;
        if (near(got, e.rgba)) {
            std.log.info("ok: {s}", .{e.what});
        } else {
            s.failures += 1;
            std.log.err("wrong: {s} - at ({d}, {d}) wanted {any}, got {any}", .{ e.what, e.x, e.y, e.rgba, got });
        }
    }
}

/// Within a rasteriser's rounding.
fn near(a: [4]u8, b: [4]u8) bool {
    for (a, b) |x, y| {
        if (@abs(@as(i16, x) - y) > 3) return false;
    }
    return true;
}

/// The four cells, however big the frame block says they are.
fn drawPattern(cmd: *rhi.CommandList, s: *const State, frame_block: rhi.Buffer, scissor: rhi.Rect) !void {
    try cmd.setPipeline(s.pipeline);
    try cmd.setUniformBuffer(0, frame_block);
    try cmd.setTexture(0, s.atlas, s.nearest);

    // Two cells, one instanced draw.
    try cmd.setVertexBuffer(0, s.quad, 0);
    try cmd.setVertexBuffer(1, s.pattern, 0);
    try cmd.draw(.{ .vertex_count = quad.len, .instance_count = 2 });

    // The third, indexed, from four vertices in.
    try cmd.setVertexBuffer(0, s.offset_corners, 0);
    try cmd.setVertexBuffer(1, s.pattern, 2 * @sizeOf(Instance));
    try cmd.setIndexBuffer(s.indices, .u16);
    try cmd.drawIndexed(.{ .index_count = corner_indices.len, .base_vertex = 4 });

    // The fourth covers everything, and is cut to one corner.
    try cmd.setVertexBuffer(0, s.quad, 0);
    try cmd.setVertexBuffer(1, s.pattern, 3 * @sizeOf(Instance));
    try cmd.setScissor(scissor);
    try cmd.draw(.{ .vertex_count = quad.len });
    try cmd.setScissor(null);
}

fn draw() !void {
    const s = &state;
    const size = try s.device.surfaceSize(s.surface);
    const width: f32 = @floatFromInt(size.width);
    const height: f32 = @floatFromInt(size.height);

    // The pattern, four times the size, a little way in from the corner -
    // and its scissor rectangle moved and grown the same way.
    const origin = 24;
    const scale = 4;
    const pattern_frame: Frame = .{
        .target_size = .{ width, height },
        .cell_scale = .{ 0.5, 1 },
        .origin = .{ origin, origin },
        .scale = .{ scale, scale },
    };
    try s.device.updateBuffer(s.pattern_frame, 0, std.mem.asBytes(&pattern_frame));
    const canvas_scissor: rhi.Rect = .{
        .x = origin + pattern_scissor.x * scale,
        .y = origin + pattern_scissor.y * scale,
        .width = pattern_scissor.width * scale,
        .height = pattern_scissor.height * scale,
    };

    // The sprites, a little transparent so the blending shows where they
    // cross.
    const sprite_frame: Frame = .{
        .target_size = .{ width, height },
        .cell_scale = .{ 0.5, 1 },
        .tint = .{ 1, 1, 1, 0.8 },
    };
    try s.device.updateBuffer(s.sprite_frame, 0, std.mem.asBytes(&sprite_frame));
    const seconds: f32 = @floatCast(webgl.now() / 1000.0);
    var sprites: [sprite_count]Instance = undefined;
    for (&sprites, 0..) |*sprite, i| sprite.* = spriteAt(i, seconds, width, height);
    try s.device.updateBuffer(s.sprites, 0, std.mem.sliceAsBytes(&sprites));

    const cmd = s.device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = s.surface }, .clear_color = .{ 0.07, 0.07, 0.09, 1 } } });
    try drawPattern(cmd, s, s.pattern_frame, canvas_scissor);
    try cmd.setUniformBuffer(0, s.sprite_frame);
    try cmd.setVertexBuffer(0, s.quad, 0);
    try cmd.setVertexBuffer(1, s.sprites, 0);
    try cmd.draw(.{ .vertex_count = quad.len, .instance_count = sprite_count });
    try cmd.endPass();
    try s.device.submit();
    try s.device.present(s.surface);
}

/// Sprite `i`, `seconds` in: bouncing around the canvas to the right of the
/// pattern, each at its own speed.
fn spriteAt(i: usize, seconds: f32, width: f32, height: f32) Instance {
    const n: f32 = @floatFromInt(i);
    const left: f32 = 24 + pattern_size * 4 + 24;
    const x = left + bounce(seconds * (70 + n * 9) + n * 97, @max(width - left - 48, 1));
    const y = bounce(seconds * (50 + n * 13) + n * 53, @max(height - 48, 1));
    return .{ .x = x, .y = y, .w = 48, .h = 48, .cell = @intCast(i % 2) };
}

/// From zero to `span` and back, for ever.
fn bounce(travelled: f32, span: f32) f32 {
    const t = @mod(travelled, 2 * span);
    return if (t < span) t else 2 * span - t;
}

// -------------------------------------------------------------------------
// Tests - on the host, against the stub. They cannot see the picture; they
// see that everything is made, drawn with and given back.
// -------------------------------------------------------------------------

const testing = std.testing;

test "the layouts are the ones the shader reads" {
    try testing.expectEqual(20, @sizeOf(Instance));
    try testing.expectEqual(16, @offsetOf(Instance, "cell"));
    // std140: the four vec2s on eight bytes each, the vec4 on sixteen.
    try testing.expectEqual(48, @sizeOf(Frame));
    try testing.expectEqual(16, @offsetOf(Frame, "origin"));
    try testing.expectEqual(32, @offsetOf(Frame, "tint"));
}

test "the check looks where the pattern put things" {
    // Each expectation is inside the cell it is about, and the black one is
    // inside none of the cells the scissor let through.
    for (expectations) |e| {
        try testing.expect(e.x < pattern_size and e.y < pattern_size);
    }
    const outside = expectations[4];
    try testing.expect(outside.x < pattern_scissor.x);
    try testing.expect(outside.y < pattern_scissor.height);
}

test "init makes everything, frame draws, and deinit gives it all back" {
    webgl.stub.reset();
    defer webgl.stub.reset();

    try testing.expect(init());
    frame();
    // The pattern is three draws and the sprites one.
    try testing.expectEqual(4, webgl.stub.state.draw_calls);
    try testing.expectEqual(sprite_count, webgl.stub.state.last_draw.instances);

    deinit();
    try testing.expectEqual(0, webgl.stub.state.live_objects);
}

test "a sprite bounces inside the canvas" {
    for (0..sprite_count) |i| {
        var t: f32 = 0;
        while (t < 30) : (t += 0.37) {
            const sprite = spriteAt(i, t, 800, 600);
            try testing.expect(sprite.x >= 0 and sprite.x + sprite.w <= 800 + 1);
            try testing.expect(sprite.y >= 0 and sprite.y + sprite.h <= 600 + 1);
        }
    }
}
