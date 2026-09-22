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
//!
//! **The 3D checks** follow, each drawing or writing something a stub cannot
//! see and reading it back:
//!
//!   * every **mip level** of a texture written with a colour of its own, and
//!     read back at its own size;
//!   * `generateMips` on a checkerboard, whose last level is their average;
//!   * the six faces of a **cube** written, one rewritten, one drawn into;
//!   * the **slices** of a volume, at two levels, and the **layers** of an
//!     array - with a pass drawn into one layer and into a level below;
//!   * a pass into a **level** of an ordinary texture;
//!   * a **multisampled** target resolved into a texture, whose edge pixels
//!     are neither colour;
//!   * a **depth-only** pass, then a colour pass that depth-tests against it,
//!     then the depth texture **sampled**.
//!
//! What the device says it cannot do - `caps`, which the backend computes -
//! is **skipped** and counted, and the page says how many: a browser that has
//! no float render targets, or a build of `fluxion-webgl` with no `texImage3D`,
//! is a reason to skip and not a failure. The page switches on the extensions
//! the caps can see (`index.html`).

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
    /// Checks the device said it could not do.
    skipped: u32 = 0,
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

/// How many checks were left out because the device's caps said no.
export fn skipped() u32 {
    return state.skipped;
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
    // Debug, so that a call WebGL refused is an error here and not a picture
    // that is quietly wrong: the queue is drained after every submit.
    var device = try rhi.Device.init(gpa, .{ .backend = .webgl, .debug = true });
    errdefer device.deinit();
    std.log.info("{f}", .{device.info()});
    logCaps(device.caps());

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
    if (webgl.is_wasm) try selfCheck(&state);
}

/// What the device says it can do, to the console: the limits, and a line per
/// format that can be sampled or drawn into.
fn logCaps(caps: *const rhi.types.Caps) void {
    const limits = caps.limits;
    std.log.info("caps: 2d {d}, cube {d}, 3d {d}, layers {d}, anisotropy {d}, colour attachments {d}; border {}, lod bias {}", .{
        limits.max_texture_2d,
        limits.max_texture_cube,
        limits.max_texture_3d,
        limits.max_texture_layers,
        limits.max_anisotropy,
        limits.max_color_attachments,
        caps.features.sampler_border,
        caps.features.sampler_lod_bias,
    });
    for (std.enums.values(rhi.Format)) |format| {
        const f = caps.formatSupport(format);
        if (!f.sampled and !f.render_target) continue;
        std.log.info("caps: {s}: sampled {}, filterable {}, render target {}, blendable {}, mips {}, samples {b}", .{
            @tagName(format), f.sampled, f.filterable, f.render_target, f.blendable, f.generate_mips, f.sample_counts,
        });
    }
}

/// Every check there is: the pattern, then the 3D ones.
fn selfCheck(s: *State) !void {
    try check(s);
    try checkThreeD(s);
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
// The 3D checks
// -------------------------------------------------------------------------

const types = rhi.types;

const red = [4]u8{ 255, 0, 0, 255 };
const green = [4]u8{ 0, 255, 0, 255 };
const yellow = [4]u8{ 255, 255, 0, 255 };
const cyan = [4]u8{ 0, 255, 255, 255 };
const magenta = [4]u8{ 255, 0, 255, 255 };
const white = [4]u8{ 255, 255, 255, 255 };

/// One vertex of the triangles the checks draw: where, and in what colour.
const Vertex = extern struct { position: [3]f32, tint: [4]f32 };

const vertex_layout = [_]rhi.VertexBufferLayout{.{ .stride = @sizeOf(Vertex) }};
const solid_attributes = [_]rhi.VertexAttribute{
    .{ .location = 0, .format = .float3, .offset = @offsetOf(Vertex, "position") },
    .{ .location = 1, .format = .float4, .offset = @offsetOf(Vertex, "tint") },
};
const position_attribute = [_]rhi.VertexAttribute{
    .{ .location = 0, .format = .float3, .offset = @offsetOf(Vertex, "position") },
};

const solid_vertex_source =
    \\#version 300 es
    \\precision highp float;
    \\layout(location = 0) in vec3 position;
    \\layout(location = 1) in vec4 tint;
    \\out vec4 painted;
    \\void main() {
    \\    gl_Position = vec4(position, 1.0);
    \\    painted = tint;
    \\}
;

const solid_fragment_source =
    \\#version 300 es
    \\precision highp float;
    \\in vec4 painted;
    \\out vec4 colour;
    \\void main() {
    \\    colour = painted;
    \\}
;

/// Reads a texture at the place on the screen where it is drawn - clip space
/// is texture space, halved and moved - and shows what it read.
const reader_vertex_source =
    \\#version 300 es
    \\precision highp float;
    \\layout(location = 0) in vec3 position;
    \\out vec2 uv;
    \\void main() {
    \\    gl_Position = vec4(position, 1.0);
    \\    uv = position.xy * 0.5 + 0.5;
    \\}
;

const reader_fragment_source =
    \\#version 300 es
    \\precision highp float;
    \\precision highp sampler2D;
    \\uniform sampler2D map;
    \\in vec2 uv;
    \\out vec4 colour;
    \\void main() {
    \\    colour = texture(map, uv);
    \\}
;

/// The same, through a shadow sampler: whether a reference depth of a half is
/// in front of what the depth texture holds there, as black or white.
const shadow_fragment_source =
    \\#version 300 es
    \\precision highp float;
    \\precision highp sampler2DShadow;
    \\uniform sampler2DShadow map;
    \\in vec2 uv;
    \\out vec4 colour;
    \\void main() {
    \\    colour = vec4(vec3(texture(map, vec3(uv, 0.5))), 1.0);
    \\}
;

fn unit(rgba: [4]u8) [4]f32 {
    var out: [4]f32 = undefined;
    for (rgba, &out) |byte, *channel| channel.* = @as(f32, @floatFromInt(byte)) / 255;
    return out;
}

/// The lower-left half of clip space - the diagonal runs from the bottom right
/// to the top left - at clip depth `z`.
fn halfTriangle(z: f32, tint: [4]u8) [3]Vertex {
    const t = unit(tint);
    return .{
        .{ .position = .{ -1, -1, z }, .tint = t },
        .{ .position = .{ 1, -1, z }, .tint = t },
        .{ .position = .{ -1, 1, z }, .tint = t },
    };
}

/// All of clip space, and then some, at clip depth `z`.
fn wholeTriangle(z: f32, tint: [4]u8) [3]Vertex {
    const t = unit(tint);
    return .{
        .{ .position = .{ -1, -1, z }, .tint = t },
        .{ .position = .{ 3, -1, z }, .tint = t },
        .{ .position = .{ -1, 3, z }, .tint = t },
    };
}

/// What the checks draw with: two shaders and the pipelines they need, made
/// once.
const Kit = struct {
    device: *rhi.Device,
    /// The shaders live as long as the pipelines made from them do.
    solid: rhi.Shader,
    reader: rhi.Shader,
    shadow_reader: rhi.Shader,
    /// Colour only.
    flat: rhi.Pipeline,
    /// Colour, tested against a depth texture and not written to it.
    tested: rhi.Pipeline,
    /// Depth and no colour.
    shadow: rhi.Pipeline,
    /// A texture as it reads, through the sampler bound to it.
    show: rhi.Pipeline,
    /// A depth texture compared against a half, through a shadow sampler.
    shadowed: rhi.Pipeline,
    /// The most ordinary sampler there is, which a depth texture is read
    /// through here - and is not, on WebGL, allowed to filter.
    linear: rhi.Sampler,
    /// A shadow sampler: it compares, and filters the comparisons.
    comparing: rhi.Sampler,

    fn init(device: *rhi.Device) !Kit {
        const solid = try device.createShader(.{ .glsl_es = .{ .vertex = solid_vertex_source, .fragment = solid_fragment_source } });
        const reader = try device.createShader(.{ .glsl_es = .{ .vertex = reader_vertex_source, .fragment = reader_fragment_source } });
        const shadow_reader = try device.createShader(.{ .glsl_es = .{ .vertex = reader_vertex_source, .fragment = shadow_fragment_source } });
        return .{
            .device = device,
            .solid = solid,
            .reader = reader,
            .shadow_reader = shadow_reader,
            .flat = try device.createPipeline(.{ .shader = solid, .attributes = &solid_attributes, .buffers = &vertex_layout }),
            .tested = try device.createPipeline(.{
                .shader = solid,
                .attributes = &solid_attributes,
                .buffers = &vertex_layout,
                .depth_format = .depth32_float,
                .depth = .{ .test_enabled = true, .write = false, .compare = .less },
            }),
            .shadow = try device.createPipeline(.{
                .shader = solid,
                .attributes = &solid_attributes,
                .buffers = &vertex_layout,
                .color_format = null,
                .depth_format = .depth32_float,
                .depth = .standard,
            }),
            .show = try device.createPipeline(.{
                .shader = reader,
                .attributes = &position_attribute,
                .buffers = &vertex_layout,
                .textures = &.{"map"},
            }),
            .shadowed = try device.createPipeline(.{
                .shader = shadow_reader,
                .attributes = &position_attribute,
                .buffers = &vertex_layout,
                .textures = &.{"map"},
            }),
            .linear = try device.createSampler(.linear),
            .comparing = try device.createSampler(.{ .compare = .less }),
        };
    }

    fn deinit(self: Kit) void {
        self.device.destroySampler(self.comparing);
        self.device.destroySampler(self.linear);
        self.device.destroyPipeline(self.shadowed);
        self.device.destroyPipeline(self.show);
        self.device.destroyPipeline(self.shadow);
        self.device.destroyPipeline(self.tested);
        self.device.destroyPipeline(self.flat);
        self.device.destroyShader(self.shadow_reader);
        self.device.destroyShader(self.reader);
        self.device.destroyShader(self.solid);
    }

    /// One pass, one draw of `vertices`, and it is submitted.
    fn draw(self: Kit, pass: rhi.RenderPassDesc, pipeline: rhi.Pipeline, vertices: []const Vertex, bound: ?rhi.Command.TextureBinding) !void {
        const buffer = try self.device.createBuffer(.{ .kind = .vertex, .size = vertices.len * @sizeOf(Vertex), .data = std.mem.sliceAsBytes(vertices) });
        defer self.device.destroyBuffer(buffer);

        const cmd = self.device.begin();
        try cmd.beginPass(pass);
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, buffer, 0);
        if (bound) |b| try cmd.setTexture(b.slot, b.texture, b.sampler);
        try cmd.draw(.{ .vertex_count = @intCast(vertices.len) });
        try cmd.endPass();
        try self.device.submit();
    }
};

// What the checks say ---------------------------------------------------

/// A check that came out as it should, or did not.
fn record(s: *State, ok: bool, what: []const u8, detail: []const u8) void {
    s.checks += 1;
    if (ok) {
        std.log.info("ok: {s}", .{what});
    } else {
        s.failures += 1;
        // An error in a browser, where there is a picture to be wrong about.
        // On the host the stub draws nothing, so every pixel is wrong, and a
        // test that ran these would fail - or be all noise - for that alone.
        if (webgl.is_wasm) {
            std.log.err("wrong: {s} - {s}", .{ what, detail });
        } else {
            std.log.debug("wrong: {s} - {s}", .{ what, detail });
        }
    }
}

/// A texel of `got` that should be `want`.
fn expectTexel(s: *State, what: []const u8, got: [4]u8, want: [4]u8) void {
    var detail: [96]u8 = undefined;
    const text = std.fmt.bufPrint(&detail, "wanted {any}, got {any}", .{ want, got }) catch "";
    record(s, near(got, want), what, text);
}

/// A check the device said it cannot do, and so was not tried.
fn skip(s: *State, what: []const u8, why: []const u8) void {
    s.skipped += 1;
    std.log.info("skipped: {s} - {s}", .{ what, why });
}

fn label(buffer: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buffer, fmt, args) catch fmt;
}

fn texelAt(pixels: []const u8, width: usize, x: usize, y: usize) [4]u8 {
    return pixels[(y * width + x) * 4 ..][0..4].*;
}

fn fill(bytes: []u8, rgba: [4]u8) void {
    var at: usize = 0;
    while (at + 4 <= bytes.len) : (at += 4) bytes[at..][0..4].* = rgba;
}

/// The corner and the middle of an image of `width` by `height`, each of
/// which should be `want`. One check for all of them: an image that is
/// partly right is wrong.
fn expectImage(s: *State, what: []const u8, pixels: []const u8, width: usize, height: usize, want: [4]u8) void {
    var detail: [96]u8 = undefined;
    if (pixels.len != width * height * 4) {
        record(s, false, what, label(&detail, "{d} bytes for {d} by {d}", .{ pixels.len, width, height }));
        return;
    }
    const probes = [_][2]usize{ .{ 0, 0 }, .{ width - 1, height - 1 }, .{ width / 2, height / 2 }, .{ width - 1, 0 } };
    for (probes) |at| {
        const got = texelAt(pixels, width, at[0], at[1]);
        if (!near(got, want)) {
            record(s, false, what, label(&detail, "at ({d}, {d}) wanted {any}, got {any}", .{ at[0], at[1], want, got }));
            return;
        }
    }
    record(s, true, what, "");
}

fn readImage(s: *State, texture: rhi.Texture, sub: types.Subresource) ![]u8 {
    return s.device.readSubresource(texture, sub, gpa);
}

// The checks -------------------------------------------------------------

fn checkThreeD(s: *State) !void {
    const kit = try Kit.init(&s.device);
    defer kit.deinit();

    try checkMipWrites(s);
    try checkUploads(s);
    try checkGenerate(s);
    try checkLevelPass(s, kit);
    try checkCube(s, kit);
    try checkVolume(s);
    try checkLayers(s, kit);
    try checkMultisample(s, kit);
    try checkDepth(s, kit);
    try checkSamplers(s, kit);
    try checkReadback(s);
    try checkFormatUploads(s);
    checkCompressed(s);
}

/// Every level of a chain gets a colour of its own, and reads back as it.
fn checkMipWrites(s: *State) !void {
    var buffer: [96]u8 = undefined;
    const texture = try s.device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(texture);

    const colours = [_][4]u8{ red, green, blue, yellow };
    for (colours, 0..) |colour, level| {
        const size = types.mipExtent(8, @intCast(level));
        const bytes = try gpa.alloc(u8, size * size * 4);
        defer gpa.free(bytes);
        fill(bytes, colour);
        try s.device.writeTexture(texture, .{ .mip = @intCast(level) }, bytes, 0, 0);
    }
    for (colours, 0..) |colour, level| {
        const size = types.mipExtent(8, @intCast(level));
        const pixels = try readImage(s, texture, .{ .mip = @intCast(level) });
        defer gpa.free(pixels);
        expectImage(s, label(&buffer, "mip level {d} of a chain, written and read back at its own size", .{level}), pixels, size, size, colour);
    }
}

/// What an upload puts where: rows in the order they were given, from a row
/// pitch wider than the row, and into a corner.
fn checkUploads(s: *State) !void {
    const usage: rhi.TextureUsage = .{ .sampled = true, .render_target = true };

    // Red over blue, one texel wide. A picture written top row first reads
    // back the same way up, and is not turned over the way a drawn one is.
    {
        var bytes: [8]u8 = undefined;
        bytes[0..4].* = red;
        bytes[4..8].* = blue;
        const texture = try s.device.createTexture(.{ .width = 1, .height = 2, .usage = usage, .data = &bytes });
        defer s.device.destroyTexture(texture);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectTexel(s, "an uploaded picture reads back top row first: the top", texelAt(pixels, 1, 0, 0), red);
        expectTexel(s, "and the bottom", texelAt(pixels, 1, 0, 1), blue);
    }

    // Four texels of red and then a row of padding four texels wide - a
    // pitch the row length can say.
    {
        var bytes: [32 + 16]u8 = @splat(0);
        bytes[0..16].* = [_]u8{ 255, 0, 0, 255 } ** 4;
        bytes[16..32].* = [_]u8{ 0, 255, 0, 255 } ** 4;
        bytes[32..48].* = [_]u8{ 0, 0, 255, 255 } ** 4;
        const texture = try s.device.createTexture(.{ .width = 4, .height = 2, .usage = usage, .data = &bytes, .row_pitch = 32 });
        defer s.device.destroyTexture(texture);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectTexel(s, "a row pitch of eight texels for a row of four: the first row, to its end", texelAt(pixels, 4, 3, 0), red);
        expectTexel(s, "and the second row, which starts after the padding", texelAt(pixels, 4, 0, 1), blue);
    }

    // Two texels and two bytes of padding: not a whole number of texels, so a
    // row length cannot say it and the rows are gathered first.
    {
        var bytes: [10 + 8]u8 = @splat(0);
        bytes[0..8].* = [_]u8{ 255, 0, 0, 255 } ** 2;
        bytes[8..10].* = .{ 9, 9 };
        bytes[10..18].* = [_]u8{ 0, 0, 255, 255 } ** 2;
        const texture = try s.device.createTexture(.{ .width = 2, .height = 2, .usage = usage, .data = &bytes, .row_pitch = 10 });
        defer s.device.destroyTexture(texture);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectTexel(s, "a row pitch that is not a whole number of texels: the first row", texelAt(pixels, 2, 1, 0), red);
        expectTexel(s, "and the second", texelAt(pixels, 2, 0, 1), blue);
    }

    // BGRA and padded at once: swapped, and from the right places.
    {
        var bytes: [16 + 8]u8 = @splat(0);
        bytes[0..8].* = [_]u8{ 0, 128, 255, 255 } ** 2;
        bytes[16..24].* = [_]u8{ 255, 0, 0, 255 } ** 2;
        const texture = try s.device.createTexture(.{ .width = 2, .height = 2, .format = .bgra8_unorm, .usage = usage, .data = &bytes, .row_pitch = 16 });
        defer s.device.destroyTexture(texture);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectTexel(s, "a padded BGRA upload, swapped: orange", texelAt(pixels, 2, 1, 0), orange);
        expectTexel(s, "and blue", texelAt(pixels, 2, 1, 1), blue);
    }

    // A corner of a black texture, written green.
    {
        var black_bytes: [4 * 4 * 4]u8 = undefined;
        fill(&black_bytes, .{ 0, 0, 0, 255 });
        const texture = try s.device.createTexture(.{ .width = 4, .height = 4, .usage = usage, .data = &black_bytes });
        defer s.device.destroyTexture(texture);
        var patch: [2 * 2 * 4]u8 = undefined;
        fill(&patch, green);
        try s.device.writeTexture(texture, .{ .x = 2, .y = 1, .width = 2, .height = 2 }, &patch, 0, 0);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        const dark = [4]u8{ 0, 0, 0, 255 };
        expectTexel(s, "a corner written at (2, 1): its first texel", texelAt(pixels, 4, 2, 1), green);
        expectTexel(s, "its last", texelAt(pixels, 4, 3, 2), green);
        expectTexel(s, "the texel left of it is as it was", texelAt(pixels, 4, 1, 1), dark);
        expectTexel(s, "and the one below", texelAt(pixels, 4, 2, 3), dark);
        expectTexel(s, "and the one above", texelAt(pixels, 4, 2, 0), dark);
    }
}

/// The last level of a checkerboard is the middle of black and white.
fn checkGenerate(s: *State) !void {
    const what = "generateMips on a checkerboard";
    if (!s.device.caps().formatSupport(.rgba8_unorm).generate_mips) return skip(s, what, "the caps say rgba8 cannot generate levels");

    var checker: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| checker[i * 4 ..][0..4].* = if ((i % 4 + i / 4) % 2 == 0) white else .{ 0, 0, 0, 255 };
    const texture = try s.device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true }, .data = &checker });
    defer s.device.destroyTexture(texture);

    const cmd = s.device.begin();
    try cmd.generateMips(texture);
    try s.device.submit();

    const grey = [4]u8{ 128, 128, 128, 255 };
    const top = try readImage(s, texture, .{ .mip = 2 });
    defer gpa.free(top);
    expectImage(s, "the 1x1 level of a checkerboard is their average", top, 1, 1, grey);
    const middle = try readImage(s, texture, .{ .mip = 1 });
    defer gpa.free(middle);
    expectImage(s, "and so is each texel of the 2x2 level", middle, 2, 2, grey);
}

/// A pass into a level of an ordinary texture: the level is as big as it says,
/// and the ones beside it are not touched.
fn checkLevelPass(s: *State, kit: Kit) !void {
    const texture = try s.device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(texture);
    for (0..4) |level| {
        const size = types.mipExtent(8, @intCast(level));
        const bytes = try gpa.alloc(u8, size * size * 4);
        defer gpa.free(bytes);
        fill(bytes, blue);
        try s.device.writeTexture(texture, .{ .mip = @intCast(level) }, bytes, 0, 0);
    }

    try kit.draw(.{ .color = .{ .target = .{ .texture = texture }, .mip_level = 1, .clear_color = .{ 0, 0, 0, 1 } } }, kit.flat, &wholeTriangle(0, green), null);

    const drawn = try readImage(s, texture, .{ .mip = 1 });
    defer gpa.free(drawn);
    expectImage(s, "a pass into mip level 1, drawn and read back at 4x4", drawn, 4, 4, green);
    const above = try readImage(s, texture, .{});
    defer gpa.free(above);
    expectImage(s, "the level above it is not touched", above, 8, 8, blue);
    const below = try readImage(s, texture, .{ .mip = 2 });
    defer gpa.free(below);
    expectImage(s, "nor is the level below", below, 2, 2, blue);
}

/// Six faces, one rewritten, one drawn into.
fn checkCube(s: *State, kit: Kit) !void {
    const what = "a cube";
    if (s.device.caps().limits.max_texture_cube < 4) return skip(s, what, "the caps give no cube maps");
    var buffer: [96]u8 = undefined;

    const faces = [6][4]u8{ red, green, blue, yellow, white, magenta };
    var data: [6 * 4 * 4 * 4]u8 = undefined;
    for (faces, 0..) |colour, face| fill(data[face * 64 ..][0..64], colour);
    const cube = try s.device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true }, .data = &data });
    defer s.device.destroyTexture(cube);

    // Face two is written again, on its own, in a colour of its own.
    var face_two: [4 * 4 * 4]u8 = undefined;
    fill(&face_two, orange);
    try s.device.writeTexture(cube, .{ .z = 2, .depth = 1 }, &face_two, 0, 0);
    // And face four is drawn into.
    try kit.draw(.{ .color = .{ .target = .{ .texture = cube }, .layer = 4, .clear_color = .{ 0, 0, 0, 1 } } }, kit.flat, &wholeTriangle(0, cyan), null);

    var want = faces;
    want[2] = orange;
    want[4] = cyan;
    for (want, 0..) |colour, face| {
        const pixels = try readImage(s, cube, .{ .layer = @intCast(face) });
        defer gpa.free(pixels);
        expectImage(s, label(&buffer, "face {d} of a cube, by layer", .{face}), pixels, 4, 4, colour);
    }

    // Every face's chain is made from that face, and no other.
    if (!s.device.caps().formatSupport(.rgba8_unorm).generate_mips) return skip(s, "the chain of a cube", "the caps say rgba8 cannot generate levels");
    const cmd = s.device.begin();
    try cmd.generateMips(cube);
    try s.device.submit();
    for (want, 0..) |colour, face| {
        const pixels = try readImage(s, cube, .{ .mip = 2, .layer = @intCast(face) });
        defer gpa.free(pixels);
        expectImage(s, label(&buffer, "the last level of face {d} of a cube, generated", .{face}), pixels, 1, 1, colour);
    }
}

/// The slices of a volume, at level zero and at level one.
fn checkVolume(s: *State) !void {
    const what = "a volume";
    if (s.device.caps().limits.max_texture_3d < 4) return skip(s, what, "the caps give no 3D textures");
    var buffer: [96]u8 = undefined;

    const texture = try s.device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(texture);

    const slices = [4][4]u8{ red, green, blue, yellow };
    var slice: [4 * 4 * 4]u8 = undefined;
    for (slices, 0..) |colour, z| {
        fill(&slice, colour);
        try s.device.writeTexture(texture, .{ .z = @intCast(z), .depth = 1 }, &slice, 0, 0);
    }
    // Level one is 2x2x2: two slices, of a size of their own.
    var small: [2 * 2 * 4]u8 = undefined;
    for ([_][4]u8{ cyan, magenta }, 0..) |colour, z| {
        fill(&small, colour);
        try s.device.writeTexture(texture, .{ .mip = 1, .z = @intCast(z), .depth = 1 }, &small, 0, 0);
    }

    for (slices, 0..) |colour, z| {
        const pixels = try readImage(s, texture, .{ .layer = @intCast(z) });
        defer gpa.free(pixels);
        expectImage(s, label(&buffer, "slice {d} of a volume", .{z}), pixels, 4, 4, colour);
    }
    for ([_][4]u8{ cyan, magenta }, 0..) |colour, z| {
        const pixels = try readImage(s, texture, .{ .mip = 1, .layer = @intCast(z) });
        defer gpa.free(pixels);
        expectImage(s, label(&buffer, "slice {d} of a volume at level one", .{z}), pixels, 2, 2, colour);
    }
}

/// A pass into one layer of an array, and into a level of another.
fn checkLayers(s: *State, kit: Kit) !void {
    const what = "an array";
    if (s.device.caps().limits.max_texture_layers < 3) return skip(s, what, "the caps give no array textures");

    var data: [3 * 8 * 8 * 4]u8 = undefined;
    for ([_][4]u8{ red, green, blue }, 0..) |colour, layer| fill(data[layer * 256 ..][0..256], colour);
    const texture = try s.device.createTexture(.{ .dimension = .d2_array, .width = 8, .height = 8, .depth_or_layers = 3, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true }, .data = &data });
    defer s.device.destroyTexture(texture);

    try kit.draw(.{ .color = .{ .target = .{ .texture = texture }, .layer = 1, .clear_color = .{ 0, 0, 0, 1 } } }, kit.flat, &wholeTriangle(0, magenta), null);
    try kit.draw(.{ .color = .{ .target = .{ .texture = texture }, .layer = 2, .mip_level = 1, .clear_color = .{ 0, 0, 0, 1 } } }, kit.flat, &wholeTriangle(0, yellow), null);

    const wanted = [_]struct { rhi.types.Subresource, usize, [4]u8, []const u8 }{
        .{ .{ .layer = 1 }, 8, magenta, "layer 1 of an array, drawn into" },
        .{ .{ .layer = 0 }, 8, red, "layer 0 is not touched by it" },
        .{ .{ .layer = 2 }, 8, blue, "nor is layer 2 at level zero" },
        .{ .{ .layer = 2, .mip = 1 }, 4, yellow, "layer 2 at level 1, drawn into" },
    };
    for (wanted) |row| {
        const pixels = try readImage(s, texture, row[0]);
        defer gpa.free(pixels);
        expectImage(s, row[3], pixels, row[1], row[1], row[2]);
    }
}

/// A triangle drawn into four samples a pixel, resolved: the pixels on its
/// edge are half of one colour and half of the other.
fn checkMultisample(s: *State, kit: Kit) !void {
    const what = "a multisampled pass with a resolve";
    if (!s.device.caps().formatSupport(.rgba8_unorm).supportsSamples(4)) return skip(s, what, "the caps give no four-sample target");

    const many = try s.device.createTexture(.{ .width = 16, .height = 16, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    defer s.device.destroyTexture(many);
    const resolved = try s.device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(resolved);
    const pipeline = try s.device.createPipeline(.{ .shader = kit.solid, .attributes = &solid_attributes, .buffers = &vertex_layout, .samples = 4 });
    defer s.device.destroyPipeline(pipeline);

    try kit.draw(.{ .color = .{ .target = .{ .texture = many }, .resolve = .{ .texture = resolved }, .clear_color = .{ 0, 0, 0, 1 } } }, pipeline, &halfTriangle(0, green), null);

    const pixels = try readImage(s, resolved, .{});
    defer gpa.free(pixels);
    expectTexel(s, "inside the triangle, resolved", texelAt(pixels, 16, 2, 13), green);
    expectTexel(s, "outside it, resolved", texelAt(pixels, 16, 13, 2), .{ 0, 0, 0, 255 });
    // Along the diagonal every pixel is half covered. Drawn into one sample
    // it would be all or nothing, and no pixel would be in between.
    var between: usize = 0;
    for (3..11) |i| {
        const g = texelAt(pixels, 16, i, i)[1];
        if (g > 20 and g < 235) between += 1;
    }
    var detail: [96]u8 = undefined;
    record(s, between >= 4, "pixels on the edge are a mix of the two", label(&detail, "only {d} of 8 were", .{between}));
}

/// A depth-only pass, a colour pass that tests against what it wrote, and
/// the depth texture read as a picture.
fn checkDepth(s: *State, kit: Kit) !void {
    const what = "a depth-only pass";
    const support = s.device.caps().formatSupport(.depth32_float);
    if (!support.render_target or !support.sampled) return skip(s, what, "the caps give no depth texture to draw and sample");

    const depth = try s.device.createTexture(.{ .width = 16, .height = 16, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(depth);
    const colour = try s.device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(colour);

    // Nearer in the lower left: clip depth minus a half is a quarter.
    try kit.draw(.{ .depth = .{ .texture = depth } }, kit.shadow, &halfTriangle(-0.5, white), null);
    // Then everything, half way back, against that: where the first pass drew,
    // this is behind it.
    try kit.draw(.{
        .color = .{ .target = .{ .texture = colour }, .clear_color = .{ 0, 0, 0, 1 } },
        .depth = .{ .texture = depth, .load = .load },
    }, kit.tested, &wholeTriangle(0, green), null);

    const pixels = try readImage(s, colour, .{});
    defer gpa.free(pixels);
    expectTexel(s, "a colour pass behind what a depth-only pass drew is cut away", texelAt(pixels, 16, 2, 13), .{ 0, 0, 0, 255 });
    expectTexel(s, "and in front of nothing it is drawn", texelAt(pixels, 16, 13, 2), green);

    // The depth itself, through the ordinary sampler: a quarter where the
    // triangle was, and the cleared far plane elsewhere.
    const grey = try s.device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(grey);
    try kit.draw(.{ .color = .{ .target = .{ .texture = grey }, .clear_color = .{ 0, 0, 0, 1 } } }, kit.show, &wholeTriangle(0, white), .{ .slot = 0, .texture = depth, .sampler = kit.linear });
    const shown = try readImage(s, grey, .{});
    defer gpa.free(shown);
    expectTexel(s, "the depth texture, sampled: a quarter where it was drawn", texelAt(shown, 16, 2, 13), .{ 64, 0, 0, 255 });
    expectTexel(s, "and the far plane elsewhere", texelAt(shown, 16, 13, 2), .{ 255, 0, 0, 255 });

    // And through a shadow sampler, asked whether a half is in front of it:
    // no where the triangle is nearer than that, yes where nothing is.
    try kit.draw(.{ .color = .{ .target = .{ .texture = grey }, .clear_color = .{ 0, 0, 0, 1 } } }, kit.shadowed, &wholeTriangle(0, white), .{ .slot = 0, .texture = depth, .sampler = kit.comparing });
    const compared = try readImage(s, grey, .{});
    defer gpa.free(compared);
    expectTexel(s, "a shadow sampler, where the depth is nearer than the reference", texelAt(compared, 16, 2, 13), .{ 0, 0, 0, 255 });
    expectTexel(s, "and where it is not", texelAt(compared, 16, 13, 2), white);
}

/// The mip filter and the level range of a sampler choose the level that is
/// read: the same chain, drawn at the size of its first level, is its first
/// level through one sampler and its third through another.
fn checkSamplers(s: *State, kit: Kit) !void {
    const chain = try s.device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(chain);
    for ([_][4]u8{ red, green, blue, yellow }, 0..) |colour, level| {
        const size = types.mipExtent(8, @intCast(level));
        const bytes = try gpa.alloc(u8, size * size * 4);
        defer gpa.free(bytes);
        fill(bytes, colour);
        try s.device.writeTexture(chain, .{ .mip = @intCast(level) }, bytes, 0, 0);
    }
    const target = try s.device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .sampled = true, .render_target = true } });
    defer s.device.destroyTexture(target);

    const cases = [_]struct { rhi.SamplerDesc, [4]u8, []const u8 }{
        // Trilinear and as anisotropic as there is: level zero at one to one.
        .{ .{ .mip_filter = .linear, .max_anisotropy = 16 }, red, "a trilinear sampler reads the first level at one texel to a pixel" },
        // Pinned to the third level, whatever the size on the screen says.
        .{ .{ .mip_filter = .nearest, .lod_min = 2, .lod_max = 2 }, blue, "a sampler whose level range is two reads the third level" },
        // Held to the first two, it cannot go past the second - the size on
        // the screen asks for the first, so that is what it gets.
        .{ .{ .mip_filter = .linear, .lod_max = 1 }, red, "and one held to the first two reads the first here" },
    };
    for (cases) |case| {
        const sampler = try s.device.createSampler(case[0]);
        defer s.device.destroySampler(sampler);
        try kit.draw(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 0, 1 } } }, kit.show, &wholeTriangle(0, white), .{ .slot = 0, .texture = chain, .sampler = sampler });
        const pixels = try readImage(s, target, .{});
        defer gpa.free(pixels);
        expectImage(s, case[2], pixels, 8, 8, case[1]);
    }
}

/// What a clear colour reads back as, from each format that can be drawn
/// into: a float or a packed one is brought to eight bits, a one-channel one
/// is grey.
fn checkReadback(s: *State) !void {
    const Case = struct { format: rhi.Format, clear: rhi.Color, want: [4]u8 };
    const cases = [_]Case{
        .{ .format = .rgba8_unorm, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        // A colour cleared to a half is stored as the sRGB byte for a half,
        // 188, and that is what is read.
        .{ .format = .rgba8_unorm_srgb, .clear = .{ 1, 0.5, 0, 1 }, .want = .{ 255, 188, 0, 255 } },
        // Red and blue are the storage's, not swapped on the way out.
        .{ .format = .bgra8_unorm, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        .{ .format = .r8_unorm, .clear = .{ 0.5, 0, 0, 1 }, .want = .{ 128, 128, 128, 255 } },
        .{ .format = .rg8_unorm, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        .{ .format = .rgb10a2_unorm, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        .{ .format = .rgba16_float, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        .{ .format = .rgba32_float, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        .{ .format = .rg11b10_float, .clear = .{ 1, 0.5, 0, 1 }, .want = orange },
        // Out of range is clamped, not wrapped.
        .{ .format = .rgba16_float, .clear = .{ 4, -2, 0.5, 1 }, .want = .{ 255, 0, 128, 255 } },
    };
    var buffer: [96]u8 = undefined;
    for (cases) |case| {
        const what = label(&buffer, "{s} cleared and read back", .{@tagName(case.format)});
        if (!s.device.caps().formatSupport(case.format).render_target) {
            skip(s, what, "the caps say it cannot be drawn into");
            continue;
        }
        const texture = try s.device.createTexture(.{ .width = 4, .height = 4, .format = case.format, .usage = .{ .sampled = true, .render_target = true } });
        defer s.device.destroyTexture(texture);
        const cmd = s.device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = texture }, .clear_color = case.clear } });
        try cmd.endPass();
        try s.device.submit();
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectImage(s, what, pixels, 4, 4, case.want);
    }
}

/// One texel of each floating-point and packed format, written as bytes in
/// the format's own layout and read back as colour.
fn checkFormatUploads(s: *State) !void {
    const Case = struct { format: rhi.Format, bytes: []const u8, want: [4]u8 };
    const half = struct {
        const one: u16 = 0x3C00;
        const half: u16 = 0x3800;
    };
    // 10:10:10:2 with red in the low bits, and 11:11:10 floats the same way:
    // a red of one, a green of a half, and no blue.
    const rgb10a2: u32 = 1023 | (512 << 10) | (3 << 30);
    const rg11b10: u32 = 0x3C0 | (0x380 << 11);
    const cases = [_]Case{
        .{ .format = .rgba16_float, .bytes = std.mem.asBytes(&[4]u16{ half.one, half.half, 0, half.one }), .want = orange },
        .{ .format = .rg16_float, .bytes = std.mem.asBytes(&[2]u16{ half.one, half.half }), .want = orange },
        .{ .format = .r16_float, .bytes = std.mem.asBytes(&[1]u16{half.half}), .want = .{ 128, 128, 128, 255 } },
        .{ .format = .rgba32_float, .bytes = std.mem.asBytes(&[4]f32{ 1, 0.5, 0, 1 }), .want = orange },
        .{ .format = .rg32_float, .bytes = std.mem.asBytes(&[2]f32{ 1, 0.5 }), .want = orange },
        .{ .format = .r32_float, .bytes = std.mem.asBytes(&[1]f32{0.5}), .want = .{ 128, 128, 128, 255 } },
        .{ .format = .rgb10a2_unorm, .bytes = std.mem.asBytes(&rgb10a2), .want = orange },
        .{ .format = .rg11b10_float, .bytes = std.mem.asBytes(&rg11b10), .want = orange },
    };
    var buffer: [96]u8 = undefined;
    for (cases) |case| {
        const what = label(&buffer, "a texel of {s}, written in its own bytes and read back", .{@tagName(case.format)});
        if (!s.device.caps().formatSupport(case.format).render_target) {
            skip(s, what, "the caps say it cannot be drawn into, so it cannot be read");
            continue;
        }
        const texture = try s.device.createTexture(.{ .width = 1, .height = 1, .format = case.format, .usage = .{ .sampled = true, .render_target = true }, .data = case.bytes });
        defer s.device.destroyTexture(texture);
        const pixels = try readImage(s, texture, .{});
        defer gpa.free(pixels);
        expectImage(s, what, pixels, 1, 1, case.want);
    }
}

/// A block of a compressed format goes in, if the device has one.
fn checkCompressed(s: *State) void {
    const what = "a compressed format";
    for (std.enums.values(rhi.Format)) |format| {
        if (!format.isCompressed() or !s.device.caps().formatSupport(format).sampled) continue;
        const block = [_]u8{0} ** 16;
        const bytes = block[0..format.info().block_bytes];
        const texture = s.device.createTexture(.{ .width = 4, .height = 4, .format = format, .data = bytes }) catch |err| {
            var detail: [96]u8 = undefined;
            return record(s, false, what, label(&detail, "{s}: {s}", .{ @tagName(format), @errorName(err) }));
        };
        defer s.device.destroyTexture(texture);
        var detail: [96]u8 = undefined;
        const written = if (s.device.writeTexture(texture, .{}, bytes, 0, 0)) |_| true else |_| false;
        return record(s, written, what, label(&detail, "{s} could not be written", .{@tagName(format)}));
    }
    skip(s, what, "the caps give no compressed format");
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

test "the 3D checks run, on what the stub says it can do, and leave nothing behind" {
    webgl.stub.reset();
    defer webgl.stub.reset();

    try testing.expect(init());
    const baseline = webgl.stub.state.live_objects;
    // The stub draws nothing, so what the checks find is not the point -
    // that they run through every call, without one being refused, is.
    try checkThreeD(&state);
    // All that the checks made is gone, but for the framebuffer the first
    // pass made, which the device keeps until it is closed.
    try testing.expectEqual(baseline + 1, webgl.stub.state.live_objects);
    // Four levels written, thirteen texels of uploads looked at, two of a chain made, three around a
    // pass into a level, six faces and the last level of each, four of a
    // depth pass and its sampling and two of a shadow sampler, three
    // samplers, ten formats cleared and read and eight written in their own bytes - and four more the stub's caps
    // rule out: a volume, an array, four samples, and a compressed format.
    try testing.expectEqual(61, state.checks);
    try testing.expectEqual(4, state.skipped);

    deinit();
    try testing.expectEqual(0, webgl.stub.state.live_objects);
}
