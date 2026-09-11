// SPDX-License-Identifier: BSD-2-Clause

//! Everything a program says to a device, as plain values.
//!
//! None of these know which backend they are for. A `Format` is a format on
//! OpenGL and on Direct3D alike, and the backend is the one that turns it into
//! `GL_RGBA8` or `DXGI_FORMAT_R8G8B8A8_UNORM`. That is what keeps a third
//! backend a new file rather than a change to every caller.
//!
//! **Origin is the top left.** Viewports, scissor rectangles and the pixels
//! that come back from `readTexture` all count from the top left corner, the
//! way Direct3D, Vulkan, Metal and every image file do. OpenGL counts from the
//! bottom left, and its backend does the arithmetic so that nobody else has
//! to.
//!
//! **Clip space is the device's.** The one thing a program cannot help
//! knowing is which clip space its projection is for, and `Device.clip`
//! answers that as a `fluxion-math` `Clip` - so a projection is one line,
//! built for the backend that was actually opened.

const std = @import("std");
const math = @import("fluxion_math");

const resources = @import("resources.zig");

pub const Buffer = resources.Buffer;
pub const Texture = resources.Texture;
pub const Sampler = resources.Sampler;
pub const Shader = resources.Shader;
pub const Pipeline = resources.Pipeline;
pub const Surface = resources.Surface;

/// Which backend a device is.
pub const Backend = enum {
    /// Accepts everything, draws nothing. For tests, servers and a program
    /// that wants to know its render code compiles.
    none,
    /// OpenGL 3.3 core, on a context somebody else made.
    gl,
    /// Direct3D 11, feature level 11.0. Windows only.
    d3d11,
    /// WebGL 2, in a browser, through `fluxion-webgl`. A `wasm32` build
    /// only - and, under test, any build, where it talks to that library's
    /// stub instead of a page.
    webgl,

    pub fn clip(self: Backend) math.Clip {
        return switch (self) {
            .none => .gl,
            .gl => .gl,
            .d3d11 => .d3d,
            // OpenGL ES's, which is OpenGL's: depth from -1 to 1.
            .webgl => .gl,
        };
    }
};

/// Which backend to open.
pub const Select = enum {
    /// OpenGL if `DeviceDesc.gl` was given; otherwise Direct3D 11 on Windows
    /// and WebGL on the web; and `error.Unsupported` where none of those
    /// applies. Never `none` - a program that wants nothing drawn has to say
    /// so.
    auto,
    none,
    gl,
    d3d11,
    webgl,
};

pub const Error = error{
    /// This build or this machine has no such backend.
    Unsupported,
    /// The backend is there and refused to make a device: no driver, no
    /// context current, a feature level too low.
    NoDevice,
    /// The driver stopped answering. Everything made from the device is gone.
    DeviceLost,
    /// A shader did not compile or link. `Device.diagnostics` has the log.
    ShaderFailed,
    /// A pipeline could not be made from an otherwise good shader: an
    /// attribute the shader has no input for, a format the target refuses.
    PipelineFailed,
    /// A handle that was destroyed, or never made by this device.
    InvalidHandle,
    /// A call that does not add up: a draw outside a pass, a vertex buffer
    /// where an index buffer was wanted, a texture that cannot be rendered to.
    /// `Device.diagnostics` says which.
    InvalidArgument,
    /// The driver said no and this library has no better name for why.
    Failed,
    OutOfMemory,
};

pub const Color = [4]f32;

pub const Extent = struct {
    width: u32,
    height: u32,
};

// -------------------------------------------------------------------------
// Formats
// -------------------------------------------------------------------------

/// How the bytes of a texel are laid out.
pub const Format = enum {
    rgba8_unorm,
    rgba8_unorm_srgb,
    bgra8_unorm,
    r8_unorm,
    rgba16_float,
    rgba32_float,
    depth24_stencil8,
    depth32_float,

    pub fn bytesPerPixel(self: Format) usize {
        return switch (self) {
            .r8_unorm => 1,
            .rgba8_unorm, .rgba8_unorm_srgb, .bgra8_unorm, .depth24_stencil8, .depth32_float => 4,
            .rgba16_float => 8,
            .rgba32_float => 16,
        };
    }

    pub fn isDepth(self: Format) bool {
        return switch (self) {
            .depth24_stencil8, .depth32_float => true,
            else => false,
        };
    }

    pub fn hasStencil(self: Format) bool {
        return self == .depth24_stencil8;
    }
};

/// How the bytes of one vertex attribute are laid out.
pub const VertexFormat = enum {
    float,
    float2,
    float3,
    float4,
    /// Four bytes, read as 0 to 1.
    ubyte4_norm,
    /// Four bytes, read as integers 0 to 255.
    ubyte4,
    uint,
    int,

    pub fn size(self: VertexFormat) u32 {
        return switch (self) {
            .float, .ubyte4_norm, .ubyte4, .uint, .int => 4,
            .float2 => 8,
            .float3 => 12,
            .float4 => 16,
        };
    }

    pub fn components(self: VertexFormat) u32 {
        return switch (self) {
            .float, .uint, .int => 1,
            .float2 => 2,
            .float3 => 3,
            .float4, .ubyte4_norm, .ubyte4 => 4,
        };
    }

    /// Does the shader see this as `int`/`uint` rather than `float`?
    pub fn isInteger(self: VertexFormat) bool {
        return switch (self) {
            .ubyte4, .uint, .int => true,
            else => false,
        };
    }
};

pub const IndexFormat = enum {
    u16,
    u32,

    pub fn size(self: IndexFormat) u32 {
        return switch (self) {
            .u16 => 2,
            .u32 => 4,
        };
    }
};

pub const Topology = enum {
    triangles,
    triangle_strip,
    lines,
    line_strip,
    points,
};

// -------------------------------------------------------------------------
// Resources
// -------------------------------------------------------------------------

pub const BufferKind = enum {
    vertex,
    index,
    /// A constant buffer, in Direct3D's words; a uniform block in OpenGL's.
    /// Laid out `std140`, which is the layout both agree on. Sizes are
    /// rounded up to sixteen bytes.
    uniform,
};

pub const BufferDesc = struct {
    kind: BufferKind,
    size: usize,
    /// Written every frame. What a sprite batch's instance buffer is; not
    /// what a mesh is. A uniform buffer is always treated as dynamic.
    dynamic: bool = false,
    /// Initial contents, at most `size` bytes. Copied; not kept.
    data: ?[]const u8 = null,
    /// For the debugger and the diagnostics. Not kept.
    label: []const u8 = "",
};

pub const TextureUsage = packed struct {
    /// Bound to a shader and read from.
    sampled: bool = true,
    /// Drawn into as a render pass's colour or depth attachment.
    render_target: bool = false,
};

pub const TextureDesc = struct {
    width: u32,
    height: u32,
    format: Format = .rgba8_unorm,
    usage: TextureUsage = .{},
    /// Initial texels, top row first. Copied; not kept.
    data: ?[]const u8 = null,
    /// Bytes from one row of `data` to the next. Zero means tightly packed.
    row_pitch: usize = 0,
    label: []const u8 = "",

    pub fn tightRowPitch(self: TextureDesc) usize {
        return @as(usize, self.width) * self.format.bytesPerPixel();
    }

    pub fn effectiveRowPitch(self: TextureDesc) usize {
        return if (self.row_pitch == 0) self.tightRowPitch() else self.row_pitch;
    }
};

pub const Filter = enum { nearest, linear };

pub const Wrap = enum { repeat, clamp_to_edge, mirror };

pub const SamplerDesc = struct {
    min_filter: Filter = .linear,
    mag_filter: Filter = .linear,
    wrap_u: Wrap = .clamp_to_edge,
    wrap_v: Wrap = .clamp_to_edge,

    /// What a pixel-art game wants.
    pub const nearest: SamplerDesc = .{ .min_filter = .nearest, .mag_filter = .nearest };
    pub const linear: SamplerDesc = .{};
};

/// One stage's source, in one language.
pub const ShaderStages = struct {
    vertex: [:0]const u8,
    fragment: [:0]const u8,
};

/// A shader, in whichever languages the program brought.
///
/// There is no cross-compiler here, and a backend that has no source in its
/// language fails with `error.ShaderFailed` and says so. A program that ships
/// on two backends ships two sources, or generates them - which is the seam
/// a shader compiler slots into later without this struct changing shape.
///
/// **The contract between the languages** is where things are bound. The two
/// GLSLs keep it the same way, because neither has `layout(binding = n)`:
///
/// | | GLSL 330 and GLSL ES 300 | HLSL 5.0 |
/// | --- | --- | --- |
/// | Vertex attribute at location `n` | `layout(location = n) in` | semantic `ATTRn` |
/// | Uniform buffer in slot `n` | block named in `PipelineDesc.uniform_blocks[n]`, `std140` | `register(bn)` |
/// | Texture in slot `n` | sampler named in `PipelineDesc.textures[n]` | `register(tn)` and `register(sn)` |
/// | Fragment colour | `out vec4` at location 0 | `SV_TARGET` |
pub const ShaderDesc = struct {
    /// GLSL 3.30 core, for the OpenGL backend.
    glsl: ?ShaderStages = null,
    /// GLSL ES 3.00, for the WebGL backend: `#version 300 es` on the first
    /// line, and a precision for `float` in the fragment stage.
    glsl_es: ?ShaderStages = null,
    /// HLSL for shader model 5.0, for the Direct3D 11 backend.
    hlsl: ?ShaderStages = null,
    label: []const u8 = "",
};

// -------------------------------------------------------------------------
// Pipelines
// -------------------------------------------------------------------------

pub const VertexStep = enum {
    /// Advance once per vertex.
    vertex,
    /// Advance once per instance: what a sprite batch's placement is.
    instance,
};

/// One vertex buffer slot: how far apart its vertices are, and whether it
/// steps per vertex or per instance.
pub const VertexBufferLayout = struct {
    stride: u32,
    step: VertexStep = .vertex,
};

pub const VertexAttribute = struct {
    /// `layout(location = n)` in GLSL; `ATTRn` in HLSL.
    location: u32,
    format: VertexFormat,
    /// Bytes from the start of the vertex.
    offset: u32,
    /// Which of `PipelineDesc.buffers` it reads from.
    buffer: u32 = 0,
};

pub const BlendFactor = enum {
    zero,
    one,
    src_color,
    one_minus_src_color,
    src_alpha,
    one_minus_src_alpha,
    dst_color,
    one_minus_dst_color,
    dst_alpha,
    one_minus_dst_alpha,
};

pub const BlendOp = enum { add, subtract, reverse_subtract, min, max };

pub const BlendState = struct {
    enabled: bool = false,
    src_rgb: BlendFactor = .one,
    dst_rgb: BlendFactor = .zero,
    op_rgb: BlendOp = .add,
    src_alpha: BlendFactor = .one,
    dst_alpha: BlendFactor = .zero,
    op_alpha: BlendOp = .add,

    /// No blending: what is drawn replaces what was there.
    pub const solid: BlendState = .{};

    /// Straight alpha: the usual sprite, with its alpha in the texture.
    pub const alpha: BlendState = .{
        .enabled = true,
        .src_rgb = .src_alpha,
        .dst_rgb = .one_minus_src_alpha,
        .src_alpha = .one,
        .dst_alpha = .one_minus_src_alpha,
    };

    /// Premultiplied alpha: colour already multiplied by alpha, which is what
    /// a text renderer and a compositor want because it composes correctly.
    pub const premultiplied: BlendState = .{
        .enabled = true,
        .src_rgb = .one,
        .dst_rgb = .one_minus_src_alpha,
        .src_alpha = .one,
        .dst_alpha = .one_minus_src_alpha,
    };

    /// Add light: particles, glows.
    pub const additive: BlendState = .{
        .enabled = true,
        .src_rgb = .src_alpha,
        .dst_rgb = .one,
        .src_alpha = .one,
        .dst_alpha = .one,
    };
};

pub const CompareFn = enum {
    never,
    less,
    equal,
    less_equal,
    greater,
    not_equal,
    greater_equal,
    always,
};

pub const DepthState = struct {
    test_enabled: bool = false,
    write: bool = false,
    compare: CompareFn = .less,

    /// What 2D wants: draw order decides.
    pub const none: DepthState = .{};

    /// What 3D wants: nearer wins.
    pub const standard: DepthState = .{ .test_enabled = true, .write = true, .compare = .less };
};

pub const CullMode = enum { none, back, front };

pub const FrontFace = enum { ccw, cw };

pub const PipelineDesc = struct {
    shader: Shader,
    attributes: []const VertexAttribute,
    buffers: []const VertexBufferLayout,
    topology: Topology = .triangles,
    blend: BlendState = .solid,
    depth: DepthState = .none,
    cull: CullMode = .none,
    front_face: FrontFace = .ccw,
    /// The GLSLs only: the uniform block bound to each slot, by name, in slot
    /// order. HLSL binds by `register(bn)` and ignores this.
    uniform_blocks: []const [:0]const u8 = &.{},
    /// The GLSLs only: the `sampler2D` bound to each slot, by name, in slot
    /// order. HLSL binds by `register(tn)`/`register(sn)` and ignores this.
    textures: []const [:0]const u8 = &.{},
    /// What this pipeline draws into. A pass whose attachments differ is an
    /// error on the backends that check and wrong pixels on the ones that
    /// do not, so it is stated here.
    color_format: Format = .rgba8_unorm,
    depth_format: ?Format = null,
    label: []const u8 = "",
};

// -------------------------------------------------------------------------
// Passes and state
// -------------------------------------------------------------------------

/// Top-left origin, in pixels of the current attachment.
pub const Viewport = struct {
    x: f32 = 0,
    y: f32 = 0,
    width: f32,
    height: f32,
    min_depth: f32 = 0,
    max_depth: f32 = 1,
};

/// Top-left origin, in pixels of the current attachment.
pub const Rect = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
};

pub const LoadOp = enum {
    /// Start from `clear_color`.
    clear,
    /// Keep what is there. Drawing on top of a previous pass.
    load,
    /// Whatever is there is fine to lose. Cheapest on a tiled GPU, and what a
    /// pass that covers every pixel should say.
    dont_care,
};

/// What a pass draws into.
pub const RenderTarget = union(enum) {
    surface: Surface,
    texture: Texture,
};

pub const ColorAttachment = struct {
    target: RenderTarget,
    load: LoadOp = .clear,
    clear_color: Color = .{ 0, 0, 0, 1 },
};

pub const DepthAttachment = struct {
    texture: Texture,
    load: LoadOp = .clear,
    clear_depth: f32 = 1,
    clear_stencil: u8 = 0,
};

pub const RenderPassDesc = struct {
    color: ColorAttachment,
    depth: ?DepthAttachment = null,
};

// -------------------------------------------------------------------------
// Devices and surfaces
// -------------------------------------------------------------------------

/// The one GL function pointer, with whatever it needs beside it.
pub const GlProc = *const fn () callconv(.c) void;

/// What the OpenGL backend needs from whoever made the context, which is
/// never this library: `fluxion-platform`, GLFW, SDL, EGL by hand.
///
/// The context must be current on the calling thread for as long as the
/// device lives. `get_proc_address` must answer for the whole of OpenGL 3.3,
/// including the commands `wglGetProcAddress` refuses - `fluxion-platform`'s
/// window already does; a program on raw WGL chains `opengl32.dll` behind it.
pub const GlHooks = struct {
    context: *anyopaque,
    get_proc_address: *const fn (*anyopaque, [*:0]const u8) ?GlProc,
    swap_buffers: *const fn (*anyopaque) void,
    /// The default framebuffer's size in pixels, which is what a pass into
    /// the surface covers.
    framebuffer_size: *const fn (*anyopaque) [2]u32,
};

pub const DeviceDesc = struct {
    backend: Select = .auto,
    /// Required for `.gl`; makes `.auto` choose it.
    gl: ?GlHooks = null,
    /// Validation on: the driver's debug layer where there is one, and
    /// `glGetError` after every submit. Worth the cost while writing a
    /// renderer and not afterwards.
    debug: bool = false,
    /// Prefer a software rasteriser: WARP on Direct3D. The same answers on
    /// every machine, which is what a test wants.
    software: bool = false,
};

pub const SurfaceDesc = struct {
    /// The window, as the platform's integer: an `HWND` on Windows. Ignored
    /// by the OpenGL backend, whose surface is the context's own framebuffer
    /// and which therefore has exactly one - and by the WebGL backend, whose
    /// one surface is the canvas the page made the context on.
    native_window: usize = 0,
    /// Zero means "the window's size".
    width: u32 = 0,
    height: u32 = 0,
    vsync: bool = true,
};

/// What a device is, once open.
pub const Info = struct {
    backend: Backend,
    /// What the driver calls itself. Points into the device; valid while it
    /// lives.
    renderer: []const u8,

    pub fn format(self: Info, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{t}: {s}", .{ self.backend, self.renderer });
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "formats know their size" {
    try testing.expectEqual(@as(usize, 4), Format.rgba8_unorm.bytesPerPixel());
    try testing.expectEqual(@as(usize, 1), Format.r8_unorm.bytesPerPixel());
    try testing.expectEqual(@as(usize, 16), Format.rgba32_float.bytesPerPixel());
    try testing.expect(Format.depth24_stencil8.isDepth());
    try testing.expect(!Format.rgba8_unorm.isDepth());
    try testing.expectEqual(@as(u32, 12), VertexFormat.float3.size());
    try testing.expectEqual(@as(u32, 4), VertexFormat.ubyte4_norm.components());
}

test "the blend presets are the ones everybody writes by hand" {
    try testing.expect(!BlendState.solid.enabled);
    try testing.expectEqual(BlendFactor.src_alpha, BlendState.alpha.src_rgb);
    try testing.expectEqual(BlendFactor.one_minus_src_alpha, BlendState.alpha.dst_rgb);
    try testing.expectEqual(BlendFactor.one, BlendState.premultiplied.src_rgb);
    try testing.expectEqual(BlendFactor.one, BlendState.additive.dst_rgb);
}

test "a texture description knows its own pitch" {
    const d: TextureDesc = .{ .width = 10, .height = 2 };
    try testing.expectEqual(@as(usize, 40), d.effectiveRowPitch());
    const padded: TextureDesc = .{ .width = 10, .height = 2, .row_pitch = 64 };
    try testing.expectEqual(@as(usize, 64), padded.effectiveRowPitch());
}

test "each backend names its clip space" {
    try testing.expectEqual(math.Clip.gl.depth, Backend.gl.clip().depth);
    try testing.expectEqual(math.Clip.d3d.depth, Backend.d3d11.clip().depth);
    try testing.expect(!Backend.d3d11.clip().flip_y);
    // WebGL is OpenGL ES, and OpenGL ES keeps OpenGL's clip space.
    try testing.expectEqual(Backend.gl.clip(), Backend.webgl.clip());
}
