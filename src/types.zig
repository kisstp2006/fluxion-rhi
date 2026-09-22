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
    /// Direct3D 12, feature level 11.0. Windows only.
    d3d12,
    /// WebGL 2, in a browser, through `fluxion-webgl`. A `wasm32` build
    /// only - and, under test, any build, where it talks to that library's
    /// stub instead of a page.
    webgl,
    /// Vulkan 1.0, loaded at run time through `fluxion-vulkan`. Linux,
    /// Windows and Android - wherever an ICD is installed, which is not
    /// every machine with a GPU driver otherwise capable of it.
    vulkan,
    /// A backend the caller supplied to `Device.initWith`, none of the ones above.
    /// `Info.name` says which it is, and `Device.clip` has its clip space, which
    /// this function cannot know.
    other,

    pub fn clip(self: Backend) math.Clip {
        return switch (self) {
            .none => .gl,
            .gl => .gl,
            .d3d11 => .d3d,
            .d3d12 => .d3d,
            // OpenGL ES's, which is OpenGL's: depth from -1 to 1.
            .webgl => .gl,
            // Zero to one like Direct3D, and - unlike it - NDC Y points
            // down rather than up, which is `flip_y`. Unverified against a
            // real driver in this port: no Vulkan ICD was available to
            // check it on; see the port plan's open risks.
            .vulkan => .{ .depth = .zero_to_one, .flip_y = true },
            // Nothing to say about a backend this library has never seen; a
            // device knows its own, see `Device.clip`.
            .other => .gl,
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
    d3d12,
    webgl,
    vulkan,
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
///
/// What each one *is* - its size, its channels, whether it is depth or
/// compressed - is a row in `format_table`, and every question a format is
/// asked is answered from that row. A new format is a name here and a row
/// there, and a backend's answer to "can you do it" is a row in its own table
/// and in `Caps`, so nothing is added to a `switch`.
pub const Format = enum {
    r8_unorm,
    rg8_unorm,
    rgba8_unorm,
    rgba8_unorm_srgb,
    bgra8_unorm,
    bgra8_unorm_srgb,
    r16_float,
    rg16_float,
    rgba16_float,
    r32_float,
    rg32_float,
    rgba32_float,
    rgb10a2_unorm,
    rg11b10_float,

    depth16_unorm,
    depth24_stencil8,
    depth32_float,
    depth32_float_stencil8,

    // Block-compressed. BC is Direct3D's and the desktop's, ETC2 is OpenGL
    // ES's, ASTC is every phone's; none of them is on every device, which is
    // what `Caps` is for.
    bc1_rgba_unorm,
    bc1_rgba_unorm_srgb,
    bc3_rgba_unorm,
    bc3_rgba_unorm_srgb,
    bc4_r_unorm,
    bc5_rg_unorm,
    bc6h_rgb_ufloat,
    bc7_rgba_unorm,
    bc7_rgba_unorm_srgb,
    etc2_rgb8_unorm,
    etc2_rgb8_unorm_srgb,
    etc2_rgba8_unorm,
    etc2_rgba8_unorm_srgb,
    astc_4x4_unorm,
    astc_4x4_unorm_srgb,
    astc_6x6_unorm,
    astc_6x6_unorm_srgb,
    astc_8x8_unorm,
    astc_8x8_unorm_srgb,

    pub const count = @typeInfo(Format).@"enum".fields.len;

    /// The row of `format_table` this format is.
    pub fn info(self: Format) *const FormatInfo {
        return &format_table[@intFromEnum(self)];
    }

    /// Bytes in one texel. Only an uncompressed format has one: a compressed
    /// texel is a fraction of a byte, so ask `rowBytes` or `imageBytes`.
    pub fn bytesPerPixel(self: Format) usize {
        const row = self.info();
        std.debug.assert(row.block_width == 1 and row.block_height == 1);
        return row.block_bytes;
    }

    pub fn isDepth(self: Format) bool {
        return self.info().depth;
    }

    pub fn hasStencil(self: Format) bool {
        return self.info().stencil;
    }

    pub fn isCompressed(self: Format) bool {
        const row = self.info();
        return row.block_width > 1 or row.block_height > 1;
    }

    pub fn isSrgb(self: Format) bool {
        return self.info().srgb;
    }

    /// Bytes in one row of `width` texels, tightly packed. For a compressed
    /// format that is a row of blocks, and the last block counts whole.
    pub fn rowBytes(self: Format, width: u32) usize {
        const row = self.info();
        return @as(usize, std.math.divCeil(u32, width, row.block_width) catch unreachable) * row.block_bytes;
    }

    /// How many rows `height` texels take: rows of blocks for a compressed format.
    pub fn rowCount(self: Format, height: u32) u32 {
        return std.math.divCeil(u32, height, self.info().block_height) catch unreachable;
    }

    /// Bytes in a tightly packed image of `width` by `height` texels.
    pub fn imageBytes(self: Format, width: u32, height: u32) usize {
        return self.rowBytes(width) * self.rowCount(height);
    }
};

/// What a format is, as data.
pub const FormatInfo = struct {
    /// Bytes in one block: one texel, for a format that is not compressed.
    block_bytes: u8,
    /// Texels across and down one block. One and one unless compressed.
    block_width: u8 = 1,
    block_height: u8 = 1,
    channels: u8,
    kind: Kind = .unorm,
    /// Stored non-linearly, and converted on the way to and from a shader.
    srgb: bool = false,
    depth: bool = false,
    stencil: bool = false,

    pub const Kind = enum {
        /// Fixed point, read as 0 to 1.
        unorm,
        float,
    };
};

const FormatRow = struct { format: Format, info: FormatInfo };

/// One row per `Format`, in any order. A format without a row does not
/// compile, and neither does one with two.
const format_rows = [_]FormatRow{
    .{ .format = .r8_unorm, .info = .{ .block_bytes = 1, .channels = 1 } },
    .{ .format = .rg8_unorm, .info = .{ .block_bytes = 2, .channels = 2 } },
    .{ .format = .rgba8_unorm, .info = .{ .block_bytes = 4, .channels = 4 } },
    .{ .format = .rgba8_unorm_srgb, .info = .{ .block_bytes = 4, .channels = 4, .srgb = true } },
    .{ .format = .bgra8_unorm, .info = .{ .block_bytes = 4, .channels = 4 } },
    .{ .format = .bgra8_unorm_srgb, .info = .{ .block_bytes = 4, .channels = 4, .srgb = true } },
    .{ .format = .r16_float, .info = .{ .block_bytes = 2, .channels = 1, .kind = .float } },
    .{ .format = .rg16_float, .info = .{ .block_bytes = 4, .channels = 2, .kind = .float } },
    .{ .format = .rgba16_float, .info = .{ .block_bytes = 8, .channels = 4, .kind = .float } },
    .{ .format = .r32_float, .info = .{ .block_bytes = 4, .channels = 1, .kind = .float } },
    .{ .format = .rg32_float, .info = .{ .block_bytes = 8, .channels = 2, .kind = .float } },
    .{ .format = .rgba32_float, .info = .{ .block_bytes = 16, .channels = 4, .kind = .float } },
    .{ .format = .rgb10a2_unorm, .info = .{ .block_bytes = 4, .channels = 4 } },
    .{ .format = .rg11b10_float, .info = .{ .block_bytes = 4, .channels = 3, .kind = .float } },

    .{ .format = .depth16_unorm, .info = .{ .block_bytes = 2, .channels = 1, .depth = true } },
    .{ .format = .depth24_stencil8, .info = .{ .block_bytes = 4, .channels = 2, .depth = true, .stencil = true } },
    .{ .format = .depth32_float, .info = .{ .block_bytes = 4, .channels = 1, .kind = .float, .depth = true } },
    .{ .format = .depth32_float_stencil8, .info = .{ .block_bytes = 8, .channels = 2, .kind = .float, .depth = true, .stencil = true } },

    .{ .format = .bc1_rgba_unorm, .info = .{ .block_bytes = 8, .block_width = 4, .block_height = 4, .channels = 4 } },
    .{ .format = .bc1_rgba_unorm_srgb, .info = .{ .block_bytes = 8, .block_width = 4, .block_height = 4, .channels = 4, .srgb = true } },
    .{ .format = .bc3_rgba_unorm, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4 } },
    .{ .format = .bc3_rgba_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4, .srgb = true } },
    .{ .format = .bc4_r_unorm, .info = .{ .block_bytes = 8, .block_width = 4, .block_height = 4, .channels = 1 } },
    .{ .format = .bc5_rg_unorm, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 2 } },
    .{ .format = .bc6h_rgb_ufloat, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 3, .kind = .float } },
    .{ .format = .bc7_rgba_unorm, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4 } },
    .{ .format = .bc7_rgba_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4, .srgb = true } },
    .{ .format = .etc2_rgb8_unorm, .info = .{ .block_bytes = 8, .block_width = 4, .block_height = 4, .channels = 3 } },
    .{ .format = .etc2_rgb8_unorm_srgb, .info = .{ .block_bytes = 8, .block_width = 4, .block_height = 4, .channels = 3, .srgb = true } },
    .{ .format = .etc2_rgba8_unorm, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4 } },
    .{ .format = .etc2_rgba8_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4, .srgb = true } },
    .{ .format = .astc_4x4_unorm, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4 } },
    .{ .format = .astc_4x4_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 4, .block_height = 4, .channels = 4, .srgb = true } },
    .{ .format = .astc_6x6_unorm, .info = .{ .block_bytes = 16, .block_width = 6, .block_height = 6, .channels = 4 } },
    .{ .format = .astc_6x6_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 6, .block_height = 6, .channels = 4, .srgb = true } },
    .{ .format = .astc_8x8_unorm, .info = .{ .block_bytes = 16, .block_width = 8, .block_height = 8, .channels = 4 } },
    .{ .format = .astc_8x8_unorm_srgb, .info = .{ .block_bytes = 16, .block_width = 8, .block_height = 8, .channels = 4, .srgb = true } },
};

const format_table: [Format.count]FormatInfo = blk: {
    var table: [Format.count]FormatInfo = undefined;
    var seen: [Format.count]bool = @splat(false);
    for (format_rows) |row| {
        const at = @intFromEnum(row.format);
        if (seen[at]) @compileError("format_table: two rows for " ++ @tagName(row.format));
        seen[at] = true;
        table[at] = row.info;
    }
    for (seen, 0..) |present, at| {
        if (!present) @compileError("format_table: no row for " ++ @tagName(@as(Format, @enumFromInt(at))));
    }
    break :blk table;
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

/// What shape of texture it is, and so how a shader addresses it.
pub const Dimension = enum {
    /// A plain image: `width` by `height`.
    d2,
    /// Six square faces, in the order +X, -X, +Y, -Y, +Z, -Z. A shader
    /// samples it by direction. `width` and `height` are equal, and
    /// `depth_or_layers` is not looked at.
    cube,
    /// A volume: `depth_or_layers` slices of `width` by `height`.
    d3,
    /// `depth_or_layers` images of the same size, chosen by index in the shader.
    d2_array,
};

/// The size of one mip level along an axis: half, rounded down, never below one.
pub fn mipExtent(size: u32, level: u32) u32 {
    return @max(1, size >> @intCast(@min(level, 31)));
}

/// How many levels a full chain from the largest of these has, down to one texel.
pub fn fullMipCount(width: u32, height: u32, depth: u32) u32 {
    const largest = @max(1, @max(width, @max(height, depth)));
    return 1 + std.math.log2_int(u32, largest);
}

pub const TextureDesc = struct {
    dimension: Dimension = .d2,
    width: u32,
    height: u32,
    /// The depth of a `.d3`, or the number of layers of a `.d2_array`. One
    /// for a `.d2`, and not looked at for a `.cube`, which is always six.
    depth_or_layers: u32 = 1,
    /// How many mip levels are stored. One is no mipmaps; zero is the whole
    /// chain down to a single texel. Filled by `generateMips` or by writing
    /// each level with `Device.writeTexture`.
    mip_levels: u32 = 1,
    /// Samples per pixel. One is an ordinary texture; more is a multisampled
    /// render target, which is resolved into a single-sample texture at the
    /// end of a pass and cannot be sampled itself.
    samples: u32 = 1,
    format: Format = .rgba8_unorm,
    usage: TextureUsage = .{},
    /// Initial texels for mip level zero of the whole texture: every layer,
    /// face or slice, one after another, top row first. Copied; not kept.
    data: ?[]const u8 = null,
    /// Bytes from one row of `data` to the next. Zero means tightly packed.
    row_pitch: usize = 0,
    label: []const u8 = "",

    /// Images stacked in the texture: six for a cube, the layer count for an
    /// array, one otherwise.
    pub fn layers(self: TextureDesc) u32 {
        return switch (self.dimension) {
            .d2, .d3 => 1,
            .cube => 6,
            .d2_array => self.depth_or_layers,
        };
    }

    /// Slices of a volume: its depth for a `.d3`, one otherwise.
    pub fn depth(self: TextureDesc) u32 {
        return if (self.dimension == .d3) self.depth_or_layers else 1;
    }

    /// The number of mip levels this will have, with zero turned into a chain.
    pub fn mipCount(self: TextureDesc) u32 {
        return if (self.mip_levels == 0) fullMipCount(self.width, self.height, self.depth()) else self.mip_levels;
    }

    pub fn tightRowPitch(self: TextureDesc) usize {
        return self.format.rowBytes(self.width);
    }

    pub fn effectiveRowPitch(self: TextureDesc) usize {
        return if (self.row_pitch == 0) self.tightRowPitch() else self.row_pitch;
    }
};

/// What a texture turned out to be, once made: the description with the
/// zero of `mip_levels` counted.
pub const TextureInfo = struct {
    dimension: Dimension,
    width: u32,
    height: u32,
    /// The depth of a volume, the layers of an array, six for a cube, one
    /// for a plain image.
    depth_or_layers: u32,
    mip_levels: u32,
    samples: u32,
    format: Format,
    usage: TextureUsage,
};

/// One image of a texture: a mip level of a layer, a face or a slice.
pub const Subresource = struct {
    mip: u32 = 0,
    /// The array layer, cube face or volume slice.
    layer: u32 = 0,
};

/// A box of texels inside one mip level, for writing.
///
/// `z` and `depth` count slices of a volume, or layers of an array, or faces
/// of a cube - whichever the texture is. A zero size means "to the end of the
/// level" on that axis, so `.{}` is the whole of level zero.
pub const TextureRegion = struct {
    mip: u32 = 0,
    x: u32 = 0,
    y: u32 = 0,
    z: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    depth: u32 = 0,
};

pub const Filter = enum { nearest, linear };

/// How a sampler picks between mip levels.
pub const MipFilter = enum {
    /// Level zero only, whatever is stored below it.
    none,
    nearest,
    linear,
};

pub const Wrap = enum {
    repeat,
    clamp_to_edge,
    mirror,
    /// Beyond the edge is `SamplerDesc.border`. Needs `Features.sampler_border`.
    border,
};

pub const BorderColor = enum { transparent_black, opaque_black, opaque_white };

pub const SamplerDesc = struct {
    min_filter: Filter = .linear,
    mag_filter: Filter = .linear,
    mip_filter: MipFilter = .none,
    wrap_u: Wrap = .clamp_to_edge,
    wrap_v: Wrap = .clamp_to_edge,
    /// The third axis: a volume's depth, and nothing else.
    wrap_w: Wrap = .clamp_to_edge,
    /// Taps along the long axis of a stretched footprint. One is off; a
    /// larger number is clamped to `Limits.max_anisotropy`.
    max_anisotropy: u8 = 1,
    /// A shadow sampler: what a lookup returns is whether the value stored
    /// passes this comparison against the one the shader gave, filtered.
    /// For a depth texture. Null is an ordinary read.
    compare: ?CompareFn = null,
    border: BorderColor = .transparent_black,
    /// The range of levels that may be read, from the largest.
    lod_min: f32 = 0,
    lod_max: f32 = 1000,
    /// Added to the level the hardware chose. Needs `Features.sampler_lod_bias`.
    lod_bias: f32 = 0,

    /// What a pixel-art game wants.
    pub const nearest: SamplerDesc = .{ .min_filter = .nearest, .mag_filter = .nearest };
    pub const linear: SamplerDesc = .{};
    /// What a 3D scene wants for a texture with a chain: trilinear, repeating.
    pub const trilinear: SamplerDesc = .{ .mip_filter = .linear, .wrap_u = .repeat, .wrap_v = .repeat, .wrap_w = .repeat };
};

/// One stage's source, in one language.
pub const ShaderStages = struct {
    vertex: [:0]const u8,
    fragment: [:0]const u8,
};

/// One stage's binary, as the 32-bit words SPIR-V is made of. Words rather
/// than bytes because Vulkan wants them aligned to four, and a slice of `u32`
/// is that by construction.
pub const ShaderWords = struct {
    vertex: []const u32,
    fragment: []const u32,
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
/// | | GLSL 330 and GLSL ES 300 | HLSL 5.0 | SPIR-V for Vulkan 1.0 |
/// | --- | --- | --- | --- |
/// | Vertex attribute at location `n` | `layout(location = n) in` | semantic `ATTRn` | `Input`, `Location n` |
/// | Uniform buffer in slot `n` | block named in `PipelineDesc.uniform_blocks[n]`, `std140` | `register(bn)` | `DescriptorSet 0`, `Binding n`, `std140` |
/// | Texture in slot `n` | sampler named in `PipelineDesc.textures[n]` | `register(tn)` and `register(sn)` | `DescriptorSet 1`, `Binding n`, a combined image sampler |
/// | Fragment colour | `out vec4` at location 0 | `SV_TARGET` | `Output`, `Location 0` |
///
/// SPIR-V has real binding numbers, so it ignores `uniform_blocks` and
/// `textures`, as HLSL does. Its two descriptor sets are the two slot spaces
/// the RHI already has: `setUniformBuffer(slot, ...)` and `setTexture(slot, ...)`.
pub const ShaderDesc = struct {
    /// GLSL 3.30 core, for the OpenGL backend.
    glsl: ?ShaderStages = null,
    /// GLSL ES 3.00, for the WebGL backend: `#version 300 es` on the first
    /// line, and a precision for `float` in the fragment stage.
    glsl_es: ?ShaderStages = null,
    /// HLSL for shader model 5.0, for the Direct3D 11 backend. It is also
    /// what the Direct3D 12 backend compiles, for shader model 5.1 or 6.
    hlsl: ?ShaderStages = null,
    /// SPIR-V, one module per stage with an entry point named `main`, for the
    /// Vulkan backend. Vulkan 1.0's, so `Shader` capability and nothing later.
    spirv: ?ShaderWords = null,
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
    /// do not, so it is stated here. Null is a pipeline for a pass with no
    /// colour attachment: a shadow map, a depth prepass.
    color_format: ?Format = .rgba8_unorm,
    /// The format of each of `RenderPassDesc.extra_colors`, in order, for a
    /// pipeline meant to draw into a multi-attachment pass.
    extra_color_formats: []const Format = &.{},
    depth_format: ?Format = null,
    /// Samples per pixel of the attachments it draws into: the pass's, and
    /// so the one the pipeline is for.
    samples: u32 = 1,
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
    /// Which mip level of a texture target is drawn into. A surface has one.
    mip_level: u32 = 0,
    /// Which layer, cube face or volume slice of a texture target.
    layer: u32 = 0,
    load: LoadOp = .clear,
    clear_color: Color = .{ 0, 0, 0, 1 },
    /// Where a multisampled target is resolved to when the pass ends: a
    /// single-sample texture of the same format and size, or the surface.
    /// Always mip level zero, layer zero of a texture - a resolve into a
    /// deeper level is a pass into a single-sample target of that level.
    /// Null keeps the samples, which is right only for a pass that will
    /// `load` them again. Must be null when the target is not multisampled.
    /// Resolving into the surface needs the surface to be the same size.
    resolve: ?RenderTarget = null,
};

pub const DepthAttachment = struct {
    texture: Texture,
    mip_level: u32 = 0,
    layer: u32 = 0,
    load: LoadOp = .clear,
    clear_depth: f32 = 1,
    clear_stencil: u8 = 0,
};

pub const RenderPassDesc = struct {
    /// Null is a pass that writes depth and nothing else - a shadow map, a
    /// depth prepass - and then `depth` is required, and `extra_colors` is
    /// empty.
    color: ?ColorAttachment = null,
    /// Additional color attachments written by the same draws as `color`,
    /// for a shader with more than one output - the geometry pass of a
    /// deferred renderer, writing colour and normal in one draw over one
    /// depth attachment. Every entry, and `color` itself once there are
    /// any, must target a texture: multiple render targets into the
    /// surface isn't something any backend supports, since there is
    /// exactly one swapchain image.
    extra_colors: []const ColorAttachment = &.{},
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
    /// The native module or application instance that owns `native_window`.
    ///
    /// The Windows Vulkan backend needs this as the `HINSTANCE` in
    /// `VkWin32SurfaceCreateInfoKHR`; it requires both it and `native_window`
    /// (`HWND`). Direct3D 11, OpenGL and WebGL ignore it. Zero is accepted for
    /// those backends but is invalid when a Windows Vulkan surface is opened.
    native_instance: usize = 0,
    /// A `VkSurfaceKHR`, already made from this device's own `VkInstance`
    /// (see `Device.vulkanInstanceHandles`), as an integer. Every other
    /// backend ignores this, and the Vulkan backend ignores `native_window`/
    /// `native_instance` - it has no way to turn either into a surface
    /// itself, since telling an X11 handle from a Wayland one needs more
    /// than an integer. Once handed to `createSurface`, the surface belongs
    /// to the device: it is destroyed by `destroySurface`, not by the caller.
    vulkan_surface: u64 = 0,
    /// Zero means "the window's size".
    width: u32 = 0,
    height: u32 = 0,
    vsync: bool = true,
};

// -------------------------------------------------------------------------
// Capabilities
// -------------------------------------------------------------------------

/// What a device can do with one format. Asked of the device once, when it
/// opens, and kept as data: `Device` checks a request against it before any
/// backend is called, so the answer is the same on every backend and never a
/// guess from the backend's name.
pub const FormatSupport = struct {
    /// Can be bound to a shader and read.
    sampled: bool = false,
    /// ...with linear filtering. Some hardware reads a 32-bit float texture
    /// but will not blend or filter it. For a depth format this says whether a
    /// *comparison* sampler filters (PCF); a plain read of depth is never
    /// filtered, on any backend.
    filterable: bool = false,
    /// Can be drawn into as a colour or a depth attachment.
    render_target: bool = false,
    /// A draw into it may blend.
    blendable: bool = false,
    /// Whether `generateMips` works on a texture of this format.
    generate_mips: bool = false,
    /// The sample counts a render target of this format can have, a bit per
    /// power of two: bit 0 is one sample, bit 2 is four.
    sample_counts: u8 = 0,
    /// The shapes of texture this format can be made as. All of them unless a
    /// device says otherwise: Direct3D 11 has no volume of a depth format,
    /// and a compressed volume is nobody's; a WebGL 2 context that lacks the
    /// binding for volumes and arrays has neither, of any format.
    dimensions: std.EnumSet(Dimension) = .initFull(),

    pub fn supportsSamples(self: FormatSupport, samples: u32) bool {
        if (samples == 0 or !std.math.isPowerOfTwo(samples)) return false;
        const bit = std.math.log2_int(u32, samples);
        return bit < 8 and self.sample_counts & (@as(u8, 1) << @intCast(bit)) != 0;
    }
};

/// The numbers a device is bound by.
pub const Limits = struct {
    /// The largest `width` or `height` of a `.d2` or `.d2_array`.
    max_texture_2d: u32,
    /// The largest side of a `.d3`.
    max_texture_3d: u32,
    /// The largest side of a `.cube`.
    max_texture_cube: u32,
    /// The most layers in a `.d2_array`.
    max_texture_layers: u32,
    /// The most taps `SamplerDesc.max_anisotropy` can ask for; one is none.
    max_anisotropy: u32,
    /// How many colour attachments a pass can have, `color` and `extra_colors` together.
    max_color_attachments: u32,
};

/// Things a device may or may not do, that are not about one format.
pub const Features = packed struct {
    /// `Wrap.border`.
    sampler_border: bool = false,
    /// `SamplerDesc.lod_bias`.
    sampler_lod_bias: bool = false,
    /// A compressed texture may be any size. False where the API wants the
    /// base level to be whole blocks (Direct3D), which `createTexture` then
    /// refuses with `Unsupported`; the levels below it are always allowed to
    /// end in a partial block.
    compressed_partial_blocks: bool = false,
    /// A texture that was drawn into is stored with the bottom of the picture
    /// in its first row, as the API's framebuffer is - so sampling it with the
    /// same coordinates as an uploaded image shows it upside down. True on
    /// OpenGL and WebGL, false on Direct3D and Vulkan. `readTexture` is not
    /// affected: it always returns the picture the right way up. A renderer
    /// that samples what it drew (a shadow map, a post-processing chain) flips
    /// the coordinate when this is set.
    render_target_origin_bottom_left: bool = false,
};

pub const Caps = struct {
    limits: Limits,
    features: Features = .{},
    formats: std.EnumArray(Format, FormatSupport) = .initFill(.{}),

    pub fn formatSupport(self: *const Caps, format: Format) FormatSupport {
        return self.formats.get(format);
    }
};

/// What a device is, once open.
pub const Info = struct {
    backend: Backend,
    /// What the driver calls itself. Points into the device; valid while it
    /// lives.
    renderer: []const u8,
    /// What the backend is called: `gl`, `d3d11` and so on, or the name a caller
    /// gave a backend of its own. `Device` fills it in, a backend need not.
    name: []const u8 = "",

    pub fn format(self: Info, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{s}: {s}", .{ if (self.name.len != 0) self.name else @tagName(self.backend), self.renderer });
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

test "every format has a row that agrees with itself" {
    for (std.enums.values(Format)) |format| {
        const row = format.info();
        try testing.expect(row.block_bytes > 0);
        try testing.expect(row.channels >= 1 and row.channels <= 4);
        try testing.expect(row.block_width >= 1 and row.block_height >= 1);
        // A stencil is only ever next to a depth.
        if (row.stencil) try testing.expect(row.depth);
        // Depth is never compressed and never sRGB.
        if (row.depth) try testing.expect(!format.isCompressed() and !row.srgb);
        // A texel with a whole number of bytes is one byte or more.
        if (!format.isCompressed()) try testing.expectEqual(@as(usize, row.block_bytes), format.bytesPerPixel());
    }
}

test "a compressed format counts blocks, and the last one counts whole" {
    // BC1: 8 bytes for each 4x4 block.
    try testing.expect(Format.bc1_rgba_unorm.isCompressed());
    try testing.expectEqual(@as(usize, 8), Format.bc1_rgba_unorm.rowBytes(4));
    try testing.expectEqual(@as(usize, 16), Format.bc1_rgba_unorm.rowBytes(5));
    try testing.expectEqual(@as(u32, 2), Format.bc1_rgba_unorm.rowCount(5));
    try testing.expectEqual(@as(usize, 32), Format.bc1_rgba_unorm.imageBytes(8, 8));
    // A level smaller than a block still stores one.
    try testing.expectEqual(@as(usize, 8), Format.bc1_rgba_unorm.imageBytes(1, 1));
    // ASTC 6x6 is 16 bytes for 36 texels; ten across is two blocks.
    try testing.expectEqual(@as(usize, 32), Format.astc_6x6_unorm.rowBytes(10));
    // An uncompressed one is just width times texel.
    try testing.expectEqual(@as(usize, 40), Format.rgba8_unorm.rowBytes(10));
    try testing.expectEqual(@as(usize, 80), Format.rgba8_unorm.imageBytes(10, 2));
}

test "mip levels halve down to one texel" {
    try testing.expectEqual(@as(u32, 256), mipExtent(1024, 2));
    try testing.expectEqual(@as(u32, 1), mipExtent(3, 5));
    try testing.expectEqual(@as(u32, 1), fullMipCount(1, 1, 1));
    try testing.expectEqual(@as(u32, 11), fullMipCount(1024, 512, 1));
    try testing.expectEqual(@as(u32, 9), fullMipCount(300, 20, 1));
    // The volume counts too.
    try testing.expectEqual(@as(u32, 7), fullMipCount(4, 4, 64));

    const cube: TextureDesc = .{ .dimension = .cube, .width = 64, .height = 64, .mip_levels = 0 };
    try testing.expectEqual(@as(u32, 6), cube.layers());
    try testing.expectEqual(@as(u32, 7), cube.mipCount());
    const volume: TextureDesc = .{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 32 };
    try testing.expectEqual(@as(u32, 32), volume.depth());
    try testing.expectEqual(@as(u32, 1), volume.layers());
}

test "a format reports the sample counts it was given" {
    const four: FormatSupport = .{ .sample_counts = 0b101 }; // 1 and 4
    try testing.expect(four.supportsSamples(1));
    try testing.expect(four.supportsSamples(4));
    try testing.expect(!four.supportsSamples(2));
    try testing.expect(!four.supportsSamples(3));
    try testing.expect(!four.supportsSamples(0));
    try testing.expect(!four.supportsSamples(256));
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
    // Direct3D 12 shares Direct3D 11's clip space.
    try testing.expectEqual(Backend.d3d11.clip(), Backend.d3d12.clip());
}

test "a surface description keeps the optional native instance separate" {
    const plain: SurfaceDesc = .{};
    try testing.expectEqual(@as(usize, 0), plain.native_window);
    try testing.expectEqual(@as(usize, 0), plain.native_instance);
    const win32: SurfaceDesc = .{ .native_window = 1, .native_instance = 2 };
    try testing.expectEqual(@as(usize, 1), win32.native_window);
    try testing.expectEqual(@as(usize, 2), win32.native_instance);
}
