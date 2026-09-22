// SPDX-License-Identifier: BSD-2-Clause

//! The OpenGL 3.3 core and Android OpenGL ES 3.0 backend.
//!
//! The context is not made here and never will be: it comes through
//! `GlHooks` from whoever opened the window, and this file starts where
//! `getProcAddress` does, with a `fluxion-gl` table filled from it.
//!
//! **Three things OpenGL does differently, absorbed here.**
//!
//! The origin is the bottom left, so every viewport and scissor rectangle is
//! flipped against the height of the level being drawn into, and what a pass
//! drew is read back with its rows reversed - a program sees the top-left
//! world the other backends have.
//!
//! A vertex array object captures buffer bindings, so there is one per
//! pipeline and the attribute pointers are re-specified whenever a vertex
//! buffer binding changes, at the draw that uses it. That costs a handful of
//! calls per draw that changes buffers and nothing otherwise.
//!
//! OpenGL 3.3 cannot instance and offset by a base vertex in the same draw,
//! and OpenGL ES 3.0 has no base-vertex draw. Unsupported combinations are
//! refused with `error.Unsupported`.
//!
//! **Uniform blocks and samplers are bound by name once**, when the pipeline
//! is made, from `PipelineDesc.uniform_blocks` and `PipelineDesc.textures`.
//! GLSL 330 has no `layout(binding = n)`; this is the seam that lets the same
//! `setUniformBuffer(slot, ...)` mean the same thing on Direct3D.
//!
//! **What a format is here is a row of `natives`**, and what a device can do
//! with it is worked out at `open` from that row and from asking the driver:
//! the version and the extension list for what the API brings, a tiny texture
//! and framebuffer for what can be drawn into, a tiny renderbuffer for which
//! sample counts it actually gives. Nothing is guessed from the backend's name,
//! and `caps` is what `Device` holds this file to.
//!
//! **Two textures are not what they look like.** A multisampled texture is a
//! renderbuffer - it is never sampled, and a renderbuffer is what every
//! version of both APIs can resolve. And there is no `glTexStorage` in 3.3,
//! so a texture is made by giving every level to `glTexImage*` in turn, and
//! a compressed one is given zeros where no data was.
//!
//! **BGRA is RGBA with red and blue swapped on the way in**, for the two
//! formats that have it, as on the WebGL backend: OpenGL ES has no BGRA
//! upload and no sRGB BGRA format anywhere, and one path is one less thing to
//! be wrong. A readback is RGBA whatever the texture was, which is the
//! contract, so nothing swaps on the way out.
//!
//! **sRGB is encoded when drawn into**, as it is on Direct3D: a pass into a
//! texture turns `GL_FRAMEBUFFER_SRGB` on, and a pass into the surface turns it
//! off, because the window's framebuffer takes what it is given. OpenGL ES
//! converts every sRGB attachment and has no switch.
//!
//! **Which way up an image is stored depends on how it got its contents.**
//! OpenGL stores what a pass drew bottom row first, and what was written from
//! memory as it was given. `readTexture` returns the picture, top row first,
//! either way, so it turns over exactly the images a pass has drawn into, and
//! `TextureRes.drawn` has a bit for each of them - each level of each layer,
//! face or slice, exactly, so that nothing goes untracked. A pass marks its
//! colour attachments, and what it resolves into, whether or not it draws
//! anything. A write clears the bit of every image it touches, whole images
//! and not boxes: a box written into a drawn image makes all of it a written
//! one, and what was drawn round the box then reads back upside down. And
//! `generateMips` gives each level the bit of the first level of its layer,
//! which is the way up it is filtered from. An image that is not one - which
//! `Device` never asks for - reads as drawn, which is how every readback
//! was made before the two were told apart.
//!
//! What a shader sees when it samples a drawn image is that same bottom-first
//! order, and no readback hides it: `Features.render_target_origin_bottom_left`
//! says so, and a renderer that samples what it drew flips the coordinate.

const std = @import("std");
const builtin = @import("builtin");
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
const is_gles = builtin.abi.isAndroid();
const Api = if (is_gles) opengl.Gles else opengl.Gl;

/// Which of the two APIs this build talks to, as a value the tables can be
/// asked about - so that what each needs is data in one place and both are
/// checked by the same test, whichever is compiled in.
const ApiKind = enum { gl, es };
const api_kind: ApiKind = if (is_gles) .es else .gl;

/// One more than any deferred pass here writes at once, with room to grow, and
/// also as many as any of the three backends can offer: Direct3D 11 has eight
/// render targets. The limit `caps` reports is the driver's, held to this,
/// because a framebuffer's key and a pass's arrays have to have a size.
const most_color_attachments = 8;
const max_vertex_slots = 8;
/// How many framebuffers are kept, by what is attached to them. A frame has a
/// handful of passes; when there are more than this the one used longest ago
/// is thrown away and made again if it is asked for.
const framebuffer_cache_len = 16;
/// The size of the textures and renderbuffers a format is tried with. Eight
/// is the smallest side every block of every compressed format divides into
/// whole, and it is nothing to a driver.
const probe_size = 8;
/// A compressed image of `probe_size` squared, for the largest block this
/// table has. Checked where the table is built.
const probe_bytes = 64;
/// A bit per power of two in `FormatSupport.sample_counts` is eight bits wide.
const sample_count_bits = 8;

/// Tokens `fluxion-gl`'s `enums` does not carry yet.
const token = struct {
    const texture_lod_bias: gt.Enum = 0x8501;
    const max_texture_lod_bias: gt.Enum = 0x84FD;

    const compressed_rgba_s3tc_dxt1: gt.Enum = 0x83F1;
    const compressed_rgba_s3tc_dxt5: gt.Enum = 0x83F3;
    const compressed_srgb_alpha_s3tc_dxt1: gt.Enum = 0x8C4D;
    const compressed_srgb_alpha_s3tc_dxt5: gt.Enum = 0x8C4F;
    const compressed_red_rgtc1: gt.Enum = 0x8DBB;
    const compressed_rg_rgtc2: gt.Enum = 0x8DBD;
    const compressed_rgb_bptc_unsigned_float: gt.Enum = 0x8E8F;
    const compressed_rgba_bptc_unorm: gt.Enum = 0x8E8C;
    const compressed_srgb_alpha_bptc_unorm: gt.Enum = 0x8E8D;
    const compressed_rgb8_etc2: gt.Enum = 0x9274;
    const compressed_srgb8_etc2: gt.Enum = 0x9275;
    const compressed_rgba8_etc2_eac: gt.Enum = 0x9278;
    const compressed_srgb8_alpha8_etc2_eac: gt.Enum = 0x9279;
    const compressed_rgba_astc_4x4: gt.Enum = 0x93B0;
    const compressed_rgba_astc_6x6: gt.Enum = 0x93B4;
    const compressed_rgba_astc_8x8: gt.Enum = 0x93B7;
    const compressed_srgb8_alpha8_astc_4x4: gt.Enum = 0x93D0;
    const compressed_srgb8_alpha8_astc_6x6: gt.Enum = 0x93D4;
    const compressed_srgb8_alpha8_astc_8x8: gt.Enum = 0x93D7;
};

/// Commands `fluxion-gl`'s tables do not carry: the 3D compressed uploads
/// (in neither), and the vector form of a sampler parameter, which is how a
/// border colour is set. All optional, because a context that lacks one is
/// answered by `caps` and not by a crash. See `Resolver`.
const Extra = struct {
    compressedTexImage3D: ?*const fn (
        target: gt.Enum,
        level: gt.Int,
        internal_format: gt.Enum,
        width: gt.Sizei,
        height: gt.Sizei,
        depth: gt.Sizei,
        border: gt.Int,
        image_size: gt.Sizei,
        data: ?*const anyopaque,
    ) callconv(.c) void = null,
    compressedTexSubImage3D: ?*const fn (
        target: gt.Enum,
        level: gt.Int,
        xoffset: gt.Int,
        yoffset: gt.Int,
        zoffset: gt.Int,
        width: gt.Sizei,
        height: gt.Sizei,
        depth: gt.Sizei,
        format: gt.Enum,
        image_size: gt.Sizei,
        data: ?*const anyopaque,
    ) callconv(.c) void = null,
    samplerParameterfv: ?*const fn (sampler: gt.Uint, pname: gt.Enum, params: [*]const gt.Float) callconv(.c) void = null,

    const options: opengl.loader.Options = .{ .prefix = "gl" };
};

/// A command of this API's table, whether that table declares it optional
/// (OpenGL ES 3.0's commands are, because an ES 2.0 driver lacks them, and
/// `open` refuses that driver) or not. One spelling for both, instead of an
/// `if (is_gles)` at every call.
fn FnOf(comptime name: []const u8) type {
    const T = @FieldType(Api, name);
    return switch (@typeInfo(T)) {
        .optional => |optional| optional.child,
        else => T,
    };
}

fn fnOf(api: *const Api, comptime name: []const u8) FnOf(name) {
    const value = @field(api, name);
    return if (@typeInfo(@TypeOf(value)) == .optional) value.? else value;
}

// -------------------------------------------------------------------------
// What an API has to have
// -------------------------------------------------------------------------

/// What one API has to have before it will do something: be a version that
/// has it in its core, or have any one of some extensions.
const Need = struct {
    /// The version, as `major * 10 + minor`, from which it is core. Zero is
    /// "always": the lowest version this backend opens on has it. Null is
    /// "not in core".
    core_since: ?u16 = null,
    /// Any one of these brings it.
    extensions: []const []const u8 = &.{},

    const always: Need = .{ .core_since = 0 };
    const never: Need = .{};

    fn since(version: u16) Need {
        return .{ .core_since = version };
    }

    fn extension(names: []const []const u8) Need {
        return .{ .extensions = names };
    }

    fn sinceOrExtension(version: u16, names: []const []const u8) Need {
        return .{ .core_since = version, .extensions = names };
    }
};

/// A `Need` for each of the two APIs, which disagree about nearly all of it:
/// ETC2 is core on ES and a 4.3 feature on the desktop, and BC is the other
/// way round.
const ApiNeed = struct {
    gl: Need = .always,
    es: Need = .always,

    fn on(self: ApiNeed, kind: ApiKind) Need {
        return switch (kind) {
            .gl => self.gl,
            .es => self.es,
        };
    }
};

/// What a context is, for asking it a `Need`.
const Context = struct {
    version: opengl.Version,
    extensions: opengl.extensions.Set,

    fn satisfies(self: Context, need: Need) bool {
        if (need.core_since) |version| {
            if (self.version.atLeast(version / 10, version % 10)) return true;
        }
        for (need.extensions) |name| {
            if (self.extensions.has(name)) return true;
        }
        return false;
    }
};

const feature_need = struct {
    /// Both APIs size a compressed image by its own width and height, and
    /// store the last block of a row whole: a base level that is not whole
    /// blocks is an ordinary one. Direct3D is the one that objects.
    const compressed_partial_blocks: ApiNeed = .{};
    /// Core on the desktop with sampler objects; on ES 3.2 or with the
    /// extension, and then only through `glSamplerParameterfv`.
    const sampler_border: ApiNeed = .{
        .es = .sinceOrExtension(32, &.{ "EXT_texture_border_clamp", "OES_texture_border_clamp", "NV_texture_border_clamp" }),
    };
    /// A sampler's bias is a desktop thing: ES 3.0 has none, and ES 3.2 only
    /// as a shader-side argument.
    const sampler_lod_bias: ApiNeed = .{ .es = .never };
    /// Core in 4.6; an extension of every driver before that, and on ES.
    const anisotropy: ApiNeed = .{
        .gl = .sinceOrExtension(46, &.{ "ARB_texture_filter_anisotropic", "EXT_texture_filter_anisotropic" }),
        .es = .extension(&.{"EXT_texture_filter_anisotropic"}),
    };
};

// -------------------------------------------------------------------------
// What a format is
// -------------------------------------------------------------------------

/// One `types.Format` as OpenGL knows it.
const Native = struct {
    /// The sized internal format; for a compressed format, its own token.
    internal: gt.Enum,
    /// How texels travel: the `format` and `type` of `glTexImage2D` and
    /// `glReadPixels`. Zero for a compressed format, which has no such pair.
    format: gt.Enum = 0,
    kind: gt.Enum = 0,
    /// Stored as RGBA, with red and blue swapped on the way in. See the header.
    swap_rb: bool = false,
    /// What the API needs to make a texture of it at all.
    need: ApiNeed = .{},
    /// ...to filter it linearly.
    filter: ApiNeed = .{},
    /// ...to blend into it.
    blend: ApiNeed = .{},
};

const Row = struct { format: types.Format, native: ?Native };

const s3tc_extensions: []const []const u8 = &.{"EXT_texture_compression_s3tc"};
const dxt1_extensions: []const []const u8 = &.{ "EXT_texture_compression_s3tc", "EXT_texture_compression_dxt1" };
const s3tc_srgb_extensions: []const []const u8 = &.{ "EXT_texture_sRGB", "EXT_texture_compression_s3tc_srgb" };
const rgtc_extensions: []const []const u8 = &.{ "EXT_texture_compression_rgtc", "ARB_texture_compression_rgtc" };
const bptc_extensions: []const []const u8 = &.{ "ARB_texture_compression_bptc", "EXT_texture_compression_bptc" };
const etc2_extensions: []const []const u8 = &.{"ARB_ES3_compatibility"};
const astc_extensions: []const []const u8 = &.{"KHR_texture_compression_astc_ldr"};

const s3tc: ApiNeed = .{ .gl = .extension(s3tc_extensions), .es = .extension(s3tc_extensions) };
const dxt1: ApiNeed = .{ .gl = .extension(dxt1_extensions), .es = .extension(dxt1_extensions) };
const s3tc_srgb: ApiNeed = .{ .gl = .extension(s3tc_srgb_extensions), .es = .extension(s3tc_srgb_extensions) };
const rgtc: ApiNeed = .{ .gl = .since(30), .es = .extension(rgtc_extensions) };
const bptc: ApiNeed = .{ .gl = .sinceOrExtension(42, bptc_extensions), .es = .extension(bptc_extensions) };
const etc2: ApiNeed = .{ .gl = .sinceOrExtension(43, etc2_extensions), .es = .always };
const astc: ApiNeed = .{ .gl = .extension(astc_extensions), .es = .sinceOrExtension(32, astc_extensions) };

/// A 32-bit float is filtered only where the API says so: everywhere on the
/// desktop, and with an extension on ES.
const float_filter: ApiNeed = .{ .es = .extension(&.{"OES_texture_float_linear"}) };
/// ...and blended into the same way.
const float_blend: ApiNeed = .{ .es = .extension(&.{"EXT_float_blend"}) };
/// A depth texture is filtered without a comparison on the desktop; on ES it
/// is incomplete unless the sampler compares or the filter is nearest, so
/// "filterable" is not something to promise there.
const depth_filter: ApiNeed = .{ .es = .never };

/// One row per `Format`, in any order. A format without a row does not
/// compile, and neither does one with two; a `null` is a format this API
/// cannot do at all, and `caps` says so.
const format_rows = [_]Row{
    .{ .format = .r8_unorm, .native = .{ .internal = c.r8, .format = c.red, .kind = c.unsigned_byte } },
    .{ .format = .rg8_unorm, .native = .{ .internal = c.rg8, .format = c.rg, .kind = c.unsigned_byte } },
    .{ .format = .rgba8_unorm, .native = .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte } },
    .{ .format = .rgba8_unorm_srgb, .native = .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte } },
    .{ .format = .bgra8_unorm, .native = .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte, .swap_rb = true } },
    .{ .format = .bgra8_unorm_srgb, .native = .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte, .swap_rb = true } },
    .{ .format = .r16_float, .native = .{ .internal = c.r16f, .format = c.red, .kind = c.half_float } },
    .{ .format = .rg16_float, .native = .{ .internal = c.rg16f, .format = c.rg, .kind = c.half_float } },
    .{ .format = .rgba16_float, .native = .{ .internal = c.rgba16f, .format = c.rgba, .kind = c.half_float } },
    .{ .format = .r32_float, .native = .{ .internal = c.r32f, .format = c.red, .kind = c.float, .filter = float_filter, .blend = float_blend } },
    .{ .format = .rg32_float, .native = .{ .internal = c.rg32f, .format = c.rg, .kind = c.float, .filter = float_filter, .blend = float_blend } },
    .{ .format = .rgba32_float, .native = .{ .internal = c.rgba32f, .format = c.rgba, .kind = c.float, .filter = float_filter, .blend = float_blend } },
    .{ .format = .rgb10a2_unorm, .native = .{ .internal = c.rgb10_a2, .format = c.rgba, .kind = c.unsigned_int_2_10_10_10_rev } },
    .{ .format = .rg11b10_float, .native = .{ .internal = c.r11f_g11f_b10f, .format = c.rgb, .kind = c.unsigned_int_10f_11f_11f_rev } },

    .{ .format = .depth16_unorm, .native = .{ .internal = c.depth_component16, .format = c.depth_component, .kind = c.unsigned_short, .filter = depth_filter } },
    .{ .format = .depth24_stencil8, .native = .{ .internal = c.depth24_stencil8, .format = c.depth_stencil, .kind = c.unsigned_int_24_8, .filter = depth_filter } },
    .{ .format = .depth32_float, .native = .{ .internal = c.depth_component32f, .format = c.depth_component, .kind = c.float, .filter = depth_filter } },
    .{ .format = .depth32_float_stencil8, .native = .{ .internal = c.depth32f_stencil8, .format = c.depth_stencil, .kind = c.float_32_unsigned_int_24_8_rev, .filter = depth_filter } },

    .{ .format = .bc1_rgba_unorm, .native = .{ .internal = token.compressed_rgba_s3tc_dxt1, .need = dxt1 } },
    .{ .format = .bc1_rgba_unorm_srgb, .native = .{ .internal = token.compressed_srgb_alpha_s3tc_dxt1, .need = s3tc_srgb } },
    .{ .format = .bc3_rgba_unorm, .native = .{ .internal = token.compressed_rgba_s3tc_dxt5, .need = s3tc } },
    .{ .format = .bc3_rgba_unorm_srgb, .native = .{ .internal = token.compressed_srgb_alpha_s3tc_dxt5, .need = s3tc_srgb } },
    .{ .format = .bc4_r_unorm, .native = .{ .internal = token.compressed_red_rgtc1, .need = rgtc } },
    .{ .format = .bc5_rg_unorm, .native = .{ .internal = token.compressed_rg_rgtc2, .need = rgtc } },
    .{ .format = .bc6h_rgb_ufloat, .native = .{ .internal = token.compressed_rgb_bptc_unsigned_float, .need = bptc } },
    .{ .format = .bc7_rgba_unorm, .native = .{ .internal = token.compressed_rgba_bptc_unorm, .need = bptc } },
    .{ .format = .bc7_rgba_unorm_srgb, .native = .{ .internal = token.compressed_srgb_alpha_bptc_unorm, .need = bptc } },
    .{ .format = .etc2_rgb8_unorm, .native = .{ .internal = token.compressed_rgb8_etc2, .need = etc2 } },
    .{ .format = .etc2_rgb8_unorm_srgb, .native = .{ .internal = token.compressed_srgb8_etc2, .need = etc2 } },
    .{ .format = .etc2_rgba8_unorm, .native = .{ .internal = token.compressed_rgba8_etc2_eac, .need = etc2 } },
    .{ .format = .etc2_rgba8_unorm_srgb, .native = .{ .internal = token.compressed_srgb8_alpha8_etc2_eac, .need = etc2 } },
    .{ .format = .astc_4x4_unorm, .native = .{ .internal = token.compressed_rgba_astc_4x4, .need = astc } },
    .{ .format = .astc_4x4_unorm_srgb, .native = .{ .internal = token.compressed_srgb8_alpha8_astc_4x4, .need = astc } },
    .{ .format = .astc_6x6_unorm, .native = .{ .internal = token.compressed_rgba_astc_6x6, .need = astc } },
    .{ .format = .astc_6x6_unorm_srgb, .native = .{ .internal = token.compressed_srgb8_alpha8_astc_6x6, .need = astc } },
    .{ .format = .astc_8x8_unorm, .native = .{ .internal = token.compressed_rgba_astc_8x8, .need = astc } },
    .{ .format = .astc_8x8_unorm_srgb, .native = .{ .internal = token.compressed_srgb8_alpha8_astc_8x8, .need = astc } },
};

const natives: std.EnumArray(types.Format, ?Native) = blk: {
    var table: std.EnumArray(types.Format, ?Native) = .initUndefined();
    var seen: [types.Format.count]bool = @splat(false);
    for (format_rows) |row| {
        const at = @intFromEnum(row.format);
        if (seen[at]) @compileError("gl natives: two rows for " ++ @tagName(row.format));
        seen[at] = true;
        table.set(row.format, row.native);
        if (row.native != null and row.format.isCompressed() and row.format.imageBytes(probe_size, probe_size) > probe_bytes) {
            @compileError("gl natives: probe_bytes is too small to try " ++ @tagName(row.format));
        }
    }
    for (seen, 0..) |has_row, at| {
        if (!has_row) @compileError("gl natives: no row for " ++ @tagName(@as(types.Format, @enumFromInt(at))));
    }
    break :blk table;
};

fn nativeOf(format: types.Format) ?*const Native {
    if (natives.values[@intFromEnum(format)]) |*row| return row;
    return null;
}

/// The attachment point a format is drawn into.
fn attachmentPoint(format: types.Format) gt.Enum {
    if (!format.isDepth()) return c.color_attachment0;
    return if (format.hasStencil()) c.depth_stencil_attachment else c.depth_attachment;
}

/// The binding point of each shape of texture, and so what `glBindTexture`
/// and `glTexImage*` are given for it.
const texture_targets: std.EnumArray(types.Dimension, gt.Enum) = .init(.{
    .d2 = c.texture_2d,
    .cube = c.texture_cube_map,
    .d3 = c.texture_3d,
    .d2_array = c.texture_2d_array,
});

// -------------------------------------------------------------------------
// What a device can do
// -------------------------------------------------------------------------

/// What the driver said about one format when it was asked.
const Probe = struct {
    /// It made a texture of that format.
    accepted: bool = false,
    /// A framebuffer with one attached was complete.
    renderable: bool = false,
    /// The sample counts a renderbuffer of it really had, a bit per power of two.
    sample_counts: u8 = 0,
    /// The shapes of texture it was really made as. Full until a probe says
    /// otherwise, so that a test of the rules has not to say it.
    dimensions: std.EnumSet(types.Dimension) = .initFull(),
};

/// What `caps` says about one format: what the API needs from the row, what
/// the driver said when tried, and what follows from the two. Pure, so that
/// both APIs' rules are checked whichever one is compiled in.
fn formatSupport(format: types.Format, native: ?Native, kind: ApiKind, ctx: Context, probe: Probe) types.FormatSupport {
    // What is not there comes in no shape at all, which is not what a default says.
    const nothing: types.FormatSupport = .{ .dimensions = .initEmpty() };
    const row = native orelse return nothing;
    if (!ctx.satisfies(row.need.on(kind)) or !probe.accepted) return nothing;

    var support: types.FormatSupport = .{
        .sampled = true,
        .filterable = ctx.satisfies(row.filter.on(kind)),
        .sample_counts = 0b1,
        .dimensions = probe.dimensions,
    };
    if (format.isCompressed()) {
        // A compressed volume is nobody's: ES 3.0 refuses ETC2 in one outright,
        // and for the others it is an extension of an extension. Said here
        // and not left to a probe, so it is the same on every driver.
        support.dimensions.remove(.d3);
        return support;
    }

    support.render_target = probe.renderable;
    if (!probe.renderable) return support;
    support.sample_counts |= probe.sample_counts;
    if (format.isDepth()) return support;

    support.blendable = ctx.satisfies(row.blend.on(kind));
    // OpenGL ES will make levels of a format only if it can be both drawn
    // into and filtered, and the desktop makes them of every such format
    // there is here.
    support.generate_mips = support.filterable;
    return support;
}

/// What `Features` says: what the API brings, and - for a border colour,
/// which is a vector - that the command to set it was there to be loaded.
fn featuresOf(kind: ApiKind, ctx: Context, has_sampler_vector: bool) types.Features {
    return .{
        .sampler_border = ctx.satisfies(feature_need.sampler_border.on(kind)) and has_sampler_vector,
        .sampler_lod_bias = ctx.satisfies(feature_need.sampler_lod_bias.on(kind)),
        .compressed_partial_blocks = ctx.satisfies(feature_need.compressed_partial_blocks.on(kind)),
        // The framebuffer is bottom-up in both APIs, and there is nothing to ask.
        .render_target_origin_bottom_left = true,
    };
}

/// Whether there is anything to ask for `Limits.max_anisotropy`: without the
/// extension the query is an error, and the answer is one tap.
fn hasAnisotropy(kind: ApiKind, ctx: Context) bool {
    return ctx.satisfies(feature_need.anisotropy.on(kind));
}

fn integer(api: *const Api, pname: gt.Enum) gt.Int {
    var value: [1]gt.Int = .{0};
    api.getIntegerv(pname, &value);
    return value[0];
}

fn positive(api: *const Api, pname: gt.Enum) u32 {
    return @intCast(@max(0, integer(api, pname)));
}

fn real(api: *const Api, pname: gt.Enum) f32 {
    var value: [1]gt.Float = .{0};
    api.getFloatv(pname, &value);
    return value[0];
}

/// The extension names of the context: `glGetString(GL_EXTENSIONS)` is null
/// on a core profile, so they come one at a time. The names point into the
/// driver's own strings, good as long as the context.
const ExtensionNames = struct {
    storage: [][]const u8,
    found: usize,

    fn set(self: ExtensionNames) opengl.extensions.Set {
        return .{ .names = self.storage[0..self.found] };
    }
};

fn gatherExtensions(gpa: Allocator, api: *const Api) Allocator.Error!ExtensionNames {
    const total: usize = positive(api, c.num_extensions);
    const storage = try gpa.alloc([]const u8, total);
    const get = fnOf(api, "getStringi");
    var found: usize = 0;
    for (0..total) |i| {
        const name = get(c.extensions, @intCast(i)) orelse continue;
        storage[found] = std.mem.span(name);
        found += 1;
    }
    return .{ .storage = storage, .found = found };
}

/// Whether the driver refused something since it was last asked, clearing
/// the queue - and the debug callback's note of it, because the caller is
/// about to say so itself and a submit should not say it twice.
fn refused(self: *Gl) ?gt.Enum {
    const code = self.api.checkError() orelse return null;
    self.had_error = false;
    return code;
}

fn failure(code: gt.Enum) Error {
    return if (code == c.out_of_memory) error.OutOfMemory else error.Failed;
}

/// Make a texture of `format` and see if the driver will, then a
/// framebuffer with it in, then renderbuffers of each sample count. Leaves
/// nothing bound and nothing in the error queue.
fn probeFormat(self: *Gl, format: types.Format, native: *const Native) Probe {
    const api = &self.api;
    var probe: Probe = .{};
    _ = api.checkError();

    var texture: gt.Uint = 0;
    api.genTextures(1, @ptrCast(&texture));
    api.bindTexture(c.texture_2d, texture);
    if (format.isCompressed()) {
        const zeros: [probe_bytes]u8 = @splat(0);
        api.compressedTexImage2D(c.texture_2d, 0, native.internal, probe_size, probe_size, 0, @intCast(format.imageBytes(probe_size, probe_size)), &zeros);
    } else {
        api.texImage2D(c.texture_2d, 0, @intCast(native.internal), probe_size, probe_size, 0, native.format, native.kind, null);
    }
    probe.accepted = api.checkError() == null;
    probe.dimensions = .initEmpty();
    if (probe.accepted) probe.dimensions.insert(.d2);

    if (probe.accepted and !format.isCompressed()) {
        var framebuffer: gt.Uint = 0;
        api.genFramebuffers(1, @ptrCast(&framebuffer));
        api.bindFramebuffer(c.framebuffer, framebuffer);
        api.framebufferTexture2D(c.framebuffer, attachmentPoint(format), c.texture_2d, texture, 0);
        setDrawBuffers(self, if (format.isDepth()) 0 else 1);
        probe.renderable = api.checkFramebufferStatus(c.framebuffer) == c.framebuffer_complete;
        api.bindFramebuffer(c.framebuffer, 0);
        api.deleteFramebuffers(1, @ptrCast(&framebuffer));
        _ = api.checkError();
    }
    api.bindTexture(c.texture_2d, 0);
    api.deleteTextures(1, @ptrCast(&texture));

    if (probe.renderable) probe.sample_counts = probeSamples(self, native);
    if (probe.accepted) {
        for ([_]types.Dimension{ .cube, .d3, .d2_array }) |shape| {
            if (probeShape(self, format, native, shape)) probe.dimensions.insert(shape);
        }
    }
    _ = api.checkError();
    return probe;
}

/// Whether the driver makes a texture of this format in a shape other than a
/// plain image, and takes a block of texels into it: a cube's six faces, a
/// stack of two layers, a volume of two slices. A depth volume is the usual
/// no. A compressed volume is not asked, being refused as a rule; a compressed
/// stack needs the 3D compressed commands, and a context that lacks one has none.
fn probeShape(self: *Gl, format: types.Format, native: *const Native, shape: types.Dimension) bool {
    const api = &self.api;
    const compressed = format.isCompressed();
    if (compressed and shape == .d3) return false;
    _ = api.checkError();

    const target = texture_targets.get(shape);
    var texture: gt.Uint = 0;
    api.genTextures(1, @ptrCast(&texture));
    api.bindTexture(target, texture);
    const layers: gt.Sizei = 2;
    const zeros: [probe_bytes * 2]u8 = @splat(0);
    const image_bytes: gt.Sizei = @intCast(format.imageBytes(probe_size, probe_size));
    const block = format.info();
    const block_bytes: gt.Sizei = @intCast(block.block_bytes);
    var supported = true;
    switch (shape) {
        .d2 => unreachable, // `probeFormat` has made one
        .cube => {
            for (0..6) |face| {
                const face_target = c.cubeFace(@intCast(face));
                if (compressed) {
                    api.compressedTexImage2D(face_target, 0, native.internal, probe_size, probe_size, 0, image_bytes, &zeros);
                } else {
                    api.texImage2D(face_target, 0, @intCast(native.internal), probe_size, probe_size, 0, native.format, native.kind, null);
                }
            }
            if (compressed) {
                api.compressedTexSubImage2D(c.cubeFace(5), 0, 0, 0, block.block_width, block.block_height, native.internal, block_bytes, &zeros);
            }
        },
        .d3, .d2_array => {
            if (compressed) {
                const image = self.extra.compressedTexImage3D;
                const sub_image = self.extra.compressedTexSubImage3D;
                if (image == null or sub_image == null) supported = false;
                if (supported) {
                    image.?(target, 0, native.internal, probe_size, probe_size, layers, 0, image_bytes * layers, &zeros);
                    // Into the last layer: an image that could be made and not written to is no use.
                    sub_image.?(target, 0, 0, 0, layers - 1, block.block_width, block.block_height, 1, native.internal, block_bytes, &zeros);
                }
            } else {
                fnOf(api, "texImage3D")(target, 0, @intCast(native.internal), probe_size, probe_size, layers, 0, native.format, native.kind, null);
            }
        },
    }
    supported = supported and api.checkError() == null;
    _ = api.checkError();
    api.bindTexture(target, 0);
    api.deleteTextures(1, @ptrCast(&texture));
    return supported;
}

/// Which multisample counts a renderbuffer of this format gets. Asked by
/// making one: a driver may round a count up to one it has, and a count that
/// is only met by rounding is not one a texture and a depth buffer could
/// agree on, so it is not claimed.
fn probeSamples(self: *Gl, native: *const Native) u8 {
    const api = &self.api;
    var mask: u8 = 0;
    var renderbuffer: gt.Uint = 0;
    api.genRenderbuffers(1, @ptrCast(&renderbuffer));
    api.bindRenderbuffer(c.renderbuffer, renderbuffer);
    for (1..sample_count_bits) |bit| {
        const count: gt.Sizei = @as(gt.Sizei, 1) << @intCast(bit);
        if (count > self.max_samples) break;
        fnOf(api, "renderbufferStorageMultisample")(c.renderbuffer, count, native.internal, probe_size, probe_size);
        if (api.checkError() != null) continue;
        var got: [1]gt.Int = .{0};
        api.getRenderbufferParameteriv(c.renderbuffer, c.renderbuffer_samples, &got);
        if (got[0] == count) mask |= @as(u8, 1) << @intCast(bit);
    }
    api.bindRenderbuffer(c.renderbuffer, 0);
    api.deleteRenderbuffers(1, @ptrCast(&renderbuffer));
    return mask;
}

/// Everything `caps` answers, from asking the context.
fn buildCaps(self: *Gl, ctx: Context) types.Caps {
    const api = &self.api;

    self.max_renderbuffer = positive(api, c.max_renderbuffer_size);
    self.max_samples = integer(api, c.max_samples);
    const features = featuresOf(api_kind, ctx, self.extra.samplerParameterfv != null);
    if (hasAnisotropy(api_kind, ctx)) self.max_anisotropy = @max(1, real(api, c.max_texture_max_anisotropy));
    if (features.sampler_lod_bias) self.max_lod_bias = @max(0, real(api, token.max_texture_lod_bias));

    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = positive(api, c.max_texture_size),
            .max_texture_3d = positive(api, c.max_3d_texture_size),
            .max_texture_cube = positive(api, c.max_cube_map_texture_size),
            .max_texture_layers = positive(api, c.max_array_texture_layers),
            .max_anisotropy = @intFromFloat(@floor(self.max_anisotropy)),
            // A pass writes as many colours as the framebuffer has places for
            // them *and* the fragment stage has outputs, and the smaller of the two
            // is what can be drawn.
            .max_color_attachments = @min(positive(api, c.max_color_attachments), positive(api, c.max_draw_buffers), most_color_attachments),
        },
        .features = features,
    };

    for (std.enums.values(types.Format)) |format| {
        const native = nativeOf(format);
        var probe: Probe = .{};
        if (native) |row| {
            if (ctx.satisfies(row.need.on(api_kind))) probe = probeFormat(self, format, row);
        }
        answer.formats.set(format, formatSupport(format, if (native) |row| row.* else null, api_kind, ctx, probe));
    }
    return answer;
}

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const Gl = struct {
    gpa: Allocator,
    hooks: types.GlHooks,
    api: Api,
    extra: Extra = .{},
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

    /// Asked once, when the device opened: what `Device` holds this backend to.
    capabilities: types.Caps = undefined,
    /// The numbers behind `capabilities` that are not in it but that the code
    /// needs: how big a renderbuffer can be (a multisampled texture is one),
    /// the most samples asked of it, and where a sampler's bias and
    /// anisotropy stop.
    max_renderbuffer: u32 = 0,
    max_samples: gt.Int = 0,
    max_lod_bias: f32 = 0,
    max_anisotropy: f32 = 1,

    /// Counts up for every texture made, so that a framebuffer's key can name
    /// what is attached to it without a pointer that a later texture, at the
    /// same address, would inherit.
    next_serial: u64 = 1,
    framebuffers: [framebuffer_cache_len]FramebufferEntry = @splat(.{}),
    clock: u64 = 0,

    // Per-submit state. Reset at every pass.
    pipeline: ?*PipelineRes = null,
    /// The height of the level being drawn into, which is what a viewport
    /// and a scissor are flipped against.
    target_height: u32 = 0,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
    index: ?IndexBinding = null,
    /// The framebuffer of the pass in progress, and its size, for the resolve
    /// at `endPass`; and what each of its colours resolves into.
    pass_framebuffer: gt.Uint = 0,
    pass_size: [2]u32 = .{ 0, 0 },
    resolves: [most_color_attachments]?Resolve = @splat(null),
};

const VertexBinding = struct {
    name: gt.Uint = 0,
    offset: u32 = 0,
};

const IndexBinding = struct {
    name: gt.Uint,
    kind: gt.Enum,
    size: u32,
};

/// Where a multisampled colour goes when its pass ends. Null is the surface.
const Resolve = struct {
    target: ?*TextureRes,
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
    /// A texture - or a renderbuffer, when it is multisampled.
    name: gt.Uint,
    serial: u64,
    /// What `glBindTexture` is given; the renderbuffer token for a renderbuffer.
    target: gt.Enum,
    dimension: types.Dimension,
    width: u32,
    height: u32,
    /// Slices of a volume, layers of an array, six for a cube, one otherwise:
    /// how many `z` there are at level zero.
    depth_or_layers: u32,
    mip_levels: u32,
    samples: u32,
    format: types.Format,
    native: *const Native,
    /// The images a pass has drawn into since they were last written, a bit
    /// for each: see `imageIndex`. OpenGL keeps a drawn image bottom row first
    /// and a written one as it was given, so this says which a readback has to
    /// turn over.
    drawn: std.bit_set.DynamicBitSetUnmanaged,

    fn slices(self: TextureRes, mip: u32) u32 {
        return slicesAt(self.dimension, self.depth_or_layers, mip);
    }
};

/// Layers, faces or slices at a mip level: a volume shrinks in depth with
/// everything else, and an array or a cube does not.
fn slicesAt(dimension: types.Dimension, depth_or_layers: u32, mip: u32) u32 {
    return if (dimension == .d3) types.mipExtent(depth_or_layers, mip) else depth_or_layers;
}

/// How many images a texture has, every layer, face and slice of every level:
/// what `TextureRes.drawn` has a bit for.
fn imageCount(dimension: types.Dimension, depth_or_layers: u32, mip_levels: u32) usize {
    var total: usize = 0;
    for (0..mip_levels) |mip| total += slicesAt(dimension, depth_or_layers, @intCast(mip));
    return total;
}

/// Where an image is in `TextureRes.drawn`: the levels one after another,
/// each with its own slices. Null for what is not an image of the texture,
/// which `Device` has already refused, and so is for the record only.
fn imageIndex(res: *const TextureRes, mip: u32, layer: u32) ?usize {
    if (mip >= res.mip_levels or layer >= res.slices(mip)) return null;
    var at: usize = layer;
    for (0..mip) |below| at += res.slices(@intCast(below));
    return if (at < res.drawn.bit_length) at else null;
}

fn markDrawn(res: *TextureRes, mip: u32, layer: u32) void {
    if (imageIndex(res, mip, layer)) |at| res.drawn.set(at);
}

/// Whether an image is stored the way a pass leaves it. What was not drawn
/// into is as it was written; and one that cannot be told is taken as drawn,
/// the way a readback has always treated a texture, because the image that a
/// program reads back is far oftener one it drew.
fn isDrawn(res: *const TextureRes, mip: u32, layer: u32) bool {
    const at = imageIndex(res, mip, layer) orelse return true;
    return res.drawn.isSet(at);
}

/// A write puts an image back to the order it was given in. The granularity is
/// the image: a box inside a drawn image makes the whole of it a written one,
/// and what was drawn in the rest of it then reads back the other way up.
fn markWritten(res: *TextureRes, region: types.TextureRegion) void {
    for (0..region.depth) |i| {
        if (imageIndex(res, region.mip, region.z + @as(u32, @intCast(i)))) |at| res.drawn.unset(at);
    }
}

/// `generateMips` filters level zero down, row pair by row pair, so a level
/// comes out the way up its first was: each layer's, and for a volume the
/// first slice that goes into a slice of a level.
fn inheritDrawn(res: *TextureRes) void {
    for (1..res.mip_levels) |level| {
        const mip: u32 = @intCast(level);
        for (0..res.slices(mip)) |layer| {
            const source: u32 = if (res.dimension == .d3) @min(@as(u32, @intCast(layer)) << @intCast(mip), res.slices(0) - 1) else @intCast(layer);
            const at = imageIndex(res, mip, @intCast(layer)) orelse continue;
            res.drawn.setValue(at, isDrawn(res, 0, source));
        }
    }
}

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
    /// A pipeline with no colour format is a depth-only one, and writes no colour.
    color_write: bool,
};

const SurfaceRes = struct {
    /// A surface is the default framebuffer; there is nothing to store but
    /// the fact that it exists.
    claimed: bool = false,
};

// -------------------------------------------------------------------------
// Opening
// -------------------------------------------------------------------------

/// The commands that only an ES 3.0 driver has, of the ones this file calls
/// through the ES table, where they are optional.
const es3_commands = [_][]const u8{
    "genVertexArrays",
    "deleteVertexArrays",
    "bindVertexArray",
    "drawArraysInstanced",
    "drawElementsInstanced",
    "vertexAttribDivisor",
    "vertexAttribIPointer",
    "bindBufferBase",
    "getUniformBlockIndex",
    "uniformBlockBinding",
    "genSamplers",
    "deleteSamplers",
    "bindSampler",
    "samplerParameteri",
    "samplerParameterf",
    "texImage3D",
    "texSubImage3D",
    "framebufferTextureLayer",
    "renderbufferStorageMultisample",
    "blitFramebuffer",
    "drawBuffers",
    "readBuffer",
    "clearBufferfv",
    "getStringi",
};

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!backend.Opened {
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

    self.api.load(Resolver{ .hooks = hooks }) catch return error.NoDevice;
    // Whatever of `Extra` the context has; the rest stays null.
    _ = opengl.loader.tryLoad(&self.extra, Resolver{ .hooks = hooks }, Extra.options);

    const version = self.api.version() catch return error.NoDevice;
    if (!version.atLeast(3, if (is_gles) 0 else 3)) return error.NoDevice;
    if (!hasEs3Functions(&self.api)) return error.NoDevice;

    if (self.api.string(c.renderer)) |name| {
        const n = @min(name.len, self.renderer.len);
        @memcpy(self.renderer[0..n], name[0..n]);
        self.renderer_len = n;
    }

    // State that never changes under this backend.
    self.api.pixelStorei(c.pack_alignment, 1);
    self.api.pixelStorei(c.unpack_alignment, 1);
    self.api.enable(c.scissor_test);
    if (!is_gles) {
        // Direct3D samples a cube across its edges and rasterises a
        // multisampled target at its own rate; OpenGL asks to be told. On ES
        // 3.0 both are simply so.
        self.api.enable(c.texture_cube_map_seamless);
        self.api.enable(c.multisample);
    }

    // Before the debug callback is installed: what the driver says about the
    // formats it turns down is an answer and not a mistake.
    const names = try gatherExtensions(gpa, &self.api);
    defer gpa.free(names.storage);
    self.capabilities = buildCaps(self, .{ .version = version, .extensions = names.set() });
    _ = self.api.checkError();

    if (!is_gles and desc.debug) {
        if (self.api.debugMessageCallback) |set| {
            self.api.enable(c.debug_output);
            self.api.enable(c.debug_output_synchronous);
            set(debugMessage, self);
        }
    }

    return .{ self, &vtable };
}

fn hasEs3Functions(api: *const Api) bool {
    inline for (es3_commands) |name| {
        if (@typeInfo(@FieldType(Api, name)) == .optional and @field(api, name) == null) return false;
    }
    return true;
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
    .caps = caps,
    .createBuffer = createBuffer,
    .destroyBuffer = destroyBuffer,
    .updateBuffer = updateBuffer,
    .createTexture = createTexture,
    .destroyTexture = destroyTexture,
    .writeTexture = writeTexture,
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
    for (&self.framebuffers) |*entry| forgetFramebuffer(self, entry);
    self.gpa.destroy(self);
}

fn caps(impl: backend.Impl) types.Caps {
    return cast(impl).capabilities;
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

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;
    const native = nativeOf(desc.format) orelse return error.Unsupported;

    // `Device` has checked all of this against `caps`, shapes included, and
    // none of it can be reached through it. It is here for a caller that
    // opens the backend by hand, and because a `caps` that is wrong should
    // not be a crash.
    if (!self.capabilities.formatSupport(desc.format).dimensions.contains(desc.dimension)) return error.Unsupported;
    if (desc.samples > 1 and (desc.width > self.max_renderbuffer or desc.height > self.max_renderbuffer)) return error.Unsupported;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const layers: u32 = switch (desc.dimension) {
        .d2 => 1,
        .cube => 6,
        .d3, .d2_array => desc.depth_or_layers,
    };
    var drawn = try std.bit_set.DynamicBitSetUnmanaged.initEmpty(self.gpa, imageCount(desc.dimension, layers, desc.mip_levels));
    errdefer drawn.deinit(self.gpa);

    // Emptied first, so that a mistake somebody else made is not blamed on this.
    _ = api.checkError();

    var name: gt.Uint = 0;
    if (desc.samples > 1) {
        api.genRenderbuffers(1, @ptrCast(&name));
        api.bindRenderbuffer(c.renderbuffer, name);
        fnOf(api, "renderbufferStorageMultisample")(c.renderbuffer, @intCast(desc.samples), native.internal, @intCast(desc.width), @intCast(desc.height));
        api.bindRenderbuffer(c.renderbuffer, 0);
    } else {
        api.genTextures(1, @ptrCast(&name));
    }

    res.* = .{
        .name = name,
        .serial = self.next_serial,
        .target = if (desc.samples > 1) c.renderbuffer else texture_targets.get(desc.dimension),
        .dimension = desc.dimension,
        .width = desc.width,
        .height = desc.height,
        .depth_or_layers = layers,
        .mip_levels = desc.mip_levels,
        .samples = desc.samples,
        .format = desc.format,
        .native = native,
        .drawn = drawn,
    };
    self.next_serial += 1;

    if (desc.samples == 1) {
        allocateLevels(self, res) catch |err| {
            api.deleteTextures(1, @ptrCast(&name));
            return err;
        };
        // The levels it has and no more, so that whatever a sampler asks for
        // the texture is complete; and one plain filter, because the default
        // wants a whole chain and samples black without one.
        api.bindTexture(res.target, name);
        api.texParameteri(res.target, c.texture_max_level, @intCast(res.mip_levels - 1));
        api.texParameteri(res.target, c.texture_min_filter, @intCast(c.linear));
        api.texParameteri(res.target, c.texture_mag_filter, @intCast(c.linear));
    }

    if (refused(self)) |code| {
        if (desc.samples > 1) api.deleteRenderbuffers(1, @ptrCast(&name)) else api.deleteTextures(1, @ptrCast(&name));
        return failure(code);
    }

    if (desc.data) |data| {
        const pitch = desc.effectiveRowPitch();
        const region: types.TextureRegion = .{ .width = desc.width, .height = desc.height, .depth = res.slices(0) };
        uploadRegion(self, res, region, data, pitch, pitch * desc.format.rowCount(desc.height)) catch |err| {
            api.deleteTextures(1, @ptrCast(&name));
            return err;
        };
    }
    return res;
}

/// Give every level of every face, layer and slice its size, in one go: there
/// is no `glTexStorage` in 3.3, so this is what "allocate the texture" means.
/// A compressed level is given zeros, which every driver takes, where a null
/// is one that some do not.
fn allocateLevels(self: *Gl, res: *TextureRes) Error!void {
    const api = &self.api;
    const native = res.native;
    const format = res.format;
    api.bindTexture(res.target, res.name);

    const zeros = try self.gpa.alloc(u8, if (format.isCompressed()) format.imageBytes(res.width, res.height) * res.slices(0) else 0);
    defer self.gpa.free(zeros);
    @memset(zeros, 0);

    for (0..res.mip_levels) |level| {
        const mip: u32 = @intCast(level);
        const width: gt.Sizei = @intCast(types.mipExtent(res.width, mip));
        const height: gt.Sizei = @intCast(types.mipExtent(res.height, mip));
        const depth: gt.Sizei = @intCast(res.slices(mip));
        const bytes: gt.Sizei = @intCast(format.imageBytes(@intCast(width), @intCast(height)));
        switch (res.dimension) {
            .d2, .cube => {
                const faces: u32 = if (res.dimension == .cube) 6 else 1;
                for (0..faces) |face| {
                    const target: gt.Enum = if (res.dimension == .cube) c.cubeFace(@intCast(face)) else res.target;
                    if (format.isCompressed()) {
                        api.compressedTexImage2D(target, @intCast(level), native.internal, width, height, 0, bytes, zeros.ptr);
                    } else {
                        api.texImage2D(target, @intCast(level), @intCast(native.internal), width, height, 0, native.format, native.kind, null);
                    }
                }
            },
            .d3, .d2_array => {
                if (format.isCompressed()) {
                    const compressed_image = (self.extra.compressedTexImage3D) orelse return error.Unsupported;
                    compressed_image(res.target, @intCast(level), native.internal, width, height, depth, 0, bytes * depth, zeros.ptr);
                } else {
                    fnOf(api, "texImage3D")(res.target, @intCast(level), @intCast(native.internal), width, height, depth, 0, native.format, native.kind, null);
                }
            },
        }
    }
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    for (&self.framebuffers) |*entry| {
        if (entry.name != 0 and entry.key.touches(res.serial)) forgetFramebuffer(self, entry);
    }
    if (res.samples > 1) self.api.deleteRenderbuffers(1, @ptrCast(&res.name)) else self.api.deleteTextures(1, @ptrCast(&res.name));
    res.drawn.deinit(self.gpa);
    self.gpa.destroy(res);
}

fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    try uploadRegion(self, res, region, bytes, row_pitch, slice_pitch);
    markWritten(res, region);
}

/// Red and blue changed places in every texel of `image_count` images of
/// `rows` rows, or the rows just gathered to be tight, or both: `out` is the
/// tight result.
fn gather(out: []u8, bytes: []const u8, row_bytes: usize, rows: usize, images: usize, row_pitch: usize, image_pitch: usize, swap_rb: bool) void {
    for (0..images) |image| {
        for (0..rows) |row| {
            const from = bytes[image * image_pitch + row * row_pitch ..][0..row_bytes];
            const to = out[(image * rows + row) * row_bytes ..][0..row_bytes];
            @memcpy(to, from);
            if (!swap_rb) continue;
            var texel: usize = 0;
            while (texel + 4 <= row_bytes) : (texel += 4) std.mem.swap(u8, &to[texel], &to[texel + 2]);
        }
    }
}

/// A box of texels into a level of a texture: the one path for creating a
/// texture with data and for writing to it.
///
/// OpenGL reads what it is given as a stack of tightly packed images, or
/// with a row length in texels; so bytes that do not lie that way - a pitch
/// that is not a whole number of texels, the rows of blocks of a compressed
/// format, the two formats stored with red and blue swapped - are gathered
/// into a copy that does. Faces of a cube, and slices whose pitch is not the
/// tight one, go a call at a time.
fn uploadRegion(self: *Gl, res: *TextureRes, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    const api = &self.api;
    const format = res.format;
    const compressed = format.isCompressed();
    const texel_bytes: usize = if (compressed) 1 else format.bytesPerPixel();
    const rows = format.rowCount(region.height);
    const row_bytes = format.rowBytes(region.width);

    var staged: []u8 = &.{};
    defer self.gpa.free(staged);
    var source = bytes;
    var pitch = row_pitch;
    var stride = slice_pitch;
    const ragged = pitch != row_bytes and (compressed or pitch % texel_bytes != 0);
    if (ragged or res.native.swap_rb) {
        staged = try self.gpa.alloc(u8, row_bytes * rows * region.depth);
        gather(staged, bytes, row_bytes, rows, region.depth, row_pitch, slice_pitch, res.native.swap_rb);
        source = staged;
        pitch = row_bytes;
        stride = row_bytes * rows;
    }

    api.bindTexture(res.target, res.name);
    api.pixelStorei(c.unpack_row_length, if (compressed or pitch == row_bytes) 0 else @intCast(pitch / texel_bytes));
    defer api.pixelStorei(c.unpack_row_length, 0);

    const one_call = region.depth == 1 or (res.dimension != .cube and stride == pitch * rows);
    if (one_call) return uploadSlices(self, res, region, region.z, region.depth, source, row_bytes * rows);
    for (0..region.depth) |i| {
        try uploadSlices(self, res, region, region.z + @as(u32, @intCast(i)), 1, source[i * stride ..], row_bytes * rows);
    }
}

/// `count` slices of the same box from `z` on, in one command: a face of a
/// cube is a 2D image of its own, everything else is one call.
fn uploadSlices(self: *Gl, res: *TextureRes, region: types.TextureRegion, z: u32, count: u32, bytes: []const u8, slice_bytes: usize) Error!void {
    const api = &self.api;
    const native = res.native;
    const level: gt.Int = @intCast(region.mip);
    const x: gt.Int = @intCast(region.x);
    const y: gt.Int = @intCast(region.y);
    const width: gt.Sizei = @intCast(region.width);
    const height: gt.Sizei = @intCast(region.height);
    const size: gt.Sizei = @intCast(slice_bytes * count);
    const compressed = res.format.isCompressed();

    switch (res.dimension) {
        .d2, .cube => {
            const target: gt.Enum = if (res.dimension == .cube) c.cubeFace(z) else res.target;
            if (compressed) {
                api.compressedTexSubImage2D(target, level, x, y, width, height, native.internal, size, bytes.ptr);
            } else {
                api.texSubImage2D(target, level, x, y, width, height, native.format, native.kind, bytes.ptr);
            }
        },
        .d3, .d2_array => {
            if (compressed) {
                const sub_image = self.extra.compressedTexSubImage3D orelse return error.Unsupported;
                sub_image(res.target, level, x, y, @intCast(z), width, height, @intCast(count), native.internal, size, bytes.ptr);
            } else {
                fnOf(api, "texSubImage3D")(res.target, level, x, y, @intCast(z), width, height, @intCast(count), native.format, native.kind, bytes.ptr);
            }
        },
    }
}

/// What is attached to a framebuffer, by texture and not by pointer.
const AttachId = struct {
    /// Zero is nothing: no texture has it.
    serial: u64 = 0,
    mip: u32 = 0,
    layer: u32 = 0,
};

/// One image of a texture, as a pass or a readback draws into it.
const Attachment = struct {
    res: *TextureRes,
    mip: u32,
    layer: u32,

    fn id(self: Attachment) AttachId {
        return .{ .serial = self.res.serial, .mip = self.mip, .layer = self.layer };
    }

    fn size(self: Attachment) [2]u32 {
        return .{ types.mipExtent(self.res.width, self.mip), types.mipExtent(self.res.height, self.mip) };
    }
};

const FramebufferKey = struct {
    colors: [most_color_attachments]AttachId = @splat(.{}),
    depth: AttachId = .{},

    fn touches(self: FramebufferKey, serial: u64) bool {
        if (self.depth.serial == serial) return true;
        for (self.colors) |color| {
            if (color.serial == serial) return true;
        }
        return false;
    }
};

const FramebufferEntry = struct {
    key: FramebufferKey = .{},
    name: gt.Uint = 0,
    used: u64 = 0,
};

fn forgetFramebuffer(self: *Gl, entry: *FramebufferEntry) void {
    if (entry.name != 0) self.api.deleteFramebuffers(1, @ptrCast(&entry.name));
    entry.* = .{};
}

/// Which of the colour attachments a framebuffer draws into: the first `count`,
/// or none, for a pass that writes depth alone. The draw buffers are part of the
/// framebuffer's state and stay, so this is said once, when it is made; a
/// framebuffer with a depth attachment and a draw buffer that names nothing
/// is not complete.
fn setDrawBuffers(self: *Gl, count: usize) void {
    var buffers: [most_color_attachments]gt.Enum = undefined;
    for (0..count) |i| buffers[i] = c.colorAttachment(@intCast(i));
    if (count == 0) buffers[0] = c.none;
    fnOf(&self.api, "drawBuffers")(@intCast(@max(count, 1)), &buffers);
    fnOf(&self.api, "readBuffer")(if (count == 0) c.none else c.color_attachment0);
}

fn attachImage(self: *Gl, point: gt.Enum, attachment: Attachment) void {
    const api = &self.api;
    const res = attachment.res;
    const mip: gt.Int = @intCast(attachment.mip);
    if (res.samples > 1) {
        api.framebufferRenderbuffer(c.framebuffer, point, c.renderbuffer, res.name);
        return;
    }
    switch (res.dimension) {
        .d2 => api.framebufferTexture2D(c.framebuffer, point, c.texture_2d, res.name, mip),
        // A cube's faces are 2D images of their own to a framebuffer, and only from
        // OpenGL 4.5 are they layers of it.
        .cube => api.framebufferTexture2D(c.framebuffer, point, c.cubeFace(attachment.layer), res.name, mip),
        .d3, .d2_array => fnOf(api, "framebufferTextureLayer")(c.framebuffer, point, res.name, mip, @intCast(attachment.layer)),
    }
}

/// The framebuffer with exactly these images attached, made the first time
/// and kept: a pass into the same images is a bind, and a level or a layer
/// is a different key from level zero. Left bound.
fn framebufferFor(self: *Gl, colors: []const Attachment, depth: ?Attachment) Error!gt.Uint {
    const api = &self.api;
    var key: FramebufferKey = .{};
    for (colors, 0..) |color, i| key.colors[i] = color.id();
    if (depth) |d| key.depth = d.id();

    self.clock += 1;
    var victim: *FramebufferEntry = &self.framebuffers[0];
    for (&self.framebuffers) |*entry| {
        if (entry.name != 0 and std.meta.eql(entry.key, key)) {
            entry.used = self.clock;
            return entry.name;
        }
        // The one to make room in: an empty place, or else the oldest.
        const rank = if (entry.name == 0) 0 else entry.used;
        const victim_rank = if (victim.name == 0) 0 else victim.used;
        if (rank < victim_rank) victim = entry;
    }
    forgetFramebuffer(self, victim);

    var name: gt.Uint = 0;
    api.genFramebuffers(1, @ptrCast(&name));
    api.bindFramebuffer(c.framebuffer, name);
    for (colors, 0..) |color, i| attachImage(self, c.colorAttachment(@intCast(i)), color);
    if (depth) |d| attachImage(self, attachmentPoint(d.res.format), d);
    setDrawBuffers(self, colors.len);
    if (api.checkFramebufferStatus(c.framebuffer) != c.framebuffer_complete) {
        api.bindFramebuffer(c.framebuffer, 0);
        api.deleteFramebuffers(1, @ptrCast(&name));
        return error.Failed;
    }
    victim.* = .{ .key = key, .name = name, .used = self.clock };
    return name;
}

fn unormFromFloat(value: f32) u8 {
    if (!(value > 0)) return 0;
    if (value >= 1) return 255;
    return @intFromFloat(@round(value * 255));
}

fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const api = &self.api;
    const format = res.format.info();

    const width = types.mipExtent(res.width, sub.mip);
    const height = types.mipExtent(res.height, sub.mip);
    const row = @as(usize, width) * 4;
    const pixels = try gpa.alloc(u8, row * height);
    errdefer gpa.free(pixels);

    api.bindFramebuffer(c.framebuffer, try framebufferFor(self, &.{.{ .res = res, .mip = sub.mip, .layer = sub.layer }}, null));
    defer api.bindFramebuffer(c.framebuffer, 0);

    if (format.kind == .float) {
        // A floating-point attachment is read as floating point: ES will not
        // give one as bytes, and the desktop would only clamp it the same way.
        const texels = try gpa.alloc(f32, pixels.len);
        defer gpa.free(texels);
        api.readPixels(0, 0, @intCast(width), @intCast(height), c.rgba, c.float, texels.ptr);
        for (pixels, texels) |*out, value| out.* = unormFromFloat(value);
    } else {
        api.readPixels(0, 0, @intCast(width), @intCast(height), c.rgba, c.unsigned_byte, pixels.ptr);
    }

    // One channel is a grey, as it is on Direct3D. What is missing after two or
    // three channels came back zero, and the alpha one.
    if (format.channels == 1) {
        var texel: usize = 0;
        while (texel < pixels.len) : (texel += 4) {
            pixels[texel + 1] = pixels[texel];
            pixels[texel + 2] = pixels[texel];
        }
    }

    // What a pass drew came back bottom row first, and the contract is the
    // picture's top row first. What was written from memory came back as it was
    // given, which is top row first already, and turning it over would hand a
    // program its own picture upside down.
    if (isDrawn(res, sub.mip, sub.layer)) flipRows(pixels, row);
    return pixels;
}

fn flipRows(pixels: []u8, row: usize) void {
    var top: usize = 0;
    var bottom: usize = pixels.len / row - 1;
    while (top < bottom) : ({
        top += 1;
        bottom -= 1;
    }) {
        const a = pixels[top * row ..][0..row];
        const b = pixels[bottom * row ..][0..row];
        for (a, b) |*x, *y| std.mem.swap(u8, x, y);
    }
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

const MipFilters = std.EnumArray(types.MipFilter, gt.Enum);

/// The minification filter of each pair of a filter and a mip filter.
const min_filters: std.EnumArray(types.Filter, MipFilters) = .init(.{
    .nearest = MipFilters.init(.{ .none = c.nearest, .nearest = c.nearest_mipmap_nearest, .linear = c.nearest_mipmap_linear }),
    .linear = MipFilters.init(.{ .none = c.linear, .nearest = c.linear_mipmap_nearest, .linear = c.linear_mipmap_linear }),
});

const wraps: std.EnumArray(types.Wrap, gt.Enum) = .init(.{
    .repeat = c.repeat,
    .clamp_to_edge = c.clamp_to_edge,
    .mirror = c.mirrored_repeat,
    .border = c.clamp_to_border,
});

const border_colors: std.EnumArray(types.BorderColor, [4]f32) = .init(.{
    .transparent_black = .{ 0, 0, 0, 0 },
    .opaque_black = .{ 0, 0, 0, 1 },
    .opaque_white = .{ 1, 1, 1, 1 },
});

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    var name: gt.Uint = 0;
    const sampler_i = fnOf(api, "samplerParameteri");
    const sampler_f = fnOf(api, "samplerParameterf");
    fnOf(api, "genSamplers")(1, @ptrCast(&name));
    sampler_i(name, c.texture_min_filter, @intCast(min_filters.get(desc.min_filter).get(desc.mip_filter)));
    sampler_i(name, c.texture_mag_filter, @intCast(filterEnum(desc.mag_filter)));
    sampler_i(name, c.texture_wrap_s, @intCast(wraps.get(desc.wrap_u)));
    sampler_i(name, c.texture_wrap_t, @intCast(wraps.get(desc.wrap_v)));
    sampler_i(name, c.texture_wrap_r, @intCast(wraps.get(desc.wrap_w)));
    sampler_f(name, c.texture_min_lod, desc.lod_min);
    sampler_f(name, c.texture_max_lod, desc.lod_max);

    // A shadow sampler compares what is stored against what the shader
    // brought; any other reads the depth as it is. Said either way, so that a
    // depth texture read with a plain sampler is never left comparing.
    sampler_i(name, c.texture_compare_mode, @intCast(if (desc.compare != null) c.compare_ref_to_texture else c.none));
    if (desc.compare) |func| sampler_i(name, c.texture_compare_func, @intCast(compare(func)));

    // What `Device` allows is up to what `caps` said; what is asked for that the
    // driver has no state for is left unset, not sent to be an error.
    if (desc.max_anisotropy > 1 and self.max_anisotropy > 1) {
        sampler_f(name, c.texture_max_anisotropy, @min(@as(f32, @floatFromInt(desc.max_anisotropy)), self.max_anisotropy));
    }
    if (desc.lod_bias != 0 and self.capabilities.features.sampler_lod_bias) {
        sampler_f(name, token.texture_lod_bias, std.math.clamp(desc.lod_bias, -self.max_lod_bias, self.max_lod_bias));
    }
    if (desc.wrap_u == .border or desc.wrap_v == .border or desc.wrap_w == .border) {
        if (self.extra.samplerParameterfv) |set_vector| {
            const color = border_colors.get(desc.border);
            set_vector(name, c.texture_border_color, &color);
        }
    }

    res.* = .{ .name = name };
    return res;
}

fn filterEnum(filter: types.Filter) gt.Enum {
    return switch (filter) {
        .nearest => c.nearest,
        .linear => c.linear,
    };
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    fnOf(&self.api, "deleteSamplers")(1, @ptrCast(&res.name));
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Shaders and pipelines
// -------------------------------------------------------------------------

fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const api = &self.api;

    const sources = (if (is_gles) desc.glsl_es else desc.glsl) orelse {
        log.print("fluxion-rhi: the {s} backend needs its GLSL source, and none was given", .{
            if (is_gles) "OpenGL ES" else "OpenGL",
        }) catch {};
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

fn compileStage(api: *const Api, kind: gt.Enum, source: [:0]const u8, log: *Io.Writer) Error!gt.Uint {
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
        const index = fnOf(api, "getUniformBlockIndex")(program, name.ptr);
        if (index == invalid_index) {
            log.print("uniform block `{s}` is not in the shader (or nothing reads it)", .{name}) catch {};
            return error.PipelineFailed;
        }
        fnOf(api, "uniformBlockBinding")(program, index, @intCast(slot));
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
    fnOf(api, "genVertexArrays")(1, @ptrCast(&vao));

    // `desc.samples` is not kept: the sample count of a draw is that of what
    // it draws into, and `Device` has already checked that the two agree.
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
        .color_write = desc.color_format != null,
    };
    return res;
}

/// `GL_INVALID_INDEX`.
const invalid_index: gt.Uint = 0xFFFFFFFF;

fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(PipelineRes, native);
    fnOf(&self.api, "deleteVertexArrays")(1, @ptrCast(&res.vao));
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
            .end_pass => try endPass(self),
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                bindPipeline(self, res);
            },
            .set_viewport => |v| {
                // Flipped: the top-left rectangle, measured from the bottom.
                const y = @as(f32, @floatFromInt(self.target_height)) - v.y - v.height;
                api.viewport(@intFromFloat(v.x), @intFromFloat(y), @intFromFloat(v.width), @intFromFloat(v.height));
                if (is_gles) api.depthRangef(v.min_depth, v.max_depth) else api.depthRange(v.min_depth, v.max_depth);
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
                fnOf(api, "bindBufferBase")(c.uniform_buffer, b.slot, res.name);
            },
            .set_texture => |b| {
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                api.activeTexture(c.texture0 + b.slot);
                // The kind that it is: a shader's `samplerCube` reads the cube
                // binding of the unit and a `sampler2D` the 2D one, so a
                // texture goes where its shape says.
                api.bindTexture(texture.target, texture.name);
                fnOf(api, "bindSampler")(b.slot, sampler.name);
            },
            .draw => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                flushBindings(self, pipeline);
                fnOf(api, "drawArraysInstanced")(pipeline.topology, @intCast(d.first_vertex), @intCast(d.vertex_count), @intCast(d.instance_count));
            },
            .generate_mips => |h| {
                const texture = as(TextureRes, device.textures.get(h).?.native);
                api.bindTexture(texture.target, texture.name);
                api.generateMipmap(texture.target);
                inheritDrawn(texture);
            },
            .draw_indexed => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                flushBindings(self, pipeline);
                const index = self.index orelse return error.InvalidArgument;
                const offset = opengl.offset(@as(usize, d.first_index) * index.size);
                if (d.base_vertex == 0) {
                    fnOf(api, "drawElementsInstanced")(pipeline.topology, @intCast(d.index_count), index.kind, offset, @intCast(d.instance_count));
                } else if (!is_gles and d.instance_count == 1) {
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

fn textureOf(device: *Device, h: types.Texture) *TextureRes {
    return as(TextureRes, device.textures.get(h).?.native);
}

fn beginPass(self: *Gl, device: *Device, pass: types.RenderPassDesc) Error!void {
    const api = &self.api;
    self.resolves = @splat(null);

    // `color` and then `extra_colors`, which is the order of the draw buffers
    // and so of a fragment shader's outputs.
    var specs: [most_color_attachments]types.ColorAttachment = undefined;
    var count: usize = 0;
    if (pass.color) |color| {
        specs[0] = color;
        count = 1;
    }
    if (count + pass.extra_colors.len > specs.len) return error.Unsupported;
    for (pass.extra_colors) |extra| {
        specs[count] = extra;
        count += 1;
    }

    // Only a pass that has nothing but the surface for a colour is drawn into
    // the default framebuffer; `Device` has seen to it that a pass with more
    // colours has none. There is no putting a depth texture on the default
    // framebuffer, so its own depth buffer serves and a `depth` beside it is
    // for the clear alone.
    const to_surface = count == 1 and specs[0].target == .surface;

    var attachments: [most_color_attachments]Attachment = undefined;
    if (!to_surface) {
        for (specs[0..count], 0..) |spec, i| {
            attachments[i] = .{ .res = textureOf(device, spec.target.texture), .mip = spec.mip_level, .layer = spec.layer };
        }
    }
    const depth: ?Attachment = if (pass.depth) |d|
        .{ .res = textureOf(device, d.texture), .mip = d.mip_level, .layer = d.layer }
    else
        null;

    var size: [2]u32 = undefined;
    if (to_surface) {
        api.bindFramebuffer(c.framebuffer, 0);
        self.pass_framebuffer = 0;
        size = self.hooks.framebuffer_size(self.hooks.context);
    } else {
        const framebuffer = try framebufferFor(self, attachments[0..count], depth);
        // What a pass draws into is stored the way a pass leaves it, from
        // here on, whether or not it draws anything: it is what a clear does too.
        for (attachments[0..count]) |image| markDrawn(image.res, image.mip, image.layer);
        api.bindFramebuffer(c.framebuffer, framebuffer);
        self.pass_framebuffer = framebuffer;
        size = if (count > 0) attachments[0].size() else depth.?.size();
    }
    // What a texture stores in an sRGB format is encoded on the way in, as it
    // is on Direct3D; the window's own framebuffer is written as it is given.
    // OpenGL ES converts every sRGB attachment and has no switch for it.
    if (!is_gles) {
        if (to_surface) api.disable(c.framebuffer_srgb) else api.enable(c.framebuffer_srgb);
    }

    for (specs[0..count], 0..) |spec, i| {
        if (spec.resolve) |target| {
            self.resolves[i] = .{ .target = switch (target) {
                .surface => null,
                .texture => |h| textureOf(device, h),
            } };
        }
    }

    self.pass_size = size;
    self.target_height = size[1];
    self.pipeline = null;
    self.bindings_dirty = true;

    // A pass begins with the whole attachment, and the whole of the depth range:
    // what an earlier pass's viewport asked of it is not this one's.
    api.viewport(0, 0, @intCast(size[0]), @intCast(size[1]));
    api.scissor(0, 0, @intCast(size[0]), @intCast(size[1]));
    if (is_gles) api.depthRangef(0, 1) else api.depthRange(0, 1);

    // A clear goes through the write masks, so they are opened first and the
    // pipeline that follows sets them back. One colour is `glClear`; several
    // each want a value of their own, and `glClear` has one for all.
    var mask: gt.Bitfield = 0;
    api.colorMask(gt.gl_true, gt.gl_true, gt.gl_true, gt.gl_true);
    if (count == 1) {
        if (specs[0].load == .clear) {
            const col = specs[0].clear_color;
            api.clearColor(col[0], col[1], col[2], col[3]);
            mask |= c.color_buffer_bit;
        }
    } else {
        for (specs[0..count], 0..) |spec, i| {
            if (spec.load == .clear) fnOf(api, "clearBufferfv")(c.color, @intCast(i), &spec.clear_color);
        }
    }
    if (pass.depth) |d| if (d.load == .clear) {
        api.depthMask(gt.gl_true);
        if (is_gles) api.clearDepthf(d.clear_depth) else api.clearDepth(d.clear_depth);
        api.clearStencil(d.clear_stencil);
        mask |= c.depth_buffer_bit | c.stencil_buffer_bit;
    };
    if (mask != 0) api.clear(mask);
}

/// Close the pass: each multisampled colour that asked to is resolved, which
/// is a blit between the pass's framebuffer and the one it resolves into.
/// A resolve cannot scale, and does not need to - `Device` has made the
/// sizes equal - but it does obey the scissor test, which is switched off
/// for it.
fn endPass(self: *Gl) Error!void {
    const api = &self.api;
    for (self.resolves, 0..) |maybe, i| {
        const resolve = maybe orelse continue;
        // The samples were drawn bottom row first, and so is what they resolve to.
        if (resolve.target) |res| markDrawn(res, 0, 0);
        const destination: gt.Uint = if (resolve.target) |res|
            try framebufferFor(self, &.{.{ .res = res, .mip = 0, .layer = 0 }}, null)
        else
            0;
        const source_color = c.colorAttachment(@intCast(i));
        api.bindFramebuffer(c.read_framebuffer, self.pass_framebuffer);
        fnOf(api, "readBuffer")(source_color);
        api.bindFramebuffer(c.draw_framebuffer, destination);
        api.disable(c.scissor_test);
        const width: gt.Int = @intCast(self.pass_size[0]);
        const height: gt.Int = @intCast(self.pass_size[1]);
        fnOf(api, "blitFramebuffer")(0, 0, width, height, 0, 0, width, height, c.color_buffer_bit, c.nearest);
        api.enable(c.scissor_test);
        // The pass's framebuffer is kept for the next pass into the same
        // images, and it reads the first colour unless told otherwise.
        api.bindFramebuffer(c.read_framebuffer, self.pass_framebuffer);
        fnOf(api, "readBuffer")(c.color_attachment0);
    }
    self.resolves = @splat(null);
    api.bindFramebuffer(c.framebuffer, 0);
    self.pipeline = null;
}

fn bindPipeline(self: *Gl, res: *PipelineRes) void {
    const api = &self.api;
    self.pipeline = res;
    self.bindings_dirty = true;

    api.useProgram(res.program);
    fnOf(api, "bindVertexArray")(res.vao);

    // A pipeline for a depth-only pass has no colour format and must not
    // write one, whatever the draw buffers of the framebuffer it is bound
    // with; and the pass before, which opened the mask to clear, is undone.
    const color: gt.Boolean = if (res.color_write) gt.gl_true else gt.gl_false;
    api.colorMask(color, color, color, color);

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
    const vertex_attrib_i_pointer = fnOf(api, "vertexAttribIPointer");
    const vertex_attrib_divisor = fnOf(api, "vertexAttribDivisor");

    for (pipeline.attributes) |attribute| {
        const binding = self.vertex_bindings[attribute.buffer];
        const layout = pipeline.buffers[attribute.buffer];
        api.bindBuffer(c.array_buffer, binding.name);
        const pointer = opengl.offset(@as(usize, binding.offset) + attribute.offset);
        const comps: gt.Int = @intCast(attribute.format.components());
        switch (attribute.format) {
            .float, .float2, .float3, .float4 => api.vertexAttribPointer(attribute.location, comps, c.float, gt.gl_false, @intCast(layout.stride), pointer),
            .ubyte4_norm => api.vertexAttribPointer(attribute.location, comps, c.unsigned_byte, gt.gl_true, @intCast(layout.stride), pointer),
            .ubyte4 => vertex_attrib_i_pointer(attribute.location, comps, c.unsigned_byte, @intCast(layout.stride), pointer),
            .uint => vertex_attrib_i_pointer(attribute.location, comps, c.unsigned_int, @intCast(layout.stride), pointer),
            .int => vertex_attrib_i_pointer(attribute.location, comps, c.int, @intCast(layout.stride), pointer),
        }
        api.enableVertexAttribArray(attribute.location);
        vertex_attrib_divisor(attribute.location, if (layout.step == .instance) 1 else 0);
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
// Tests. Anything that draws needs a context and lives in the examples, which
// have a window; what is here is the table, the arithmetic, and `caps` from a
// driver that is only a script.
// -------------------------------------------------------------------------

const testing = std.testing;

test "every format has a row, and a row that GL can be given" {
    for (std.enums.values(types.Format)) |format| {
        const row = nativeOf(format) orelse continue;
        try testing.expect(row.internal != 0);
        if (format.isCompressed()) {
            // A compressed format is uploaded whole, and has no transfer pair.
            try testing.expectEqual(@as(gt.Enum, 0), row.format);
        } else {
            try testing.expect(row.format != 0);
            try testing.expect(row.kind != 0);
        }
        // What is swapped on the way in has red and blue to swap.
        if (row.swap_rb) try testing.expectEqual(@as(usize, 4), format.bytesPerPixel());
    }
    // No format has been left out by being made `null`: this backend can spell them all.
    for (std.enums.values(types.Format)) |format| try testing.expect(nativeOf(format) != null);
}

test "the two BGRA formats are stored as their RGBA twin" {
    const bgra = nativeOf(.bgra8_unorm).?;
    const rgba = nativeOf(.rgba8_unorm).?;
    try testing.expect(bgra.swap_rb);
    try testing.expectEqual(rgba.internal, bgra.internal);
    const srgb_bgra = nativeOf(.bgra8_unorm_srgb).?;
    try testing.expect(srgb_bgra.swap_rb);
    try testing.expectEqual(nativeOf(.rgba8_unorm_srgb).?.internal, srgb_bgra.internal);
}

test "a texture is gathered tight, and swapped, from a padded pitch" {
    // Two BGRA texels a row, three bytes of padding after each, two rows.
    const bytes = [_]u8{
        1, 2,  3,  4,  5,  6,  7,  8,  0, 0, 0,
        9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0,
    };
    var out: [16]u8 = undefined;
    gather(&out, &bytes, 8, 2, 1, 11, 22, true);
    try testing.expectEqualSlices(u8, &.{ 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 16 }, &out);
    // And not swapped, only tightened.
    gather(&out, &bytes, 8, 2, 1, 11, 22, false);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, &out);
}

test "floats come back as bytes, clamped" {
    try testing.expectEqual(@as(u8, 0), unormFromFloat(-1));
    try testing.expectEqual(@as(u8, 0), unormFromFloat(std.math.nan(f32)));
    try testing.expectEqual(@as(u8, 255), unormFromFloat(7));
    try testing.expectEqual(@as(u8, 128), unormFromFloat(0.5));
}

/// A context that is only a version and a list of names.
fn context(version: []const u8, names: []const []const u8) !Context {
    return .{ .version = try opengl.Version.parse(version), .extensions = .{ .names = names } };
}

const accepted: Probe = .{ .accepted = true, .renderable = true, .sample_counts = 0b0110 };

test "what a compressed format needs is different on the desktop and on ES" {
    const ctx_gl = try context("3.3.0 Fluxion", &.{});
    const ctx_es = try context("OpenGL ES 3.0 Fluxion", &.{});
    const bc7 = nativeOf(.bc7_rgba_unorm).?.*;
    const etc2_row = nativeOf(.etc2_rgb8_unorm).?.*;

    // ETC2 is ES 3.0's own and a desktop 4.3 thing; BC is the other way about.
    try testing.expect(!formatSupport(.etc2_rgb8_unorm, etc2_row, .gl, ctx_gl, accepted).sampled);
    try testing.expect(formatSupport(.etc2_rgb8_unorm, etc2_row, .es, ctx_es, accepted).sampled);
    try testing.expect(!formatSupport(.bc7_rgba_unorm, bc7, .es, ctx_es, accepted).sampled);
    try testing.expect(!formatSupport(.bc7_rgba_unorm, bc7, .gl, ctx_gl, accepted).sampled);

    const newer = try context("4.6.0 Fluxion", &.{});
    try testing.expect(formatSupport(.bc7_rgba_unorm, bc7, .gl, newer, accepted).sampled);
    try testing.expect(formatSupport(.etc2_rgb8_unorm, etc2_row, .gl, newer, accepted).sampled);

    // Or by an extension, with a version that is not enough.
    const with_bptc = try context("3.3.0 Fluxion", &.{"GL_ARB_texture_compression_bptc"});
    try testing.expect(formatSupport(.bc7_rgba_unorm, bc7, .gl, with_bptc, accepted).sampled);
    const es_bptc = try context("OpenGL ES 3.0 Fluxion", &.{"GL_EXT_texture_compression_bptc"});
    try testing.expect(formatSupport(.bc7_rgba_unorm, bc7, .es, es_bptc, accepted).sampled);

    // A driver that says no is believed, extension or none.
    try testing.expect(!formatSupport(.bc7_rgba_unorm, bc7, .gl, newer, .{}).sampled);

    // And a compressed format is never drawn into, whatever a probe said of it.
    const s = formatSupport(.bc7_rgba_unorm, bc7, .gl, newer, accepted);
    try testing.expect(!s.render_target and !s.blendable and !s.generate_mips);
    try testing.expect(s.filterable);
    try testing.expect(s.supportsSamples(1) and !s.supportsSamples(2));
}

test "ES will not filter or blend a 32-bit float without being told it may" {
    const rgba32 = nativeOf(.rgba32_float).?.*;
    const depth = nativeOf(.depth32_float).?.*;
    const plain = try context("OpenGL ES 3.0 Fluxion", &.{});
    const told = try context("OpenGL ES 3.0 Fluxion", &.{ "GL_OES_texture_float_linear", "GL_EXT_float_blend" });

    var s = formatSupport(.rgba32_float, rgba32, .es, plain, accepted);
    try testing.expect(s.sampled and s.render_target);
    try testing.expect(!s.filterable and !s.blendable and !s.generate_mips);
    s = formatSupport(.rgba32_float, rgba32, .es, told, accepted);
    try testing.expect(s.filterable and s.blendable and s.generate_mips);

    // The desktop never asks.
    const desktop = try context("3.3.0 Fluxion", &.{});
    s = formatSupport(.rgba32_float, rgba32, .gl, desktop, accepted);
    try testing.expect(s.filterable and s.blendable and s.generate_mips);

    // A depth format is not blended, has no levels made for it, and on ES is not filtered.
    s = formatSupport(.depth32_float, depth, .es, plain, accepted);
    try testing.expect(s.render_target and !s.filterable and !s.blendable and !s.generate_mips);
    s = formatSupport(.depth32_float, depth, .gl, desktop, accepted);
    try testing.expect(s.render_target and s.filterable and !s.blendable and !s.generate_mips);
}

test "a format that could not be drawn into claims no sample count but one" {
    const ctx_gl = try context("3.3.0 Fluxion", &.{});
    const s = formatSupport(.rgba8_unorm, nativeOf(.rgba8_unorm).?.*, .gl, ctx_gl, .{ .accepted = true, .renderable = false, .sample_counts = 0b110 });
    try testing.expect(!s.render_target);
    try testing.expectEqual(@as(u8, 0b1), s.sample_counts);
    try testing.expect(!s.generate_mips and !s.blendable);
}

/// What `Features` is when a context has the given border and bias, and both
/// APIs have what they always have: any size of compressed texture, and a
/// framebuffer that is bottom-up.
fn sampling(border: bool, bias: bool) types.Features {
    return .{ .sampler_border = border, .sampler_lod_bias = bias, .compressed_partial_blocks = true, .render_target_origin_bottom_left = true };
}

test "what a sampler can do is different on the desktop and on ES" {
    const desktop = try context("3.3.0 Fluxion", &.{});
    const es_30 = try context("OpenGL ES 3.0 Fluxion", &.{});
    const es_32 = try context("OpenGL ES 3.2 Fluxion", &.{});
    const es_border = try context("OpenGL ES 3.0 Fluxion", &.{"GL_EXT_texture_border_clamp"});

    // The desktop has a border and a bias, and has no anisotropy without the extension.
    try testing.expectEqual(sampling(true, true), featuresOf(.gl, desktop, true));
    try testing.expect(!hasAnisotropy(.gl, desktop));
    // ES 3.0 has neither, and not the vector form that a border needs either.
    try testing.expectEqual(sampling(false, false), featuresOf(.es, es_30, true));
    try testing.expect(!hasAnisotropy(.es, es_30));
    // It gets the border in 3.2, or by extension, and never the bias.
    try testing.expectEqual(sampling(true, false), featuresOf(.es, es_32, true));
    try testing.expectEqual(sampling(true, false), featuresOf(.es, es_border, true));
    // A border colour is a vector: a context without the command has no border.
    try testing.expect(!featuresOf(.gl, desktop, false).sampler_border);
    try testing.expect(!featuresOf(.es, es_border, false).sampler_border);

    // Anisotropy: 4.6 has it in the core, and every version before by extension, on both APIs.
    try testing.expect(hasAnisotropy(.gl, try context("4.6.0 Fluxion", &.{})));
    try testing.expect(hasAnisotropy(.gl, try context("3.3.0 Fluxion", &.{"GL_EXT_texture_filter_anisotropic"})));
    try testing.expect(hasAnisotropy(.gl, try context("3.3.0 Fluxion", &.{"GL_ARB_texture_filter_anisotropic"})));
    try testing.expect(hasAnisotropy(.es, try context("OpenGL ES 3.0 Fluxion", &.{"GL_EXT_texture_filter_anisotropic"})));
    // Not ES 3.2, which has none of it in the core: an ES version is not a desktop one.
    try testing.expect(!hasAnisotropy(.es, es_32));
}

test "a format comes in the shapes the driver made it in, and a compressed one is never a volume" {
    const desktop = try context("4.6.0 Fluxion", &.{"GL_EXT_texture_compression_s3tc"});
    const es = try context("OpenGL ES 3.0 Fluxion", &.{});
    var flat: std.EnumSet(types.Dimension) = .initEmpty();
    flat.insert(.d2);
    flat.insert(.cube);

    // A depth format that the driver would not make as a volume has no volume.
    var no_volume: Probe = accepted;
    no_volume.dimensions.remove(.d3);
    var s = formatSupport(.depth32_float, nativeOf(.depth32_float).?.*, .gl, desktop, no_volume);
    try testing.expect(s.dimensions.contains(.d2) and s.dimensions.contains(.d2_array) and s.dimensions.contains(.cube));
    try testing.expect(!s.dimensions.contains(.d3));

    // And what it is asked as is what it says, whatever else it could do.
    var only_flat: Probe = accepted;
    only_flat.dimensions = flat;
    s = formatSupport(.rgba8_unorm, nativeOf(.rgba8_unorm).?.*, .gl, desktop, only_flat);
    try testing.expect(s.dimensions.eql(flat));

    // A compressed format is not a volume even where a probe would have said it was,
    // on either API; a stack of layers and a cube are the driver's word.
    s = formatSupport(.bc1_rgba_unorm, nativeOf(.bc1_rgba_unorm).?.*, .gl, desktop, accepted);
    try testing.expect(!s.dimensions.contains(.d3) and s.dimensions.contains(.d2_array) and s.dimensions.contains(.cube) and s.dimensions.contains(.d2));
    s = formatSupport(.etc2_rgb8_unorm, nativeOf(.etc2_rgb8_unorm).?.*, .es, es, accepted);
    try testing.expect(!s.dimensions.contains(.d3) and s.dimensions.contains(.d2_array));
    var no_stack: Probe = accepted;
    no_stack.dimensions.remove(.d2_array);
    s = formatSupport(.bc1_rgba_unorm, nativeOf(.bc1_rgba_unorm).?.*, .gl, desktop, no_stack);
    try testing.expect(!s.dimensions.contains(.d2_array) and s.dimensions.contains(.cube));

    // Nothing is claimed of a shape for a format that is not there at all.
    s = formatSupport(.bc1_rgba_unorm, nativeOf(.bc1_rgba_unorm).?.*, .gl, try context("3.3.0 Fluxion", &.{}), accepted);
    try testing.expectEqual(@as(usize, 0), s.dimensions.count());
}

/// A texture with nothing in it but what its images are counted from.
fn bookkeeping(dimension: types.Dimension, depth_or_layers: u32, mip_levels: u32) !TextureRes {
    const layers: u32 = switch (dimension) {
        .d2 => 1,
        .cube => 6,
        .d3, .d2_array => depth_or_layers,
    };
    return .{
        .name = 0,
        .serial = 1,
        .target = texture_targets.get(dimension),
        .dimension = dimension,
        .width = 16,
        .height = 16,
        .depth_or_layers = layers,
        .mip_levels = mip_levels,
        .samples = 1,
        .format = .rgba8_unorm,
        .native = nativeOf(.rgba8_unorm).?,
        .drawn = try .initEmpty(testing.allocator, imageCount(dimension, layers, mip_levels)),
    };
}

test "an image is a level of a layer, and each has its own bit" {
    // A volume of four has four slices, then two, then one: seven images.
    var volume = try bookkeeping(.d3, 4, 3);
    defer volume.drawn.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 7), volume.drawn.bit_length);
    try testing.expectEqual(@as(?usize, 0), imageIndex(&volume, 0, 0));
    try testing.expectEqual(@as(?usize, 3), imageIndex(&volume, 0, 3));
    try testing.expectEqual(@as(?usize, 5), imageIndex(&volume, 1, 1));
    try testing.expectEqual(@as(?usize, 6), imageIndex(&volume, 2, 0));
    // Not images at all: past the last slice of a level, past the last level.
    try testing.expectEqual(@as(?usize, null), imageIndex(&volume, 1, 2));
    try testing.expectEqual(@as(?usize, null), imageIndex(&volume, 3, 0));

    // An array of three has three in each of three levels.
    var array = try bookkeeping(.d2_array, 3, 3);
    defer array.drawn.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 9), array.drawn.bit_length);
    try testing.expectEqual(@as(?usize, 4), imageIndex(&array, 1, 1));
    try testing.expectEqual(@as(?usize, 8), imageIndex(&array, 2, 2));

    var cube = try bookkeeping(.cube, 1, 2);
    defer cube.drawn.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 12), cube.drawn.bit_length);
    try testing.expectEqual(@as(?usize, 11), imageIndex(&cube, 1, 5));
}

test "a pass draws an image, a write undoes it, and a chain takes after its first level" {
    var array = try bookkeeping(.d2_array, 3, 3);
    defer array.drawn.deinit(testing.allocator);

    // Nothing has been drawn: an image is as it was written.
    for (0..3) |mip| {
        for (0..3) |layer| try testing.expect(!isDrawn(&array, @intCast(mip), @intCast(layer)));
    }

    markDrawn(&array, 0, 1);
    markDrawn(&array, 1, 2);
    try testing.expect(isDrawn(&array, 0, 1) and isDrawn(&array, 1, 2));
    try testing.expect(!isDrawn(&array, 0, 0) and !isDrawn(&array, 0, 2) and !isDrawn(&array, 1, 1) and !isDrawn(&array, 2, 1));

    // A write to an image is a write to all of it, however little of it; the neighbours do not notice.
    markWritten(&array, .{ .mip = 0, .z = 1, .depth = 1, .width = 2, .height = 2 });
    try testing.expect(!isDrawn(&array, 0, 1) and isDrawn(&array, 1, 2));
    // Several layers at once.
    markDrawn(&array, 0, 0);
    markDrawn(&array, 0, 2);
    markWritten(&array, .{ .z = 0, .depth = 3 });
    try testing.expect(!isDrawn(&array, 0, 0) and !isDrawn(&array, 0, 2) and isDrawn(&array, 1, 2));

    // A chain generated from level zero is, in each layer, the way up its first level is.
    markDrawn(&array, 0, 0);
    inheritDrawn(&array);
    for (1..3) |mip| {
        try testing.expect(isDrawn(&array, @intCast(mip), 0));
        try testing.expect(!isDrawn(&array, @intCast(mip), 1) and !isDrawn(&array, @intCast(mip), 2));
    }

    // What is not an image is taken as drawn, and marking it does nothing.
    try testing.expect(isDrawn(&array, 5, 0) and isDrawn(&array, 0, 9));
    markDrawn(&array, 5, 0);
    markWritten(&array, .{ .mip = 5, .z = 0, .depth = 1 });

    // A volume's slice of a level is the way up of the first slice that goes into it.
    var volume = try bookkeeping(.d3, 4, 3);
    defer volume.drawn.deinit(testing.allocator);
    markDrawn(&volume, 0, 2);
    markDrawn(&volume, 0, 3);
    inheritDrawn(&volume);
    try testing.expect(!isDrawn(&volume, 1, 0) and isDrawn(&volume, 1, 1));
    try testing.expect(!isDrawn(&volume, 2, 0));
    markDrawn(&volume, 0, 0);
    inheritDrawn(&volume);
    try testing.expect(isDrawn(&volume, 1, 0) and isDrawn(&volume, 1, 1) and isDrawn(&volume, 2, 0));
}

test "rows are turned over whole" {
    var image = [_]u8{ 1, 1, 2, 2, 3, 3 };
    flipRows(&image, 2);
    try testing.expectEqualSlices(u8, &.{ 3, 3, 2, 2, 1, 1 }, &image);
    var one = [_]u8{ 9, 8 };
    flipRows(&one, 2);
    try testing.expectEqualSlices(u8, &.{ 9, 8 }, &one);
}

/// A driver that answers what `open` asks and stubs everything else, with a
/// version and an extension list to pretend to. Desktop only: the ES table
/// has other names to fill and a script for those is not this one.
const Fake = struct {
    var version: [:0]const u8 = "3.3.0 Fluxion";
    var names: []const [:0]const u8 = &.{};
    var max_anisotropy: f32 = 1;
    var last_samples: gt.Int = 0;
    var anchor: u8 = 0;

    pub fn glGetString(name: gt.Enum) callconv(.c) ?[*:0]const gt.Char {
        return switch (name) {
            c.version => version.ptr,
            c.renderer => "A Script",
            else => null,
        };
    }

    pub fn glGetStringi(name: gt.Enum, index: gt.Uint) callconv(.c) ?[*:0]const gt.Char {
        if (name != c.extensions or index >= names.len) return null;
        return names[index].ptr;
    }

    pub fn glGetIntegerv(pname: gt.Enum, data: [*]gt.Int) callconv(.c) void {
        data[0] = switch (pname) {
            c.num_extensions => @intCast(names.len),
            c.max_texture_size => 8192,
            c.max_3d_texture_size => 512,
            c.max_cube_map_texture_size => 4096,
            c.max_array_texture_layers => 256,
            c.max_color_attachments => 8,
            c.max_draw_buffers => 6,
            c.max_renderbuffer_size => 8192,
            c.max_samples => 8,
            else => 0,
        };
    }

    pub fn glGetFloatv(pname: gt.Enum, data: [*]gt.Float) callconv(.c) void {
        data[0] = if (pname == c.max_texture_max_anisotropy) max_anisotropy else 0;
    }

    pub fn glGetError() callconv(.c) gt.Enum {
        return c.no_error;
    }

    pub fn glCheckFramebufferStatus(target: gt.Enum) callconv(.c) gt.Enum {
        _ = target;
        return c.framebuffer_complete;
    }

    pub fn glRenderbufferStorageMultisample(target: gt.Enum, samples: gt.Sizei, internal_format: gt.Enum, width: gt.Sizei, height: gt.Sizei) callconv(.c) void {
        _ = target;
        _ = internal_format;
        _ = width;
        _ = height;
        last_samples = samples;
    }

    pub fn glGetRenderbufferParameteriv(target: gt.Enum, pname: gt.Enum, params: [*]gt.Int) callconv(.c) void {
        _ = target;
        params[0] = if (pname == c.renderbuffer_samples) last_samples else 0;
    }

    fn stub() callconv(.c) void {}

    fn get(context_: *anyopaque, name: [*:0]const u8) ?types.GlProc {
        _ = context_;
        const wanted = std.mem.span(name);
        inline for (@typeInfo(Fake).@"struct".decls) |decl| {
            if (std.mem.eql(u8, decl.name, wanted)) return @ptrCast(&@field(Fake, decl.name));
        }
        return @ptrCast(&stub);
    }

    fn swap(context_: *anyopaque) void {
        _ = context_;
    }

    fn size(context_: *anyopaque) [2]u32 {
        _ = context_;
        return .{ 64, 64 };
    }
};

fn openFake(version: [:0]const u8, names: []const [:0]const u8) !backend.Opened {
    Fake.version = version;
    Fake.names = names;
    return open(testing.allocator, .{ .backend = .gl, .gl = .{
        .context = &Fake.anchor,
        .get_proc_address = Fake.get,
        .swap_buffers = Fake.swap,
        .framebuffer_size = Fake.size,
    } });
}

test "caps come from what the driver says, and a 3.3 driver is asked for nothing it lacks" {
    if (is_gles) return error.SkipZigTest;

    const impl, const table = try openFake("3.3.0 Fluxion", &.{"GL_EXT_texture_compression_s3tc"});
    defer table.deinit(impl);
    const answer = table.caps(impl);

    try testing.expectEqual(@as(u32, 8192), answer.limits.max_texture_2d);
    try testing.expectEqual(@as(u32, 512), answer.limits.max_texture_3d);
    try testing.expectEqual(@as(u32, 4096), answer.limits.max_texture_cube);
    try testing.expectEqual(@as(u32, 256), answer.limits.max_texture_layers);
    // The fewer of the framebuffer's places and the draw buffers.
    try testing.expectEqual(@as(u32, 6), answer.limits.max_color_attachments);
    // No extension, no anisotropy; one tap is none.
    try testing.expectEqual(@as(u32, 1), answer.limits.max_anisotropy);
    // A desktop with samplers has both, and needs no extension to say so.
    try testing.expect(answer.features.sampler_border and answer.features.sampler_lod_bias);
    // What both APIs have: any size of compressed texture, and a framebuffer that is bottom-up.
    try testing.expect(answer.features.compressed_partial_blocks and answer.features.render_target_origin_bottom_left);

    // What 3.3 has in its core.
    try testing.expect(answer.formatSupport(.rgba8_unorm).render_target);
    try testing.expect(answer.formatSupport(.bc4_r_unorm).sampled);
    // What it has through an extension, and what it has not.
    try testing.expect(answer.formatSupport(.bc1_rgba_unorm).sampled);
    try testing.expect(answer.formatSupport(.bc3_rgba_unorm).sampled);
    try testing.expect(!answer.formatSupport(.bc1_rgba_unorm_srgb).sampled);
    try testing.expect(!answer.formatSupport(.bc7_rgba_unorm).sampled);
    try testing.expect(!answer.formatSupport(.etc2_rgb8_unorm).sampled);
    try testing.expect(!answer.formatSupport(.astc_4x4_unorm).sampled);
    // A count the driver gave is a count claimed, and one it did not is not.
    try testing.expect(answer.formatSupport(.rgba8_unorm).supportsSamples(4));
    try testing.expect(answer.formatSupport(.rgba8_unorm).supportsSamples(8));
    try testing.expect(!answer.formatSupport(.rgba8_unorm).supportsSamples(16));
}

test "a newer driver has more, and anisotropy is what it says it is" {
    if (is_gles) return error.SkipZigTest;

    Fake.max_anisotropy = 16;
    defer Fake.max_anisotropy = 1;
    const impl, const table = try openFake("4.6.0 Fluxion", &.{"GL_EXT_texture_filter_anisotropic"});
    defer table.deinit(impl);
    const answer = table.caps(impl);

    try testing.expectEqual(@as(u32, 16), answer.limits.max_anisotropy);
    try testing.expect(answer.formatSupport(.bc7_rgba_unorm).sampled);
    try testing.expect(answer.formatSupport(.etc2_rgba8_unorm).sampled);
    // Still no S3TC: that is an extension on every version.
    try testing.expect(!answer.formatSupport(.bc1_rgba_unorm).sampled);
    try testing.expect(!answer.formatSupport(.astc_6x6_unorm).sampled);
}

test "caps never claim a format the table cannot do" {
    if (is_gles) return error.SkipZigTest;

    const impl, const table = try openFake("4.6.0 Fluxion", &.{
        "GL_EXT_texture_compression_s3tc",
        "GL_EXT_texture_sRGB",
        "GL_KHR_texture_compression_astc_ldr",
    });
    defer table.deinit(impl);
    const answer = table.caps(impl);

    for (std.enums.values(types.Format)) |format| {
        const support = answer.formatSupport(format);
        if (support.sampled or support.render_target or support.filterable or support.blendable or support.generate_mips) {
            try testing.expect(nativeOf(format) != null);
        }
        // Each of these needs the one before it.
        if (support.render_target or support.filterable) try testing.expect(support.sampled);
        if (support.blendable or support.generate_mips) try testing.expect(support.render_target);
        if (format.isCompressed()) try testing.expect(!support.render_target);
        if (format.isDepth()) try testing.expect(!support.blendable and !support.generate_mips);
        // One sample is always there for what is there at all.
        try testing.expectEqual(support.sampled, support.supportsSamples(1));
        if (support.sample_counts > 0b1) try testing.expect(support.render_target);
    }
    // With those three extensions and a 4.6 core, every format there is is here.
    for (std.enums.values(types.Format)) |format| try testing.expect(answer.formatSupport(format).sampled);
}

test "a device with no hooks is refused, not opened" {
    try testing.expectError(error.NoDevice, open(std.testing.allocator, .{ .backend = .gl }));
}
