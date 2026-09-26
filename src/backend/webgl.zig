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
//! **What this device can do is `caps`, and `caps` is computed.** Every format
//! is a row in `natives` - what WebGL calls it, how its bytes are laid out,
//! what it needs - and `computeCaps` turns the rows into `FormatSupport` by
//! asking the context: a framebuffer is built around a texel of each format
//! and asked whether it is complete, which is the only honest answer to "can
//! this be drawn into" when the answer depends on an extension the page did or
//! did not switch on. Limits are the context's own. Nothing is guessed from
//! the name of the backend.
//!
//! **The page switches extensions on, not this module.** `fluxion-webgl` has
//! no `getExtension`, so this backend can neither enable nor list them. It
//! finds out what the page did by what then works: `EXT_color_buffer_float`
//! or `EXT_color_buffer_half_float` make a float framebuffer complete, and
//! `EXT_texture_filter_anisotropic` makes its limit answer. A page that wants
//! float render targets or anisotropy calls `getExtension` on the context it
//! made, before it instantiates the module - `examples/web/index.html` does.
//! What no query can reveal (`OES_texture_float_linear`, `EXT_float_blend`)
//! is reported as absent, and the formats that need it say so in `natives`.
//!
//! **What the binding cannot say, the caps do not claim.** `wire` lists what
//! `fluxion-webgl` has no call for - `texImage3D` (so no volume and no array
//! texture), `renderbufferStorageMultisample` and `blitFramebuffer` (so no
//! multisampling), `drawBuffers` (so one colour attachment), the compressed
//! upload calls, `samplerParameterf` - and the caps leave each of them out
//! rather than pretend. A test fails the day the binding grows one of them, as
//! a reminder that a `false` there has become a task.
//!
//! **A texture is bound to what it is.** A cube map is `texture_cube_map`, its
//! faces are reached by a face target on the same texture, and every mip
//! level of every face is allocated when the texture is made - there is no
//! `texStorage` in the binding, so it is one `texImage2D` per image.
//!
//! **A drawn image and a written one are stored upside down from each other.**
//! GL keeps a drawn image bottom row first and an uploaded one as it was
//! given, so `readTexture` turns over the images a pass has drawn into - it
//! keeps a bit for each - and hands back the others as they were written.
//! Either way the rows come back top first, which is the contract.
//!
//! **What a sampler cannot do to a texture, it is not asked to.** A depth
//! texture and a 32-bit float one are incomplete to WebGL if a linear filter
//! reads them, and are sampled as black rather than filtered less. A sampler
//! that asks for one anyway is given a nearest twin for those textures, made
//! the first time; a sampler that compares keeps its filter, because that is
//! what a shadow map is.
//!
//! **No base vertex.** WebGL 2 has no `drawElementsBaseVertex` at all, so an
//! indexed draw with one moves the per-vertex attribute pointers that many
//! vertices on instead. The pointers are specified at the draw in any case,
//! which makes it free while the base vertex stays put - and unlike OpenGL
//! 3.3, it combines with instancing.
//!
//! **No BGRA.** WebGL takes no `bgra` upload, so a BGRA texture is stored as
//! RGBA with red and blue swapped on the way in. The storage is then the
//! colour the caller meant, which is why nothing is swapped on the way out:
//! a readback is RGBA whatever the texture was, which is the contract, and a
//! BGRA target that was drawn red reads back red.
//!
//! **No debug output.** There is no callback to hand the driver; with
//! `DeviceDesc.debug` the error queue is drained after every submit instead.
//!
//! It builds for every target - against `fluxion-webgl`'s stub off wasm - and
//! `Device` opens it off wasm only under test. The suite at the bottom checks
//! the bookkeeping on a machine with no browser: what the caps are, what was
//! bound where, what was given back, which way each rectangle was flipped, and
//! the values the backend computes before it calls out (pitches, levels,
//! sampler parameters). Whether the picture is right is `examples/web.zig`'s
//! question, asked of a real browser.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const webgl = @import("fluxion_webgl");
const c = webgl.enums;
const Enum = webgl.types.Enum;
/// The calls as the wire has them: an import in a browser, the stub's own
/// function everywhere else. `Context` wraps almost all of them; the one
/// query it does not is `getParameterInt`, and `parameter` is the one place
/// that reaches past it.
const raw = webgl.api.raw;

const types = @import("../types.zig");
const backend = @import("../backend.zig");
const commands = @import("../commands.zig");
const Device = @import("../Device.zig");

const Error = backend.Error;

// -------------------------------------------------------------------------
// What the wire can do
// -------------------------------------------------------------------------

/// What `fluxion-webgl` has no call for, and so what this device cannot do
/// however well WebGL 2 could. Each is a fact about the binding and not about
/// the browser, which is why the caps derived from it say "no" rather than
/// asking: there is nothing to ask with. The day a call appears, the test at
/// the bottom of this file fails, and the flag and the code behind it are
/// changed together.
const wire = struct {
    /// `texImage3D`, `texSubImage3D`: what a volume and an array texture are
    /// both made and written with. Without them `texture_3d` and
    /// `texture_2d_array` cannot be allocated, filled, or - having no
    /// `framebufferTextureLayer` either - drawn into.
    const layered = false;
    /// `renderbufferStorageMultisample` and `blitFramebuffer`: a multisampled
    /// target, and the resolve that ends a pass into it. `getInternalformatParameter`
    /// would say how many samples a format has.
    const multisample = false;
    /// `drawBuffers`: without it a framebuffer writes its first colour
    /// attachment and nothing else.
    const draw_buffers = false;
    /// `compressedTexImage2D`, `compressedTexSubImage2D`.
    const compressed_upload = false;
    /// `getExtension`, `getSupportedExtensions`. A compressed format is only
    /// there once its extension has been switched on, and nothing here can.
    const extension_query = false;
    /// `samplerParameterf`. Sampler state goes over as integers, so a level
    /// range is whole levels.
    const fractional_lod = false;
};

/// Whether the context has switched a WebGL extension on. Only `false` can be
/// said while `wire.extension_query` is: see there.
fn extensionEnabled(name: []const u8) bool {
    _ = name;
    return wire.extension_query;
}

/// One integer of context state, or zero where the context does not know the
/// token. For a token that belongs to an extension that means "not switched
/// on", and WebGL queues `invalid_enum` for it, which `computeCaps` drains.
fn parameter(pname: Enum) i32 {
    return raw.getParameterInt(pname);
}

/// Tokens `fluxion-webgl`'s `enums` does not name. It says to pass those as
/// literals; a name says more, and the values are the ones the OpenGL ES 3.0
/// registry and the WebGL extension registry print.
const more = struct {
    // Sized formats.
    const r16f: Enum = 0x822D;
    const rg16f: Enum = 0x822F;
    const r32f: Enum = 0x822E;
    const rg32f: Enum = 0x8230;
    const rgb10_a2: Enum = 0x8059;
    const r11f_g11f_b10f: Enum = 0x8C3A;
    const depth32f_stencil8: Enum = 0x8CAD;
    // Pixel types.
    const unsigned_int_2_10_10_10_rev: Enum = 0x8368;
    const unsigned_int_10f_11f_11f_rev: Enum = 0x8C3B;
    const float_32_unsigned_int_24_8_rev: Enum = 0x8DAD;
    // Sampler state.
    const texture_min_lod: Enum = 0x813A;
    const texture_max_lod: Enum = 0x813B;
    const texture_compare_mode: Enum = 0x884C;
    const texture_compare_func: Enum = 0x884D;
    const compare_ref_to_texture: Enum = 0x884E;
    const texture_max_anisotropy: Enum = 0x84FE;
    /// The limit, answered once the page has switched on
    /// `EXT_texture_filter_anisotropic`.
    const max_texture_max_anisotropy: Enum = 0x84FF;
    // Limits.
    const max_3d_texture_size: Enum = 0x8073;
    const max_color_attachments: Enum = 0x8CDF;
    // Compressed formats: the token `compressedTexImage2D` takes, by the
    // extension that brings it.
    const s3tc_rgba_dxt1: Enum = 0x83F1;
    const s3tc_rgba_dxt5: Enum = 0x83F3;
    const s3tc_srgb_alpha_dxt1: Enum = 0x8C4D;
    const s3tc_srgb_alpha_dxt5: Enum = 0x8C4F;
    const rgtc_red: Enum = 0x8DBB;
    const rgtc_red_green: Enum = 0x8DBD;
    const bptc_rgba: Enum = 0x8E8C;
    const bptc_srgb_alpha: Enum = 0x8E8D;
    const bptc_rgb_ufloat: Enum = 0x8E8F;
    const etc2_rgb8: Enum = 0x9274;
    const etc2_srgb8: Enum = 0x9275;
    const etc2_rgba8: Enum = 0x9278;
    const etc2_srgb8_alpha8: Enum = 0x9279;
    const astc_4x4: Enum = 0x93B0;
    const astc_6x6: Enum = 0x93B4;
    const astc_8x8: Enum = 0x93B7;
    const astc_4x4_srgb: Enum = 0x93D0;
    const astc_6x6_srgb: Enum = 0x93D4;
    const astc_8x8_srgb: Enum = 0x93D7;
};

/// The WebGL extension each family of compressed formats is in, by the name
/// `getExtension` takes.
const extension = struct {
    const s3tc = "WEBGL_compressed_texture_s3tc";
    const s3tc_srgb = "WEBGL_compressed_texture_s3tc_srgb";
    const rgtc = "EXT_texture_compression_rgtc";
    const bptc = "EXT_texture_compression_bptc";
    const etc2 = "WEBGL_compressed_texture_etc";
    const astc = "WEBGL_compressed_texture_astc";
};

// -------------------------------------------------------------------------
// Formats
// -------------------------------------------------------------------------

/// What one `types.Format` is in WebGL 2. Everything the backend knows about
/// a format that `types.Format.info()` does not is here, and everything that
/// does say (bytes, channels, depth, stencil, sRGB) is taken from there and
/// not repeated.
const Native = struct {
    /// What the driver stores: a sized internal format, or a compressed one.
    internal: Enum,
    /// How the caller's bytes are laid out: the `format` and `type` of
    /// `texImage2D`. Zero for a compressed format, which goes over as blocks
    /// and has neither.
    format: Enum = 0,
    kind: Enum = 0,
    /// WebGL has no BGRA upload, so these are stored as RGBA and the caller's
    /// red and blue change places on the way in. The storage is then what the
    /// caller meant, and reading it back needs no swap.
    swap_rb: bool = false,
    /// Linear filtering works with nothing switched on. False for 32-bit
    /// float, which needs `OES_texture_float_linear` - undetectable from here
    /// - and for depth, which a sampler only filters when it compares.
    filter: bool = true,
    /// Blending works once the format can be drawn into. False for 32-bit
    /// float, which needs `EXT_float_blend`, undetectable from here.
    blend: bool = true,
    /// The extension a compressed format is in. Null for what is core.
    extension: ?[]const u8 = null,
};

const NativeRow = struct { types.Format, ?Native };

/// One row per `Format`, in any order: what WebGL 2 makes of it, or null for
/// a format it cannot do at all. A format without a row does not compile, and
/// neither does one with two. What a *device* can do with a row is `caps`.
const native_rows = [_]NativeRow{
    .{ .r8_unorm, .{ .internal = c.r8, .format = c.red, .kind = c.unsigned_byte } },
    .{ .rg8_unorm, .{ .internal = c.rg8, .format = c.rg, .kind = c.unsigned_byte } },
    .{ .rgba8_unorm, .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte } },
    .{ .rgba8_unorm_srgb, .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte } },
    .{ .bgra8_unorm, .{ .internal = c.rgba8, .format = c.rgba, .kind = c.unsigned_byte, .swap_rb = true } },
    .{ .bgra8_unorm_srgb, .{ .internal = c.srgb8_alpha8, .format = c.rgba, .kind = c.unsigned_byte, .swap_rb = true } },
    .{ .r16_float, .{ .internal = more.r16f, .format = c.red, .kind = c.half_float } },
    .{ .rg16_float, .{ .internal = more.rg16f, .format = c.rg, .kind = c.half_float } },
    .{ .rgba16_float, .{ .internal = c.rgba16f, .format = c.rgba, .kind = c.half_float } },
    .{ .r32_float, .{ .internal = more.r32f, .format = c.red, .kind = c.float, .filter = false, .blend = false } },
    .{ .rg32_float, .{ .internal = more.rg32f, .format = c.rg, .kind = c.float, .filter = false, .blend = false } },
    .{ .rgba32_float, .{ .internal = c.rgba32f, .format = c.rgba, .kind = c.float, .filter = false, .blend = false } },
    .{ .rgb10a2_unorm, .{ .internal = more.rgb10_a2, .format = c.rgba, .kind = more.unsigned_int_2_10_10_10_rev } },
    .{ .rg11b10_float, .{ .internal = more.r11f_g11f_b10f, .format = c.rgb, .kind = more.unsigned_int_10f_11f_11f_rev } },

    .{ .depth16_unorm, .{ .internal = c.depth_component16, .format = c.depth_component, .kind = c.unsigned_short, .filter = false, .blend = false } },
    .{ .depth24_stencil8, .{ .internal = c.depth24_stencil8, .format = c.depth_stencil, .kind = c.unsigned_int_24_8, .filter = false, .blend = false } },
    .{ .depth32_float, .{ .internal = c.depth_component32f, .format = c.depth_component, .kind = c.float, .filter = false, .blend = false } },
    .{ .depth32_float_stencil8, .{ .internal = more.depth32f_stencil8, .format = c.depth_stencil, .kind = more.float_32_unsigned_int_24_8_rev, .filter = false, .blend = false } },

    .{ .bc1_rgba_unorm, .{ .internal = more.s3tc_rgba_dxt1, .extension = extension.s3tc } },
    .{ .bc1_rgba_unorm_srgb, .{ .internal = more.s3tc_srgb_alpha_dxt1, .extension = extension.s3tc_srgb } },
    .{ .bc3_rgba_unorm, .{ .internal = more.s3tc_rgba_dxt5, .extension = extension.s3tc } },
    .{ .bc3_rgba_unorm_srgb, .{ .internal = more.s3tc_srgb_alpha_dxt5, .extension = extension.s3tc_srgb } },
    .{ .bc4_r_unorm, .{ .internal = more.rgtc_red, .extension = extension.rgtc } },
    .{ .bc5_rg_unorm, .{ .internal = more.rgtc_red_green, .extension = extension.rgtc } },
    .{ .bc6h_rgb_ufloat, .{ .internal = more.bptc_rgb_ufloat, .extension = extension.bptc } },
    .{ .bc7_rgba_unorm, .{ .internal = more.bptc_rgba, .extension = extension.bptc } },
    .{ .bc7_rgba_unorm_srgb, .{ .internal = more.bptc_srgb_alpha, .extension = extension.bptc } },
    .{ .etc2_rgb8_unorm, .{ .internal = more.etc2_rgb8, .extension = extension.etc2 } },
    .{ .etc2_rgb8_unorm_srgb, .{ .internal = more.etc2_srgb8, .extension = extension.etc2 } },
    .{ .etc2_rgba8_unorm, .{ .internal = more.etc2_rgba8, .extension = extension.etc2 } },
    .{ .etc2_rgba8_unorm_srgb, .{ .internal = more.etc2_srgb8_alpha8, .extension = extension.etc2 } },
    .{ .astc_4x4_unorm, .{ .internal = more.astc_4x4, .extension = extension.astc } },
    .{ .astc_4x4_unorm_srgb, .{ .internal = more.astc_4x4_srgb, .extension = extension.astc } },
    .{ .astc_6x6_unorm, .{ .internal = more.astc_6x6, .extension = extension.astc } },
    .{ .astc_6x6_unorm_srgb, .{ .internal = more.astc_6x6_srgb, .extension = extension.astc } },
    .{ .astc_8x8_unorm, .{ .internal = more.astc_8x8, .extension = extension.astc } },
    .{ .astc_8x8_unorm_srgb, .{ .internal = more.astc_8x8_srgb, .extension = extension.astc } },
};

const natives: std.EnumArray(types.Format, ?Native) = blk: {
    var table: std.EnumArray(types.Format, ?Native) = .initUndefined();
    var seen: [types.Format.count]bool = @splat(false);
    for (native_rows) |row| {
        const at = @intFromEnum(row[0]);
        if (seen[at]) @compileError("natives: two rows for " ++ @tagName(row[0]));
        seen[at] = true;
        table.set(row[0], row[1]);
    }
    for (seen, 0..) |has_row, at| {
        if (!has_row) @compileError("natives: no row for " ++ @tagName(@as(types.Format, @enumFromInt(at))));
    }
    break :blk table;
};

/// The token a texture of each shape is bound to. Layered ones are here for
/// what they are; whether one can be made is `wire.layered`.
const dimension_targets: std.EnumArray(types.Dimension, Enum) = .init(.{
    .d2 = c.texture_2d,
    .cube = c.texture_cube_map,
    .d3 = c.texture_3d,
    .d2_array = c.texture_2d_array,
});

/// Where a format is attached to a framebuffer: the colour point for colour,
/// and for depth the one that takes the stencil along when there is one.
fn attachmentPoint(format: types.Format) Enum {
    if (!format.isDepth()) return c.color_attachment0;
    return if (format.hasStencil()) c.depth_stencil_attachment else c.depth_attachment;
}

/// What a device can do with a format WebGL 2 has, given whether a
/// framebuffer around it turned out complete. Kept apart from the asking so
/// the rules can be tested on answers the stub cannot give.
fn supportFor(format: types.Format, native: Native, renders: bool) types.FormatSupport {
    if (format.isCompressed()) {
        // Sampled once its extension is on and there is a call to upload it
        // with; neither can be said yet. Never drawn into, never resampled.
        if (!wire.compressed_upload or !extensionEnabled(native.extension orelse return .{})) return .{};
        return .{ .sampled = true, .filterable = true, .sample_counts = single_sample };
    }
    const depth = format.isDepth();
    return .{
        .sampled = true,
        .filterable = native.filter,
        .render_target = renders,
        .blendable = renders and native.blend and !depth,
        // `generateMipmap` wants a format that is both drawable and filterable.
        .generate_mips = renders and native.filter and !depth,
        .sample_counts = if (renders) single_sample else 0,
    };
}

/// One sample: the only count a target can have while `wire.multisample` is
/// false. Bit zero, as `FormatSupport.sample_counts` counts.
const single_sample: u8 = 0b1;

/// Whether a framebuffer around one texel of a format is complete. The
/// context's own word on "can this be drawn into", which is right whatever
/// extensions the page switched on and whatever the specification promises.
fn probeRenderable(gl: webgl.Context, format: types.Format, native: Native) bool {
    const texture = gl.createTexture() catch return false;
    defer gl.deleteTexture(texture);
    const fbo = gl.createFramebuffer() catch return false;
    defer gl.deleteFramebuffer(fbo);

    gl.bindTexture(c.texture_2d, texture);
    gl.texImage2D(.{ .internal_format = native.internal, .width = 1, .height = 1, .format = native.format, .kind = native.kind });
    gl.bindFramebuffer(c.framebuffer, fbo);
    gl.framebufferTexture2D(c.framebuffer, attachmentPoint(format), c.texture_2d, texture, 0);
    const complete = if (gl.checkFramebuffer(c.framebuffer)) |_| true else |_| false;
    gl.bindFramebuffer(c.framebuffer, .none);
    gl.bindTexture(c.texture_2d, .none);
    return complete;
}

/// The whole of `Caps`, asked of the context.
fn computeCaps(gl: webgl.Context) types.Caps {
    // Anisotropy has an answer once the page switched its extension on; before
    // that the token is refused, and one tap is what there is.
    const taps = parameter(more.max_texture_max_anisotropy);

    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = gl.limits.max_texture_size,
            .max_texture_cube = gl.limits.max_cube_map_texture_size,
            // No `texImage3D`, so no volume and no layers whatever the context allows.
            .max_texture_3d = if (wire.layered) @intCast(@max(parameter(more.max_3d_texture_size), 0)) else 0,
            .max_texture_layers = if (wire.layered) @intCast(@max(parameter(c.max_array_texture_layers), 0)) else 0,
            .max_anisotropy = if (taps > 1) @intCast(taps) else 1,
            // No `drawBuffers`: the first attachment is the only one written.
            .max_color_attachments = if (wire.draw_buffers) @intCast(@max(parameter(more.max_color_attachments), 1)) else 1,
        },
        // ES 3.0 has neither: a border colour is desktop GL's, and a bias is
        // the shader's to add with `texture(sampler, uv, bias)`.
        .features = .{
            .sampler_border = false,
            .sampler_lod_bias = false,
            // The framebuffer is bottom-up here as on desktop GL.
            .render_target_origin_bottom_left = true,
        },
    };

    for (std.enums.values(types.Format)) |format| {
        const native = natives.get(format) orelse continue;
        const renders = !format.isCompressed() and probeRenderable(gl, format, native);
        var support = supportFor(format, native, renders);
        // No `texImage3D`: a format comes in no shape that needs it.
        if (!wire.layered and (support.sampled or support.render_target)) {
            support.dimensions.remove(.d3);
            support.dimensions.remove(.d2_array);
        }
        answer.formats.set(format, support);
    }

    // A probe that failed, and a token that was refused, both left errors in
    // the queue. They are not the program's.
    _ = gl.checkError();
    return answer;
}

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const WebGl = struct {
    gpa: Allocator,
    gl: webgl.Context,
    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,
    /// What this device can do, asked once when it opened.
    caps: types.Caps,
    /// The one surface there is, the canvas. Made once, handed back on
    /// every `createSurface`.
    surface: SurfaceRes = .{},
    /// The framebuffer every pass into a texture and every readback goes
    /// through, made the first time one is needed. What is attached to it is
    /// whatever the pass or the read names, and it is emptied again after, so
    /// that a texture destroyed later is not kept alive by it.
    fbo: webgl.Framebuffer = .none,
    attached: Attached = .{},

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

/// How many images of a texture whether a pass has drawn into them is
/// remembered for: six faces of sixteen levels, which is a cube of 32768 and
/// more than any context makes. A texture with more is read as drawn.
const max_tracked_images = 96;

const TextureRes = struct {
    texture: webgl.Texture,
    /// What it is bound to: `texture_2d`, or `texture_cube_map`.
    target: Enum,
    native: Native,
    format: types.Format,
    width: u32,
    height: u32,
    /// Faces, and levels in each: what `drawn` is indexed by.
    faces: u32,
    levels: u32,
    /// The images a pass has drawn into. GL stores a drawn image bottom row
    /// first, and an uploaded one as it was given, so a readback turns over
    /// exactly the first kind to give both top row first. See `isDrawn`.
    drawn: std.bit_set.IntegerBitSet(max_tracked_images) = .initEmpty(),
    /// Whether a linear filter reads it, from the device's caps: a sampler
    /// that asks for one is given a nearest twin otherwise. See `samplerFor`.
    filterable: bool,
};

const SamplerRes = struct {
    sampler: webgl.Sampler,
    /// What it was made from, kept for the twin.
    desc: types.SamplerDesc,
    /// The same sampler with every filter nearest, made the first time it
    /// reads a texture that cannot be filtered. See `samplerFor`.
    unfiltered: webgl.Sampler = .none,
};

/// What is on the scratch framebuffer right now.
const Attached = struct {
    color: bool = false,
    /// Which depth point holds a texture, if any: it depends on the format.
    depth: ?Enum = null,
};

const ShaderRes = struct {
    program: webgl.Program,
    /// Pipelines made from it that are still alive. The glue forgets a program
    /// the moment it is deleted, so a pipeline that outlived its shader would
    /// draw with nothing: the program goes when the last of them does.
    pipelines: u32 = 0,
    /// Destroyed by the caller, and kept until `pipelines` is nought.
    destroyed: bool = false,
};

const PipelineRes = struct {
    shader: *ShaderRes,
    program: webgl.Program,
    vao: webgl.VertexArray,
    attributes: []types.VertexAttribute,
    buffers: []types.VertexBufferLayout,
    topology: Enum,
    blend: types.BlendState,
    depth: types.DepthState,
    cull: types.CullMode,
    front_face: types.FrontFace,
    /// A pipeline with no colour format draws depth and nothing else.
    writes_color: bool,
};

const SurfaceRes = struct {
    /// A surface is the canvas's drawing buffer; there is nothing to store
    /// but the fact that it has been handed out.
    claimed: bool = false,
};

// -------------------------------------------------------------------------
// Opening
// -------------------------------------------------------------------------

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!backend.Opened {
    const gl: webgl.Context = .init();
    // Everything below is WebGL 2: vertex arrays, instancing, uniform blocks
    // and samplers are all core there, and the last two are in WebGL 1 in no
    // form at all. A page that got a WebGL 1 context has no device here.
    if (!gl.version.atLeast(.webgl2)) return error.NoDevice;

    const self = try gpa.create(WebGl);
    self.* = .{ .gpa = gpa, .gl = gl, .debug = desc.debug, .caps = undefined };
    self.renderer_len = gl.string(c.renderer, &self.renderer).len;

    // State that never changes under this backend.
    gl.pixelStorei(c.pack_alignment, 1);
    gl.pixelStorei(c.unpack_alignment, 1);
    gl.enable(c.scissor_test);

    self.caps = computeCaps(gl);

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
    .caps = caps,
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

fn cast(impl: backend.Impl) *WebGl {
    return @ptrCast(@alignCast(impl));
}

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

fn deinit(impl: backend.Impl) void {
    const self = cast(impl);
    if (self.fbo != .none) self.gl.deleteFramebuffer(self.fbo);
    self.gpa.destroy(self);
}

fn caps(impl: backend.Impl) types.Caps {
    return cast(impl).caps;
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

/// Whether a texture of this target can be made and written, given what the
/// binding can call. The two flat shapes always can.
fn targetUsable(target: Enum) bool {
    return target == c.texture_2d or target == c.texture_cube_map or wire.layered;
}

/// The size of a texture at one mip level.
fn levelExtent(res: *const TextureRes, mip: u32) [2]u32 {
    return .{ types.mipExtent(res.width, mip), types.mipExtent(res.height, mip) };
}

/// The target that names one face of a texture: the texture's own, unless it
/// is a cube, whose faces are six targets on one texture.
fn faceTarget(target: Enum, face: u32) Enum {
    return if (target == c.texture_cube_map) c.cubeFace(face) else target;
}

/// Where an image is in `TextureRes.drawn`, or null past what is tracked.
fn imageIndex(res: *const TextureRes, face: u32, level: u32) ?usize {
    const at = @as(usize, face) * res.levels + level;
    return if (at < max_tracked_images) at else null;
}

fn markDrawn(res: *TextureRes, face: u32, level: u32) void {
    if (imageIndex(res, face, level)) |at| res.drawn.set(at);
}

/// Whether an image is stored the way a pass leaves it. What was not drawn
/// into is as it was uploaded; and one that cannot be told is taken as drawn,
/// which is how every image was read before the two were told apart.
fn isDrawn(res: *const TextureRes, face: u32, level: u32) bool {
    const at = imageIndex(res, face, level) orelse return true;
    return res.drawn.isSet(at);
}

/// A write puts an image back to the order it was given in. The granularity
/// is the image: a box inside a drawn image makes the whole of it a written
/// one, and what was drawn in the rest of it then reads back the other way up.
fn markWritten(res: *TextureRes, region: types.TextureRegion) void {
    for (0..region.depth) |i| {
        if (imageIndex(res, region.z + @as(u32, @intCast(i)), region.mip)) |at| res.drawn.unset(at);
    }
}

/// `generateMipmap` filters level zero down, so a level comes out the way up
/// its first was: each face's.
fn inheritDrawn(res: *TextureRes) void {
    for (0..res.faces) |face| {
        if (!isDrawn(res, @intCast(face), 0)) continue;
        for (1..res.levels) |level| markDrawn(res, @intCast(face), @intCast(level));
    }
}

/// A rectangle of one image of a texture: a face, at a level.
const Box = struct {
    face: u32,
    level: u32,
    x: u32 = 0,
    y: u32 = 0,
    width: u32,
    height: u32,
};

/// The whole of image `index` of a texture, which is made of one per face per
/// level, level by level within a face. There is no `texStorage` in the
/// binding, so this is what "all the levels are there" is a loop over.
fn imageAt(width: u32, height: u32, levels: u32, index: u32) Box {
    const level = index % levels;
    return .{
        .face = index / levels,
        .level = level,
        .width = types.mipExtent(width, level),
        .height = types.mipExtent(height, level),
    };
}

fn textureOf(device: *Device, h: types.Texture) *TextureRes {
    return as(TextureRes, device.textures.get(h).?.native);
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    const gl = self.gl;

    // Everything `caps` promised is done below; this is what it left out, for
    // a caller that did not ask first.
    const native = natives.get(desc.format) orelse return error.Unsupported;
    const target = dimension_targets.get(desc.dimension);
    if (!targetUsable(target) or desc.samples != 1) return error.Unsupported;
    if (desc.format.isCompressed() and !wire.compressed_upload) return error.Unsupported;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const texture = gl.createTexture() catch return error.OutOfMemory;
    errdefer gl.deleteTexture(texture);
    gl.bindTexture(target, texture);

    res.* = .{
        .texture = texture,
        .target = target,
        .native = native,
        .format = desc.format,
        .width = desc.width,
        .height = desc.height,
        .faces = desc.layers(),
        .levels = desc.mip_levels,
        .filterable = self.caps.formatSupport(desc.format).filterable,
    };

    const levels = desc.mip_levels;
    const pitch = desc.effectiveRowPitch();
    // A face after a face, each `pitch` a row and as many rows as it has.
    const face_bytes = pitch * desc.format.rowCount(desc.height);
    for (0..@as(usize, desc.layers()) * levels) |i| {
        const image = imageAt(desc.width, desc.height, levels, @intCast(i));
        // What was given is level zero; the levels below are for
        // `generateMips` or for writes.
        const data: ?[]const u8 = if (image.level == 0 and desc.data != null) desc.data.?[image.face * face_bytes ..] else null;
        try upload(self, res, image, data, pitch, .allocate);
    }

    // Every level is stored, so say so: the default is a whole chain down to
    // one texel, and a texture that stops short of it samples black.
    gl.texParameteri(target, c.texture_max_level, @intCast(levels - 1));
    // Read without a sampler it is filtered as far as its format allows.
    const filter: Enum = if (res.filterable) c.linear else c.nearest;
    gl.texParameteri(target, c.texture_min_filter, @intCast(filter));
    gl.texParameteri(target, c.texture_mag_filter, @intCast(filter));
    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    // One left on the framebuffer by a pass that never ended would outlive it.
    detachPass(self);
    self.gl.deleteTexture(res.texture);
    self.gpa.destroy(res);
}

fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (res.format.isCompressed() and !wire.compressed_upload) return error.Unsupported;

    self.gl.bindTexture(res.target, res.texture);
    // `z` and `depth` are faces here, and each is its own upload.
    for (0..region.depth) |i| {
        const box: Box = .{
            .face = region.z + @as(u32, @intCast(i)),
            .level = region.mip,
            .x = region.x,
            .y = region.y,
            .width = region.width,
            .height = region.height,
        };
        try upload(self, res, box, bytes[i * slice_pitch ..], row_pitch, .replace);
    }
    markWritten(res, region);
}

/// How the bytes of one upload reach WebGL.
const UploadPlan = struct {
    /// Texels from one row of the source to the next, when that is not the
    /// width: what `UNPACK_ROW_LENGTH` is set to. Zero when the rows are
    /// tight, and always zero after a repack.
    row_length: u32,
    /// The bytes must be gathered into a tight copy first: red and blue have
    /// to change places, or the pitch is not a whole number of texels and
    /// `UNPACK_ROW_LENGTH` cannot say it.
    repack: bool,
};

fn planUpload(native: Native, texel_bytes: usize, width: u32, row_pitch: usize) UploadPlan {
    if (native.swap_rb or row_pitch % texel_bytes != 0) return .{ .row_length = 0, .repack = true };
    const tight = @as(usize, width) * texel_bytes;
    return .{ .row_length = if (row_pitch == tight) 0 else @intCast(row_pitch / texel_bytes), .repack = false };
}

/// Put texels into one rectangle of one image - or, with `.allocate` and no
/// data, make the image and leave it empty, which is what a render target
/// wants. The texture is bound.
fn upload(
    self: *WebGl,
    res: *const TextureRes,
    box: Box,
    data: ?[]const u8,
    row_pitch: usize,
    how: enum { allocate, replace },
) Error!void {
    const gl = self.gl;
    const native = res.native;
    var image: webgl.Context.Image = .{
        .target = faceTarget(res.target, box.face),
        .level = @intCast(box.level),
        .internal_format = native.internal,
        .width = @intCast(box.width),
        .height = @intCast(box.height),
        .format = native.format,
        .kind = native.kind,
    };

    var staged: ?[]u8 = null;
    defer if (staged) |bytes| self.gpa.free(bytes);
    var row_length: u32 = 0;
    if (data) |bytes| {
        const texel = res.format.bytesPerPixel();
        const plan = planUpload(native, texel, box.width, row_pitch);
        if (plan.repack) {
            staged = try repack(self.gpa, bytes, texel, box.width, box.height, row_pitch, native.swap_rb);
            image.pixels = staged;
        } else {
            image.pixels = bytes;
            row_length = plan.row_length;
        }
    }

    if (row_length != 0) gl.pixelStorei(c.unpack_row_length, @intCast(row_length));
    defer if (row_length != 0) gl.pixelStorei(c.unpack_row_length, 0);
    switch (how) {
        .allocate => gl.texImage2D(image),
        .replace => gl.texSubImage2D(@intCast(box.x), @intCast(box.y), image),
    }
}

/// `rows` rows of `width` texels, `row_pitch` bytes apart, as a tightly packed
/// copy - with red and blue changed places in every texel if `swap_rb`. The
/// copy is the caller's.
fn repack(gpa: Allocator, bytes: []const u8, texel_bytes: usize, width: u32, rows: u32, row_pitch: usize, swap_rb: bool) Allocator.Error![]u8 {
    const row = @as(usize, width) * texel_bytes;
    const out = try gpa.alloc(u8, row * rows);
    for (0..rows) |y| {
        const to = out[y * row ..][0..row];
        @memcpy(to, bytes[y * row_pitch ..][0..row]);
        if (swap_rb) {
            var x: usize = 0;
            while (x < row) : (x += texel_bytes) std.mem.swap(u8, &to[x], &to[x + 2]);
        }
    }
    return out;
}

// -------------------------------------------------------------------------
// The scratch framebuffer
// -------------------------------------------------------------------------

/// The framebuffer that passes and readbacks attach to, made when first needed.
fn passFramebuffer(self: *WebGl) Error!webgl.Framebuffer {
    if (self.fbo == .none) self.fbo = self.gl.createFramebuffer() catch return error.OutOfMemory;
    return self.fbo;
}

/// Attach one image of a texture to the framebuffer bound now, at the point
/// its format goes to.
fn attach(self: *WebGl, res: *const TextureRes, layer: u32, mip: u32) void {
    self.gl.framebufferTexture2D(c.framebuffer, attachmentPoint(res.format), faceTarget(res.target, layer), res.texture, @intCast(mip));
}

/// Empty the framebuffer of whatever a pass or a read put on it, and leave
/// the canvas bound. The point a depth format went to is remembered, because
/// a depth-only format and one with stencil are two different points and one
/// left behind would make the next pass incomplete.
fn detachPass(self: *WebGl) void {
    if (!self.attached.color and self.attached.depth == null) return;
    const gl = self.gl;
    gl.bindFramebuffer(c.framebuffer, self.fbo);
    if (self.attached.color) gl.framebufferTexture2D(c.framebuffer, c.color_attachment0, c.texture_2d, .none, 0);
    if (self.attached.depth) |point| gl.framebufferTexture2D(c.framebuffer, point, c.texture_2d, .none, 0);
    gl.bindFramebuffer(c.framebuffer, .none);
    self.attached = .{};
}

fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const gl = self.gl;

    const size = levelExtent(res, sub.mip);
    const row = @as(usize, size[0]) * 4;
    const pixels = try gpa.alloc(u8, row * size[1]);
    errdefer gpa.free(pixels);

    detachPass(self);
    gl.bindFramebuffer(c.framebuffer, try passFramebuffer(self));
    attach(self, res, sub.layer, sub.mip);
    self.attached.color = true;
    defer detachPass(self);
    // A format that cannot be drawn into cannot be attached, and there is no
    // way to read a texture that is not attached.
    gl.checkFramebuffer(c.framebuffer) catch return error.Unsupported;

    const width: i32 = @intCast(size[0]);
    const height: i32 = @intCast(size[1]);
    if (res.format.info().kind == .float) {
        // A floating-point attachment is read as floats and nothing else, and
        // is brought to eight bits here. Zeroed first: a read that is refused
        // leaves it as it was.
        const floats = try gpa.alloc(f32, pixels.len);
        defer gpa.free(floats);
        @memset(floats, 0);
        gl.readPixels(0, 0, width, height, c.rgba, c.float, std.mem.sliceAsBytes(floats));
        for (pixels, floats) |*out, value| out.* = unitToByte(value);
    } else {
        gl.readPixels(0, 0, width, height, c.rgba, c.unsigned_byte, pixels);
    }

    // What came back for a one-channel format is red, with green and blue
    // nought and alpha one; the contract is a grey.
    if (res.format.info().channels == 1) greyFromRed(pixels);
    // GL gives a drawn image bottom row first, and the contract is top row
    // first. An image that was only ever uploaded is already in the order it
    // was given, which is top row first too, and turning it over would hand a
    // caller its own picture upside down.
    if (isDrawn(res, sub.layer, sub.mip)) flipRows(pixels, row);
    return pixels;
}

/// A colour channel as eight bits: clamped to nought to one, and NaN is nought.
fn unitToByte(value: f32) u8 {
    const clamped: f32 = if (value >= 1) 1 else if (value > 0) value else 0;
    return @intFromFloat(@round(clamped * 255));
}

/// Copy red into green and blue in every texel of an RGBA image.
fn greyFromRed(pixels: []u8) void {
    var at: usize = 0;
    while (at + 4 <= pixels.len) : (at += 4) {
        pixels[at + 1] = pixels[at];
        pixels[at + 2] = pixels[at];
    }
}

fn flipRows(pixels: []u8, row: usize) void {
    var top: usize = 0;
    var bottom: usize = pixels.len / row;
    while (top + 1 < bottom) : (top += 1) {
        bottom -= 1;
        const a = pixels[top * row ..][0..row];
        const b = pixels[bottom * row ..][0..row];
        for (a, b) |*x, *y| std.mem.swap(u8, x, y);
    }
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

/// The minification filter for a filter and a mip filter: the first word of
/// the token is how a level is read, the second how two levels are joined.
const min_filters: std.EnumArray(types.Filter, std.EnumArray(types.MipFilter, Enum)) = .init(.{
    .nearest = .init(.{ .none = c.nearest, .nearest = c.nearest_mipmap_nearest, .linear = c.nearest_mipmap_linear }),
    .linear = .init(.{ .none = c.linear, .nearest = c.linear_mipmap_nearest, .linear = c.linear_mipmap_linear }),
});

/// Wrap tokens. `border` is desktop GL's: ES 3.0 has no border colour, and
/// `Features.sampler_border` says so, so it is a null here and never asked.
const wraps: std.EnumArray(types.Wrap, ?Enum) = .init(.{
    .repeat = c.repeat,
    .clamp_to_edge = c.clamp_to_edge,
    .mirror = c.mirrored_repeat,
    .border = null,
});

/// The widest level range a sampler can state: ES 3.0's initial
/// `TEXTURE_MIN_LOD` and `TEXTURE_MAX_LOD` are minus and plus this, and it is
/// what `SamplerDesc.lod_max` defaults to.
const lod_limit = 1000;

/// A level bound as the integer the wire carries (`wire.fractional_lod`). A
/// fraction is widened to the whole level around it - the lower bound rounds
/// down and the upper one up - so the range is never smaller than asked.
fn lodParam(value: f32, side: enum { min, max }) i32 {
    const whole = switch (side) {
        .min => @floor(value),
        .max => @ceil(value),
    };
    return @intFromFloat(std.math.clamp(whole, -lod_limit, lod_limit));
}

const SamplerParam = struct { pname: Enum, value: i32 };

/// Two filters, three wraps, two level bounds, a comparison's mode and
/// function, anisotropy: every parameter a sampler is made of.
const max_sampler_params = 10;

/// The parameters of one sampler, in the order they are set: data, so that
/// what a description turns into can be read without a context.
const SamplerParams = struct {
    items: [max_sampler_params]SamplerParam = undefined,
    len: usize = 0,

    fn add(self: *SamplerParams, pname: Enum, value: i32) void {
        self.items[self.len] = .{ .pname = pname, .value = value };
        self.len += 1;
    }

    fn addEnum(self: *SamplerParams, pname: Enum, value: Enum) void {
        self.add(pname, @intCast(value));
    }

    fn slice(self: *const SamplerParams) []const SamplerParam {
        return self.items[0..self.len];
    }

    /// The value set for a parameter, or null if it was not set.
    fn find(self: *const SamplerParams, pname: Enum) ?i32 {
        for (self.slice()) |param| if (param.pname == pname) return param.value;
        return null;
    }
};

/// What a description is as sampler state. `max_anisotropy` is the device's
/// limit; the description's own has been clamped to it already.
fn samplerParams(desc: types.SamplerDesc, max_anisotropy: u32) Error!SamplerParams {
    var out: SamplerParams = .{};
    out.addEnum(c.texture_min_filter, min_filters.get(desc.min_filter).get(desc.mip_filter));
    out.addEnum(c.texture_mag_filter, filterEnum(desc.mag_filter));
    out.addEnum(c.texture_wrap_s, wraps.get(desc.wrap_u) orelse return error.Unsupported);
    out.addEnum(c.texture_wrap_t, wraps.get(desc.wrap_v) orelse return error.Unsupported);
    out.addEnum(c.texture_wrap_r, wraps.get(desc.wrap_w) orelse return error.Unsupported);
    out.add(more.texture_min_lod, lodParam(desc.lod_min, .min));
    out.add(more.texture_max_lod, lodParam(desc.lod_max, .max));
    if (desc.compare) |func| {
        out.addEnum(more.texture_compare_mode, more.compare_ref_to_texture);
        out.addEnum(more.texture_compare_func, compare(func));
    }
    if (desc.max_anisotropy > 1 and max_anisotropy > 1) {
        out.add(more.texture_max_anisotropy, @intCast(@min(desc.max_anisotropy, max_anisotropy)));
    }
    return out;
}

fn makeSampler(self: *WebGl, desc: types.SamplerDesc) Error!webgl.Sampler {
    const params = try samplerParams(desc, self.caps.limits.max_anisotropy);
    const sampler = self.gl.createSampler() catch return error.OutOfMemory;
    for (params.slice()) |param| self.gl.samplerParameteri(sampler, param.pname, param.value);
    return sampler;
}

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);

    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    res.* = .{ .sampler = try makeSampler(self, desc), .desc = desc };
    return res;
}

/// The sampler with every filter nearest: what a depth texture is read with
/// when nobody asked it to compare, and what a 32-bit float one is read with
/// always.
fn unfilteredDesc(desc: types.SamplerDesc) types.SamplerDesc {
    var out = desc;
    out.min_filter = .nearest;
    out.mag_filter = .nearest;
    if (out.mip_filter == .linear) out.mip_filter = .nearest;
    out.max_anisotropy = 1;
    return out;
}

/// Whether a sampler asks for a filter that this texture, read as it is, would
/// make incomplete. WebGL samples such a texture as black rather than
/// filtering it less: a linear read of a depth texture (unless the sampler
/// compares) and of a 32-bit float one (without an extension no query can
/// find) both are. `caps` says `filterable` is false for them, and Direct3D
/// would have read them anyway, so here a sampler that would is given a
/// nearest twin instead.
fn needsUnfiltered(desc: types.SamplerDesc, filterable: bool) bool {
    if (filterable or desc.compare != null) return false;
    return desc.min_filter == .linear or desc.mag_filter == .linear or desc.mip_filter == .linear;
}

/// The sampler object to bind for this pair: the sampler's own, or its nearest
/// twin, made the first time it is needed.
fn samplerFor(self: *WebGl, res: *SamplerRes, texture: *const TextureRes) Error!webgl.Sampler {
    if (!needsUnfiltered(res.desc, texture.filterable)) return res.sampler;
    if (res.unfiltered == .none) res.unfiltered = try makeSampler(self, unfilteredDesc(res.desc));
    return res.unfiltered;
}

fn filterEnum(filter: types.Filter) Enum {
    return switch (filter) {
        .nearest => c.nearest,
        .linear => c.linear,
    };
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    self.gl.deleteSampler(res.sampler);
    if (res.unfiltered != .none) self.gl.deleteSampler(res.unfiltered);
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
    res.destroyed = true;
    if (res.pipelines == 0) releaseShader(self, res);
}

fn releaseShader(self: *WebGl, res: *ShaderRes) void {
    self.gl.deleteProgram(res.program);
    self.gpa.destroy(res);
}

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const gl = self.gl;
    const shader_res = as(ShaderRes, shader);
    const program = shader_res.program;

    if (desc.buffers.len > max_vertex_slots) {
        log.print("fluxion-rhi: the WebGL backend binds at most {d} vertex buffers", .{max_vertex_slots}) catch {};
        return error.PipelineFailed;
    }

    // What it draws into must be something a pass can be: as many colours as
    // the device writes, at a sample count its format has.
    const colors = @as(u32, if (desc.color_format != null) 1 else 0) + @as(u32, @intCast(desc.extra_color_formats.len));
    if (colors > self.caps.limits.max_color_attachments) {
        log.print("fluxion-rhi: {d} colour attachments, and the device writes {d}", .{ colors, self.caps.limits.max_color_attachments }) catch {};
        return error.PipelineFailed;
    }
    if (desc.color_format orelse desc.depth_format) |format| {
        if (!self.caps.formatSupport(format).supportsSamples(desc.samples)) {
            log.print("fluxion-rhi: the device cannot draw {d} samples of {s}", .{ desc.samples, @tagName(format) }) catch {};
            return error.PipelineFailed;
        }
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

    shader_res.pipelines += 1;
    res.* = .{
        .shader = shader_res,
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
        .writes_color = desc.color_format != null,
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
    const shader = res.shader;
    self.gpa.destroy(res);
    shader.pipelines -= 1;
    if (shader.destroyed and shader.pipelines == 0) releaseShader(self, shader);
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

fn present(impl: backend.Impl, native: backend.Native, mode: types.PresentMode) Error!void {
    // Nothing to do. The browser shows the drawing buffer when the frame
    // callback that drew into it returns, at the display's own rate - which
    // is the only vsync a page has.
    _ = impl;
    _ = native;
    _ = mode;
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
                // The textures come off the framebuffer, and the canvas is
                // what is bound again.
                detachPass(self);
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
                const texture = textureOf(device, b.texture);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                gl.activeTexture(c.textureUnit(b.slot));
                gl.bindTexture(texture.target, texture.texture);
                gl.bindSampler(b.slot, try samplerFor(self, sampler, texture));
            },
            .draw => |d| {
                const pipeline = self.pipeline orelse return error.InvalidArgument;
                try flushBindings(self, pipeline, 0);
                gl.drawArraysInstanced(pipeline.topology, @intCast(d.first_vertex), @intCast(d.vertex_count), @intCast(d.instance_count));
            },
            .generate_mips => |h| {
                // Every face at once, for a cube: they are one texture.
                const texture = textureOf(device, h);
                gl.bindTexture(texture.target, texture.texture);
                gl.generateMipmap(texture.target);
                inheritDrawn(texture);
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

    // What the caps left out, for a caller that did not ask: one colour
    // attachment, and no multisampled target to resolve.
    if (pass.extra_colors.len != 0) return error.Unsupported;
    if (pass.color) |color| if (color.resolve != null) return error.Unsupported;

    // Whatever an earlier pass left on the framebuffer, if it never ended.
    detachPass(self);

    var size: [2]u32 = .{ 0, 0 };
    const to_surface = if (pass.color) |color| color.target == .surface else false;
    if (to_surface) {
        // The canvas has a depth buffer of its own, and it is what `depth`
        // means here: a texture cannot be attached to the default framebuffer,
        // so the depth texture next to a surface is not drawn into, and only
        // its clear values are used.
        gl.bindFramebuffer(c.framebuffer, .none);
        size = canvasSize();
    } else {
        gl.bindFramebuffer(c.framebuffer, try passFramebuffer(self));
        // What the pass draws into sits on the framebuffer for this pass and
        // comes off it at `end_pass`, so two passes into the same texture
        // with different depth buffers do not see each other's.
        if (pass.color) |color| {
            const res = textureOf(device, color.target.texture);
            attach(self, res, color.layer, color.mip_level);
            markDrawn(res, color.layer, color.mip_level);
            self.attached.color = true;
            size = levelExtent(res, color.mip_level);
        }
        if (pass.depth) |depth| {
            const res = textureOf(device, depth.texture);
            attach(self, res, depth.layer, depth.mip_level);
            self.attached.depth = attachmentPoint(res.format);
            // A depth-only pass is as big as its depth image.
            if (pass.color == null) size = levelExtent(res, depth.mip_level);
        }
        gl.checkFramebuffer(c.framebuffer) catch return error.Failed;
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
    if (pass.color) |color| if (color.load == .clear) {
        gl.colorMask(true, true, true, true);
        gl.clearColor(color.clear_color[0], color.clear_color[1], color.clear_color[2], color.clear_color[3]);
        mask |= c.color_buffer_bit;
    };
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
    // A pipeline with no colour format is for a pass with no colour: nothing
    // it draws may reach one that is there.
    gl.colorMask(res.writes_color, res.writes_color, res.writes_color, res.writes_color);

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

// -------------------------------------------------------------------------
// Tests - the format table and the caps
// -------------------------------------------------------------------------

/// What a client layout's token says about the channels it carries.
const layout_channels = [_]struct { Enum, u8 }{
    .{ c.red, 1 },
    .{ c.rg, 2 },
    .{ c.rgb, 3 },
    .{ c.rgba, 4 },
    .{ c.depth_component, 1 },
    .{ c.depth_stencil, 2 },
};

fn channelsOfLayout(layout: Enum) ?u8 {
    for (layout_channels) |row| if (row[0] == layout) return row[1];
    return null;
}

/// A device on the stub, and what it asked the stub while it opened.
fn openedCaps() types.Caps {
    stub.reset();
    return computeCaps(.init());
}

fn solid(comptime texels: usize, colour: [4]u8) [texels * 4]u8 {
    var out: [texels * 4]u8 = undefined;
    for (0..texels) |i| out[i * 4 ..][0..4].* = colour;
    return out;
}

test "every format has a row, and the row agrees with what the format says of itself" {
    var rows: usize = 0;
    for (std.enums.values(types.Format)) |format| {
        // A null is a format WebGL cannot do at all; none is, so far.
        const native = natives.get(format) orelse continue;
        rows += 1;
        const row = format.info();
        try testing.expect(native.internal != 0);

        if (format.isCompressed()) {
            // Blocks go over as bytes, with no client layout, and are in an
            // extension somebody has to switch on.
            try testing.expectEqual(@as(Enum, 0), native.format);
            try testing.expectEqual(@as(Enum, 0), native.kind);
            try testing.expect(native.extension != null);
            try testing.expect(!native.swap_rb);
            continue;
        }

        try testing.expect(native.extension == null);
        // The layout carries the channels the format has.
        try testing.expectEqual(@as(?u8, row.channels), channelsOfLayout(native.format));
        // Depth is depth, both ways round.
        try testing.expectEqual(row.depth, native.format == c.depth_component or native.format == c.depth_stencil);
        try testing.expectEqual(row.stencil, native.format == c.depth_stencil);
        // A swap is for the one layout WebGL has no upload for.
        if (native.swap_rb) try testing.expectEqual(@as(u8, 4), row.channels);
        try testing.expectEqual(format == .bgra8_unorm or format == .bgra8_unorm_srgb, native.swap_rb);
        // sRGB is a matter of the internal format.
        try testing.expectEqual(row.srgb, native.internal == c.srgb8_alpha8);
        // What cannot be filtered or blended is depth or a 32-bit channel:
        // the two things that need an extension nothing here can see.
        if (!native.filter) try testing.expect(row.depth or (row.kind == .float and row.block_bytes >= 4 * row.channels));
        if (!native.blend) try testing.expect(row.depth or (row.kind == .float and row.block_bytes >= 4 * row.channels));
    }
    try testing.expectEqual(types.Format.count, rows);
}

test "the table has a row for every format and a name for every extension" {
    // The compile-time check is the real one; this is what it is checking.
    inline for (std.meta.fields(types.Format)) |field| {
        const format: types.Format = @enumFromInt(field.value);
        var found = false;
        for (native_rows) |row| found = found or row[0] == format;
        try testing.expect(found);
    }
    for (std.enums.values(types.Format)) |format| {
        if (natives.get(format)) |native| {
            if (native.extension) |name| try testing.expect(std.mem.indexOf(u8, name, "compress") != null);
        }
    }
}

/// What every format is made as, with no `texImage3D` in the binding: plain
/// images and cubes.
const flat_dimensions: std.EnumSet(types.Dimension) = blk: {
    var set: std.EnumSet(types.Dimension) = .initFull();
    set.remove(.d3);
    set.remove(.d2_array);
    break :blk set;
};

test "the caps never claim a format the table cannot do, or a flag its rules do not allow" {
    const answer = openedCaps();
    for (std.enums.values(types.Format)) |format| {
        const support = answer.formatSupport(format);
        if (natives.get(format) == null) {
            try testing.expectEqual(types.FormatSupport{}, support);
            continue;
        }
        if (support.blendable) try testing.expect(support.render_target);
        if (support.generate_mips) try testing.expect(support.render_target and support.filterable and support.sampled);
        if (support.filterable) try testing.expect(support.sampled);
        // A depth format is drawn into and read, never blended or resampled.
        if (format.isDepth()) try testing.expect(!support.blendable and !support.generate_mips);
        // No multisampling in the binding, and none for what cannot be drawn into.
        try testing.expect(support.sample_counts <= single_sample);
        if (!support.render_target) try testing.expectEqual(@as(u8, 0), support.sample_counts);
        // Nothing compressed until there is a call to upload it with.
        if (format.isCompressed()) try testing.expectEqual(types.FormatSupport{}, support);
    }
}

test "the stub's caps are what the context reports and what the binding lacks" {
    const answer = openedCaps();

    // The context's own sizes: the stub says 2048.
    try testing.expectEqual(@as(u32, 2048), answer.limits.max_texture_2d);
    try testing.expectEqual(@as(u32, 2048), answer.limits.max_texture_cube);
    // No `texImage3D`, no `drawBuffers`, and the anisotropy extension is not on.
    try testing.expectEqual(@as(u32, 0), answer.limits.max_texture_3d);
    try testing.expectEqual(@as(u32, 0), answer.limits.max_texture_layers);
    try testing.expectEqual(@as(u32, 1), answer.limits.max_color_attachments);
    try testing.expectEqual(@as(u32, 1), answer.limits.max_anisotropy);
    // ES 3.0 has neither a border colour nor a sampler bias.
    try testing.expect(!answer.features.sampler_border);
    try testing.expect(!answer.features.sampler_lod_bias);

    // The stub calls every framebuffer complete, so everything that can be
    // allocated can be drawn into.
    const rgba8 = answer.formatSupport(.rgba8_unorm);
    try testing.expectEqual(types.FormatSupport{ .sampled = true, .filterable = true, .render_target = true, .blendable = true, .generate_mips = true, .sample_counts = 1, .dimensions = flat_dimensions }, rgba8);
    try testing.expectEqual(rgba8, answer.formatSupport(.bgra8_unorm_srgb));
    // 32-bit float reads and draws, is not filtered or blended, and so has no chain.
    try testing.expectEqual(
        types.FormatSupport{ .sampled = true, .filterable = false, .render_target = true, .blendable = false, .generate_mips = false, .sample_counts = 1, .dimensions = flat_dimensions },
        answer.formatSupport(.rgba32_float),
    );
    // Half float filters and blends.
    try testing.expect(answer.formatSupport(.rgba16_float).filterable and answer.formatSupport(.rgba16_float).blendable);
    // Depth is drawn into and sampled, and not filtered.
    try testing.expectEqual(
        types.FormatSupport{ .sampled = true, .filterable = false, .render_target = true, .blendable = false, .generate_mips = false, .sample_counts = 1, .dimensions = flat_dimensions },
        answer.formatSupport(.depth32_float),
    );
}

test "a format that does not draw is not claimed to, and takes its chain and its samples with it" {
    const float = natives.get(.rgba16_float).?;
    const yes = supportFor(.rgba16_float, float, true);
    const no = supportFor(.rgba16_float, float, false);
    try testing.expect(yes.render_target and yes.blendable and yes.generate_mips);
    try testing.expectEqual(single_sample, yes.sample_counts);
    // Sampled and filtered all the same: that is the format's own.
    try testing.expectEqual(types.FormatSupport{ .sampled = true, .filterable = true }, no);

    // A compressed format is not until there is a call and an extension.
    try testing.expectEqual(types.FormatSupport{}, supportFor(.bc7_rgba_unorm, natives.get(.bc7_rgba_unorm).?, false));
}

test "what the binding lacks is what the caps leave out, until the binding grows it" {
    // Each row: a call `fluxion-webgl` would need, and the flag that says this
    // backend has it. When a call appears the flag is stale, this fails, and
    // the flag and the code behind it are changed together.
    const needs = [_]struct { []const u8, bool }{
        .{ "texImage3D", wire.layered },
        .{ "texSubImage3D", wire.layered },
        .{ "framebufferTextureLayer", wire.layered },
        .{ "renderbufferStorageMultisample", wire.multisample },
        .{ "blitFramebuffer", wire.multisample },
        .{ "getInternalformatParameter", wire.multisample },
        .{ "drawBuffers", wire.draw_buffers },
        .{ "compressedTexImage2D", wire.compressed_upload },
        .{ "compressedTexSubImage2D", wire.compressed_upload },
        .{ "getExtension", wire.extension_query },
        .{ "getSupportedExtensions", wire.extension_query },
        .{ "samplerParameterf", wire.fractional_lod },
    };
    inline for (needs) |row| try testing.expectEqual(row[1], @hasDecl(raw, row[0]));
}

test "a compressed format is written and read back if the device has one" {
    const answer = openedCaps();
    for (std.enums.values(types.Format)) |format| {
        if (!format.isCompressed() or !answer.formatSupport(format).sampled) continue;

        var device = try openDevice();
        defer device.deinit();
        const block = [_]u8{0x5a} ** 16;
        const texture = try device.createTexture(.{ .width = 4, .height = 4, .format = format, .data = block[0..format.info().block_bytes] });
        try device.writeTexture(texture, .{}, block[0..format.info().block_bytes], 0, 0);
        return;
    }
    // Not one is, and the caps say so - which the test above holds them to.
    return error.SkipZigTest;
}

// -------------------------------------------------------------------------
// Tests - textures
// -------------------------------------------------------------------------

test "the images of a texture are a level of a face, level by level within a face" {
    // A cube of 8 with four levels: six faces of 8, 4, 2, 1.
    const extents = [_]u32{ 8, 4, 2, 1 };
    for (0..24) |i| {
        const image = imageAt(8, 8, 4, @intCast(i));
        try testing.expectEqual(@as(u32, @intCast(i / 4)), image.face);
        try testing.expectEqual(@as(u32, @intCast(i % 4)), image.level);
        try testing.expectEqual(extents[i % 4], image.width);
        try testing.expectEqual(extents[i % 4], image.height);
    }
    // A wide one halves each side, and stops at one.
    const wide = imageAt(8, 2, 4, 2);
    try testing.expectEqual(@as(u32, 2), wide.width);
    try testing.expectEqual(@as(u32, 1), wide.height);
    const last = imageAt(8, 2, 4, 3);
    try testing.expectEqual(@as(u32, 1), last.width);
    try testing.expectEqual(@as(u32, 1), last.height);

    // A cube's faces are six targets on one texture, in GL's order.
    try testing.expectEqual(c.cubeFace(3), faceTarget(c.texture_cube_map, 3));
    try testing.expectEqual(c.texture_cube_map_positive_x, faceTarget(c.texture_cube_map, 0));
    try testing.expectEqual(c.texture_2d, faceTarget(c.texture_2d, 0));
    try testing.expectEqual(c.texture_cube_map, dimension_targets.get(.cube));
    try testing.expectEqual(c.texture_2d, dimension_targets.get(.d2));
}

test "how a row of bytes goes over is worked out before it does" {
    const rgba = natives.get(.rgba8_unorm).?;
    const bgra = natives.get(.bgra8_unorm_srgb).?;
    const half = natives.get(.r16_float).?;

    // Tight, and padded to a whole number of texels: a row length, or none.
    try testing.expectEqual(UploadPlan{ .row_length = 0, .repack = false }, planUpload(rgba, 4, 8, 32));
    try testing.expectEqual(UploadPlan{ .row_length = 16, .repack = false }, planUpload(rgba, 4, 8, 64));
    try testing.expectEqual(UploadPlan{ .row_length = 6, .repack = false }, planUpload(half, 2, 3, 12));
    // A pitch that is not a whole number of texels cannot be a row length.
    try testing.expectEqual(UploadPlan{ .row_length = 0, .repack = true }, planUpload(rgba, 4, 2, 10));
    // And BGRA is always gathered, since its bytes change either way.
    try testing.expectEqual(UploadPlan{ .row_length = 0, .repack = true }, planUpload(bgra, 4, 8, 32));
}

test "a repack is tight, and swaps red and blue only when it is told to" {
    // Two texels a row, a row of padding between.
    const source = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    const kept = try repack(testing.allocator, &source, 4, 2, 2, 16, false);
    defer testing.allocator.free(kept);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17 }, kept);

    const swapped = try repack(testing.allocator, &source, 4, 2, 2, 16, true);
    defer testing.allocator.free(swapped);
    try testing.expectEqualSlices(u8, &.{ 3, 2, 1, 4, 7, 6, 5, 8, 12, 11, 10, 13, 16, 15, 14, 17 }, swapped);
}

test "a texture is made with every level, and the data goes to level zero of every face" {
    var device = try openDevice();
    defer device.deinit();

    // Eight by four: 8x4, 4x2, 2x1, 1x1. The last thing allocated is the
    // last level, and it has nothing in it.
    _ = try device.createTexture(.{ .width = 8, .height = 4, .mip_levels = 0 });
    try testing.expectEqual(1, stub.state.last_image.width);
    try testing.expectEqual(1, stub.state.last_image.height);
    try testing.expectEqual(0, stub.state.last_image.len);

    // One level is one image, and it is the data.
    const one = solid(8 * 4, .{ 9, 8, 7, 6 });
    _ = try device.createTexture(.{ .width = 8, .height = 4, .data = &one });
    try testing.expectEqual(8, stub.state.last_image.width);
    try testing.expectEqual(one.len, stub.state.last_image.len);
    try testing.expectEqual([4]u8{ 9, 8, 7, 6 }, stub.state.last_image.first);

    // Six faces of four by four in one buffer: the last thing uploaded is the
    // sixth, so its colour is what the face offset found.
    var faces: [6 * 4 * 4 * 4]u8 = undefined;
    for (0..6) |face| faces[face * 64 ..][0..64].* = solid(16, .{ @intCast(face + 1), 0, 0, 255 });
    _ = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &faces });
    try testing.expectEqual([4]u8{ 6, 0, 0, 255 }, stub.state.last_image.first);

    // The same with a padded row: faces are as many rows as they have, at the
    // pitch given, and a pitch of a whole number of texels goes over as it is.
    var padded: [6 * 4 * 32]u8 = @splat(0);
    for (0..6) |face| {
        for (0..4) |row| padded[face * 128 + row * 32 ..][0..16].* = solid(4, .{ @intCast(face + 1), 1, 1, 255 });
    }
    _ = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &padded, .row_pitch = 32 });
    try testing.expectEqual([4]u8{ 6, 1, 1, 255 }, stub.state.last_image.first);
}

test "each mip level is written at its own size" {
    var device = try openDevice();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 4, .mip_levels = 0, .usage = .{ .render_target = true } });
    const sizes = [_][2]u32{ .{ 8, 4 }, .{ 4, 2 }, .{ 2, 1 }, .{ 1, 1 } };
    for (sizes, 0..) |size, level| {
        const colour: [4]u8 = .{ @intCast(level * 16), 2, 3, 255 };
        var bytes: [8 * 4 * 4]u8 = undefined;
        const texels = size[0] * size[1];
        for (0..texels) |i| bytes[i * 4 ..][0..4].* = colour;
        try device.writeTexture(texture, .{ .mip = @intCast(level) }, bytes[0 .. texels * 4], 0, 0);

        try testing.expectEqual(@as(i32, @intCast(size[0])), stub.state.last_image.width);
        try testing.expectEqual(@as(i32, @intCast(size[1])), stub.state.last_image.height);
        try testing.expectEqual(texels * 4, stub.state.last_image.len);
        try testing.expectEqual(colour, stub.state.last_image.first);

        // And read at the size of the level, top row first.
        const pixels = try device.readSubresource(texture, .{ .mip = @intCast(level) }, testing.allocator);
        defer testing.allocator.free(pixels);
        try testing.expectEqual(texels * 4, pixels.len);
    }
}

test "a corner of a level is written where it says, with the row it was given" {
    var device = try openDevice();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0 });
    // Two by two at (2, 2) of level one, from rows of eight bytes padded to sixteen.
    var bytes: [16 + 8]u8 = @splat(0);
    bytes[0..8].* = solid(2, .{ 1, 2, 3, 4 });
    bytes[16..24].* = solid(2, .{ 1, 2, 3, 4 });
    try device.writeTexture(texture, .{ .mip = 1, .x = 2, .y = 2, .width = 2, .height = 2 }, &bytes, 16, 0);
    try testing.expectEqual(2, stub.state.last_image.width);
    try testing.expectEqual(2, stub.state.last_image.height);
    try testing.expectEqual([4]u8{ 1, 2, 3, 4 }, stub.state.last_image.first);
}

test "a cube is written and read a face at a time, or all six" {
    var device = try openDevice();
    defer device.deinit();

    const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .mip_levels = 0, .usage = .{ .render_target = true } });
    // A face at a time: what reaches the driver is that face's own colour.
    for (0..6) |face| {
        const colour: [4]u8 = .{ @intCast(face * 40), 1, 2, 255 };
        const bytes = solid(16, colour);
        try device.writeTexture(cube, .{ .z = @intCast(face), .depth = 1 }, &bytes, 0, 0);
        try testing.expectEqual(colour, stub.state.last_image.first);
        try testing.expectEqual(4, stub.state.last_image.width);
    }
    // Faces two to four, in one call: the last is the fourth, so a slice
    // pitch is what found it.
    var run: [3 * 64]u8 = undefined;
    for (0..3) |i| run[i * 64 ..][0..64].* = solid(16, .{ @intCast(i + 100), 0, 0, 255 });
    try device.writeTexture(cube, .{ .z = 2, .depth = 3 }, &run, 0, 0);
    try testing.expectEqual([4]u8{ 102, 0, 0, 255 }, stub.state.last_image.first);
    // And with faces further apart than their own size.
    var spread: [2 * 128]u8 = @splat(0);
    spread[0..64].* = solid(16, .{ 7, 0, 0, 255 });
    spread[128..192].* = solid(16, .{ 8, 0, 0, 255 });
    try device.writeTexture(cube, .{ .z = 4, .depth = 2 }, &spread, 0, 128);
    try testing.expectEqual([4]u8{ 8, 0, 0, 255 }, stub.state.last_image.first);
    // A face of a level below the first is that level's size.
    const small = solid(1, .{ 1, 2, 3, 4 });
    try device.writeTexture(cube, .{ .mip = 2, .z = 5, .depth = 1 }, &small, 0, 0);
    try testing.expectEqual(1, stub.state.last_image.width);

    // Read face by face at each level's size.
    for (0..6) |face| {
        const pixels = try device.readSubresource(cube, .{ .layer = @intCast(face) }, testing.allocator);
        testing.allocator.free(pixels);
    }
    const level2 = try device.readSubresource(cube, .{ .mip = 2, .layer = 3 }, testing.allocator);
    defer testing.allocator.free(level2);
    try testing.expectEqual(@as(usize, 4), level2.len);
}

test "generating mips is a bind and a call, on the texture's own target" {
    var device = try openDevice();
    defer device.deinit();

    const flat = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .render_target = true } });
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .render_target = true } });
    for ([_]types.Texture{ flat, cube }) |texture| {
        const before = stub.state.calls;
        const cmd = device.begin();
        try cmd.generateMips(texture);
        try device.submit();
        try testing.expectEqual(before + 2, stub.state.calls);
    }
    // Not for what cannot be filtered: a 32-bit float chain is a chain of
    // guesses at a filter the device may not have.
    const float = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .format = .rgba32_float, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.generateMips(float);
    try testing.expectError(error.Unsupported, device.submit());
}

test "what the binding cannot do is refused before it is tried, and the reason is named" {
    var device = try openDevice();
    defer device.deinit();

    // No `texImage3D`: no volume, no array.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4 }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "shape") != null);
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = 2 }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "shape") != null);

    // No `renderbufferStorageMultisample`: nothing to resolve.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .samples = 4, .usage = .{ .sampled = false, .render_target = true } }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "multisample") != null);

    // No compressed upload.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .format = .bc1_rgba_unorm }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 8, .height = 8, .format = .astc_4x4_unorm }));

    // No `drawBuffers`: one colour attachment.
    const a = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const b = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = a } }, .extra_colors = &.{.{ .target = .{ .texture = b } }} });
    try cmd.endPass();
    try testing.expectError(error.InvalidArgument, device.submit());
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "max_color_attachments") != null);

    // ES 3.0 has no border and no bias.
    try testing.expectError(error.Unsupported, device.createSampler(.{ .wrap_v = .border }));
    try testing.expectError(error.Unsupported, device.createSampler(.{ .lod_bias = 1 }));
}

test "a BGRA target is drawn as it is and read as it is, and only an upload swaps" {
    var device = try openDevice();
    defer device.deinit();

    // The sRGB one goes in swapped like the other.
    _ = try device.createTexture(.{ .width = 1, .height = 1, .format = .bgra8_unorm_srgb, .data = &.{ 10, 20, 30, 40 } });
    try testing.expectEqual(c.rgba, stub.state.last_image.format);
    try testing.expectEqual(@as(i32, @intCast(c.srgb8_alpha8)), stub.state.last_image.internal_format);
    try testing.expectEqual([4]u8{ 30, 20, 10, 40 }, stub.state.last_image.first);

    // Drawn red into a BGRA target, it is red when read: the storage is the
    // colour, so there is nothing to swap on the way out.
    const target = try device.createTexture(.{ .width = 2, .height = 2, .format = .bgra8_unorm, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 1, 0, 0, 1 } } });
    try cmd.endPass();
    try device.submit();
    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, pixels[0..4]);

    // A padded BGRA upload is swapped row by row from the right place.
    const padded = [_]u8{ 1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8 };
    const two = try device.createTexture(.{ .width = 1, .height = 2, .format = .bgra8_unorm });
    try device.writeTexture(two, .{ .y = 1, .height = 1 }, padded[8..12], 4, 0);
    try testing.expectEqual([4]u8{ 7, 6, 5, 8 }, stub.state.last_image.first);
    try device.writeTexture(two, .{}, &padded, 8, 0);
    try testing.expectEqual(@as(i32, 2), stub.state.last_image.height);
}

test "a one-channel read is grey, and a float read is brought to eight bits" {
    var device = try openDevice();
    defer device.deinit();

    const grey = try device.createTexture(.{ .width = 2, .height = 1, .format = .r8_unorm, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = grey }, .clear_color = .{ 0.5, 0, 0, 1 } } });
    try cmd.endPass();
    try device.submit();
    const pixels = try device.readTexture(grey, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqualSlices(u8, &.{ 128, 128, 128, 255, 128, 128, 128, 255 }, pixels);

    // The stub answers a float read with bytes it has no business writing
    // there; what matters is that whatever comes back is a byte, of the right count.
    const half = try device.createTexture(.{ .width = 3, .height = 2, .format = .rgba16_float, .usage = .{ .render_target = true } });
    const floats = try device.readTexture(half, testing.allocator);
    defer testing.allocator.free(floats);
    try testing.expectEqual(@as(usize, 3 * 2 * 4), floats.len);

    try testing.expectEqual(@as(u8, 0), unitToByte(std.math.nan(f32)));
    try testing.expectEqual(@as(u8, 0), unitToByte(-3));
    try testing.expectEqual(@as(u8, 255), unitToByte(7));
    try testing.expectEqual(@as(u8, 255), unitToByte(std.math.inf(f32)));
    try testing.expectEqual(@as(u8, 128), unitToByte(0.5));
    try testing.expectEqual(@as(u8, 0), unitToByte(0));
}

test "rows turn over top for bottom, whatever their number" {
    var one = [_]u8{ 1, 2 };
    flipRows(&one, 2);
    try testing.expectEqualSlices(u8, &.{ 1, 2 }, &one);
    var two = [_]u8{ 1, 2, 3, 4 };
    flipRows(&two, 2);
    try testing.expectEqualSlices(u8, &.{ 3, 4, 1, 2 }, &two);
    var three = [_]u8{ 1, 2, 3, 4, 5, 6 };
    flipRows(&three, 2);
    try testing.expectEqualSlices(u8, &.{ 5, 6, 3, 4, 1, 2 }, &three);
    var four = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    flipRows(&four, 2);
    try testing.expectEqualSlices(u8, &.{ 7, 8, 5, 6, 3, 4, 1, 2 }, &four);
}

// -------------------------------------------------------------------------
// Tests - samplers
// -------------------------------------------------------------------------

test "a sampler is the state its description turns into, and nothing in it is a guess" {
    // The plain one: filters, three clamps, the whole level range, no comparison.
    const plain = try samplerParams(.linear, 1);
    try testing.expectEqual(@as(?i32, @intCast(c.linear)), plain.find(c.texture_min_filter));
    try testing.expectEqual(@as(?i32, @intCast(c.linear)), plain.find(c.texture_mag_filter));
    for ([_]Enum{ c.texture_wrap_s, c.texture_wrap_t, c.texture_wrap_r }) |axis| {
        try testing.expectEqual(@as(?i32, @intCast(c.clamp_to_edge)), plain.find(axis));
    }
    try testing.expectEqual(@as(?i32, 0), plain.find(more.texture_min_lod));
    try testing.expectEqual(@as(?i32, 1000), plain.find(more.texture_max_lod));
    try testing.expectEqual(@as(?i32, null), plain.find(more.texture_compare_mode));
    try testing.expectEqual(@as(?i32, null), plain.find(more.texture_max_anisotropy));

    // The mip filter picks the minification token, and the wrap in the third
    // axis is its own.
    const tri = try samplerParams(.{ .mip_filter = .linear, .wrap_u = .repeat, .wrap_v = .mirror, .wrap_w = .repeat }, 1);
    try testing.expectEqual(@as(?i32, @intCast(c.linear_mipmap_linear)), tri.find(c.texture_min_filter));
    try testing.expectEqual(@as(?i32, @intCast(c.repeat)), tri.find(c.texture_wrap_s));
    try testing.expectEqual(@as(?i32, @intCast(c.mirrored_repeat)), tri.find(c.texture_wrap_t));
    try testing.expectEqual(@as(?i32, @intCast(c.repeat)), tri.find(c.texture_wrap_r));
    const each = [_]struct { types.Filter, types.MipFilter, Enum }{
        .{ .nearest, .none, c.nearest },
        .{ .nearest, .nearest, c.nearest_mipmap_nearest },
        .{ .nearest, .linear, c.nearest_mipmap_linear },
        .{ .linear, .none, c.linear },
        .{ .linear, .nearest, c.linear_mipmap_nearest },
        .{ .linear, .linear, c.linear_mipmap_linear },
    };
    for (each) |row| {
        const p = try samplerParams(.{ .min_filter = row[0], .mip_filter = row[1] }, 1);
        try testing.expectEqual(@as(?i32, @intCast(row[2])), p.find(c.texture_min_filter));
    }

    // A shadow sampler compares, with the function it was given.
    const shadow = try samplerParams(.{ .compare = .less_equal }, 1);
    try testing.expectEqual(@as(?i32, @intCast(more.compare_ref_to_texture)), shadow.find(more.texture_compare_mode));
    try testing.expectEqual(@as(?i32, @intCast(c.lequal)), shadow.find(more.texture_compare_func));

    // Anisotropy is set when there is some to set, and never above what the
    // device has.
    try testing.expectEqual(@as(?i32, null), (try samplerParams(.{ .max_anisotropy = 8 }, 1)).find(more.texture_max_anisotropy));
    try testing.expectEqual(@as(?i32, null), (try samplerParams(.{ .max_anisotropy = 1 }, 16)).find(more.texture_max_anisotropy));
    try testing.expectEqual(@as(?i32, 4), (try samplerParams(.{ .max_anisotropy = 8 }, 4)).find(more.texture_max_anisotropy));

    // Every parameter fits the list that holds them.
    const all = try samplerParams(.{ .compare = .always, .max_anisotropy = 16 }, 16);
    try testing.expect(all.len <= max_sampler_params);

    // There is no border in ES 3.0.
    try testing.expectError(error.Unsupported, samplerParams(.{ .wrap_u = .border }, 1));
    try testing.expectError(error.Unsupported, samplerParams(.{ .wrap_w = .border }, 1));
}

test "a level range goes over as whole levels, and is never narrower than asked" {
    try testing.expectEqual(@as(i32, 0), lodParam(0, .min));
    try testing.expectEqual(@as(i32, 1), lodParam(1, .min));
    try testing.expectEqual(@as(i32, 0), lodParam(0.5, .min));
    try testing.expectEqual(@as(i32, 3), lodParam(2.25, .max));
    try testing.expectEqual(@as(i32, 2), lodParam(2, .max));
    try testing.expectEqual(@as(i32, -1), lodParam(-0.5, .min));
    // Bounded by what ES 3.0 will take, and finite whatever it is given.
    try testing.expectEqual(@as(i32, 1000), lodParam(1e9, .max));
    try testing.expectEqual(@as(i32, 1000), lodParam(std.math.inf(f32), .max));
    try testing.expectEqual(@as(i32, -1000), lodParam(-1e9, .min));
    _ = lodParam(std.math.nan(f32), .min);
}

test "a sampler made from a description is given back with its twin" {
    var device = try openDevice();
    defer device.deinit();

    const linear = try device.createSampler(.linear);
    const comparing = try device.createSampler(.{ .compare = .less });
    const nearest = try device.createSampler(.nearest);
    const shadow_map = try device.createTexture(.{ .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    const colour = try device.createTexture(.{ .width = 8, .height = 8 });
    const float = try device.createTexture(.{ .width = 8, .height = 8, .format = .rgba32_float });
    const surface = try device.createSurface(.{});

    const Case = struct { texture: types.Texture, sampler: types.Sampler, twin: bool };
    const cases = [_]Case{
        // Depth read with a filter nobody said was for comparing: WebGL would
        // sample black, so it is read nearest.
        .{ .texture = shadow_map, .sampler = linear, .twin = true },
        // Compared, it filters, and that is what a shadow map is.
        .{ .texture = shadow_map, .sampler = comparing, .twin = false },
        .{ .texture = shadow_map, .sampler = nearest, .twin = false },
        // 32-bit float is the same story.
        .{ .texture = float, .sampler = linear, .twin = true },
        // And what filters is read as asked.
        .{ .texture = colour, .sampler = linear, .twin = false },
    };
    for (cases) |case| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
        try cmd.setTexture(3, case.texture, case.sampler);
        try cmd.endPass();
        try device.submit();

        const own = samplerName(&device, case.sampler);
        try testing.expect(stub.state.samplers[3] != 0);
        try testing.expectEqual(case.twin, stub.state.samplers[3] != own);
    }
    // The twin is made once, however often it is asked for.
    const twin = as(SamplerRes, device.samplers.get(linear).?.native).unfiltered;
    try testing.expect(twin != .none);
    const before = stub.state.live_objects;
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
    try cmd.setTexture(0, shadow_map, linear);
    try cmd.setTexture(1, shadow_map, linear);
    try cmd.endPass();
    try device.submit();
    try testing.expectEqual(before, stub.state.live_objects);
    try testing.expectEqual(twin.index(), stub.state.samplers[1]);

    // Which is what the filters became.
    const params = try samplerParams(unfilteredDesc(.trilinear), 1);
    try testing.expectEqual(@as(?i32, @intCast(c.nearest)), params.find(c.texture_mag_filter));
    try testing.expectEqual(@as(?i32, @intCast(c.nearest_mipmap_nearest)), params.find(c.texture_min_filter));
    try testing.expect(!needsUnfiltered(.trilinear, true));
    try testing.expect(needsUnfiltered(.trilinear, false));
    try testing.expect(!needsUnfiltered(.nearest, false));
}

// -------------------------------------------------------------------------
// Tests - passes
// -------------------------------------------------------------------------

test "a pass into a level or a face is as big as that image, and its rectangles count from its top" {
    var device = try openDevice();
    defer device.deinit();

    const chain = try device.createTexture(.{ .width = 16, .height = 16, .mip_levels = 0, .usage = .{ .render_target = true } });
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .render_target = true } });

    // Level two of sixteen is four, and the flip is about four.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = chain }, .mip_level = 2 } });
        try cmd.setViewport(.{ .x = 1, .y = 1, .width = 2, .height = 2 });
        try cmd.setScissor(.{ .x = 0, .y = 3, .width = 4, .height = 1 });
        try cmd.endPass();
        try device.submit();
        try testing.expectEqual(.{ 1, 1, 2, 2 }, stub.state.last_viewport);
        try testing.expectEqual(.{ 0, 0, 4, 1 }, stub.state.last_scissor);
    }
    // A face at level one of a cube of eight is four.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = cube }, .mip_level = 1, .layer = 4 } });
        try cmd.endPass();
        try device.submit();
        try testing.expectEqual(.{ 0, 0, 4, 4 }, stub.state.last_viewport);
        try testing.expectEqual(.{ 0, 0, 4, 4 }, stub.state.last_scissor);
    }
    // And the texture comes off the framebuffer when the pass ends.
    try testing.expect(!cast(device.impl).attached.color);
}

test "a pass with only depth is as big as its depth image, and a pipeline for it writes no colour" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    const shadow = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .color_format = null,
        .depth_format = .depth32_float,
        .depth = .standard,
    });
    const colour = try device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{} });
    try testing.expect(!as(PipelineRes, device.pipelines.get(shadow).?.native).writes_color);
    try testing.expect(as(PipelineRes, device.pipelines.get(colour).?.native).writes_color);

    const map = try device.createTexture(.{ .width = 32, .height = 16, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .depth = .{ .texture = map } });
    try cmd.setPipeline(shadow);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();
    try testing.expectEqual(.{ 0, 0, 32, 16 }, stub.state.last_viewport);
    try testing.expectEqual(1, stub.state.draw_calls);
    try testing.expectEqual(@as(?Enum, null), cast(device.impl).attached.depth);

    // The point is the one that takes the stencil along, when there is one.
    try testing.expectEqual(c.depth_attachment, attachmentPoint(.depth32_float));
    try testing.expectEqual(c.depth_attachment, attachmentPoint(.depth16_unorm));
    try testing.expectEqual(c.depth_stencil_attachment, attachmentPoint(.depth24_stencil8));
    try testing.expectEqual(c.depth_stencil_attachment, attachmentPoint(.depth32_float_stencil8));
    try testing.expectEqual(c.color_attachment0, attachmentPoint(.rgba16_float));
}

test "a depth attachment goes with the colour it is drawn beside, and the canvas keeps its own" {
    var device = try openDevice();
    defer device.deinit();

    const colour = try device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .render_target = true } });
    const depth = try device.createTexture(.{ .width = 16, .height = 16, .format = .depth24_stencil8, .usage = .{ .render_target = true } });
    const surface = try device.createSurface(.{});

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = colour } }, .depth = .{ .texture = depth } });
    // Mid-pass, both are on the framebuffer.
    try cmd.endPass();
    // A pass into the surface takes no texture, and clears the canvas's depth.
    try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } }, .depth = .{ .texture = depth, .clear_depth = 0.5 } });
    try cmd.endPass();
    try device.submit();
    try testing.expect(cast(device.impl).attached.depth == null and !cast(device.impl).attached.color);
    try testing.expectEqual(.{ 0, 0, 800, 600 }, stub.state.last_viewport);
}

test "a pipeline for what the device cannot draw is refused, and says why" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    // Four samples: there is no multisampled target.
    try testing.expectError(error.PipelineFailed, device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{}, .samples = 4 }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "4 samples") != null);
    // Two colours: there is one attachment.
    try testing.expectError(error.PipelineFailed, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .extra_color_formats = &.{.rgba8_unorm},
    }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "colour attachments") != null);
    // And nothing was left behind by either.
    try testing.expect(stub.state.live_objects <= 2);
}

test "the scratch framebuffer is made once, emptied after every pass, and given back" {
    var device = try openDevice();

    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    try testing.expect(cast(device.impl).fbo == .none);
    for (0..3) |_| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
        try cmd.endPass();
        try device.submit();
    }
    const first = cast(device.impl).fbo;
    try testing.expect(first != .none);
    const pixels = try device.readTexture(target, testing.allocator);
    testing.allocator.free(pixels);
    try testing.expectEqual(first, cast(device.impl).fbo);

    device.deinit();
    try testing.expectEqual(0, stub.state.live_objects);
}

test "an image a pass drew into is turned over on the way out, and one that was only written is not" {
    var device = try openDevice();
    defer device.deinit();

    const chain = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    const flat = as(TextureRes, device.textures.get(chain).?.native);
    const faces = as(TextureRes, device.textures.get(cube).?.native);

    // Made, and written to: as it was given, at every level.
    const bytes = solid(8 * 8, .{ 1, 2, 3, 4 });
    try device.writeTexture(chain, .{}, &bytes, 0, 0);
    for (0..4) |level| try testing.expect(!isDrawn(flat, 0, @intCast(level)));

    // A pass into the second level turns that one over, and no other.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = chain }, .mip_level = 1 } });
        try cmd.endPass();
        try device.submit();
    }
    try testing.expect(isDrawn(flat, 0, 1));
    try testing.expect(!isDrawn(flat, 0, 0) and !isDrawn(flat, 0, 2) and !isDrawn(flat, 0, 3));

    // A chain made from a first level that was written is written.
    {
        const cmd = device.begin();
        try cmd.generateMips(chain);
        try device.submit();
    }
    try testing.expect(!isDrawn(flat, 0, 0) and !isDrawn(flat, 0, 2));

    // And from one that was drawn, drawn: every level under it.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = chain } } });
        try cmd.endPass();
        try cmd.generateMips(chain);
        try device.submit();
    }
    for (0..4) |level| try testing.expect(isDrawn(flat, 0, @intCast(level)));

    // A cube keeps a record for each face.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = cube }, .layer = 3 } });
        try cmd.endPass();
        try cmd.generateMips(cube);
        try device.submit();
    }
    try testing.expect(isDrawn(faces, 3, 0) and isDrawn(faces, 3, 1) and isDrawn(faces, 3, 2));
    try testing.expect(!isDrawn(faces, 2, 0) and !isDrawn(faces, 4, 0) and !isDrawn(faces, 2, 2));

    // Past what is remembered, it is read as it always was: turned over.
    var deep: TextureRes = undefined;
    deep.faces = 6;
    deep.levels = 17;
    deep.drawn = .initEmpty();
    try testing.expectEqual(@as(?usize, 5 * 17 + 4), imageIndex(&deep, 5, 4));
    try testing.expectEqual(@as(?usize, null), imageIndex(&deep, 5, 16));
    try testing.expect(isDrawn(&deep, 5, 16));
    try testing.expect(!isDrawn(&deep, 5, 4));
}

test "a write to a drawn image puts it back the way it was given" {
    var device = try openDevice();
    defer device.deinit();

    const chain = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0, .usage = .{ .sampled = true, .render_target = true } });
    const flat = as(TextureRes, device.textures.get(chain).?.native);

    // Two passes draw two levels; a write should undo only the one it touches.
    inline for (.{ 1, 2 }) |level| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = chain }, .mip_level = level } });
        try cmd.endPass();
        try device.submit();
    }
    try testing.expect(isDrawn(flat, 0, 1) and isDrawn(flat, 0, 2));

    const bytes = solid(4 * 4, .{ 1, 2, 3, 4 });
    try device.writeTexture(chain, .{ .mip = 1 }, &bytes, 0, 0);
    try testing.expect(!isDrawn(flat, 0, 1) and isDrawn(flat, 0, 2));
}

test "a shader destroyed before its pipeline is kept until the pipeline goes" {
    var device = try openDevice();
    defer device.deinit();

    const shader = try device.createShader(shader_desc);
    const first = try device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{} });
    const second = try device.createPipeline(.{ .shader = shader, .attributes = &.{}, .buffers = &.{} });
    // The program and the two arrays.
    try testing.expectEqual(3, stub.state.live_objects);

    device.destroyShader(shader);
    try testing.expectEqual(3, stub.state.live_objects);
    device.destroyPipeline(first);
    try testing.expectEqual(2, stub.state.live_objects);
    // The last pipeline takes the program with it.
    device.destroyPipeline(second);
    try testing.expectEqual(0, stub.state.live_objects);

    // In the order that always worked, nothing changes.
    const again = try device.createShader(shader_desc);
    const pipeline = try device.createPipeline(.{ .shader = again, .attributes = &.{}, .buffers = &.{} });
    device.destroyPipeline(pipeline);
    try testing.expectEqual(1, stub.state.live_objects);
    device.destroyShader(again);
    try testing.expectEqual(0, stub.state.live_objects);
}
