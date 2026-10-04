// SPDX-License-Identifier: BSD-3-Clause

//! A texture's texels as RGBA, eight bits a channel, top row first: what
//! `readTexture` hands back on every backend that reads a texture through a
//! copy. One kernel for each way a texel can be packed, whatever the format
//! is called; what a format has in each channel and how many channels it has
//! is `types.Format.info`'s to say.

const std = @import("std");
const types = @import("../types.zig");

/// How a texel is packed.
pub const Decode = enum { unorm8, bgra8, float16, float32, rgb10a2, rg11b10_float };

/// How a format's texels are packed, or null for what is never read as
/// colour: depth, and what is compressed.
pub fn decodeOf(format: types.Format) ?Decode {
    return switch (format) {
        .r8_unorm, .rg8_unorm, .rgba8_unorm, .rgba8_unorm_srgb => .unorm8,
        .bgra8_unorm, .bgra8_unorm_srgb => .bgra8,
        .r16_float, .rg16_float, .rgba16_float => .float16,
        .r32_float, .rg32_float, .rgba32_float => .float32,
        .rgb10a2_unorm => .rgb10a2,
        .rg11b10_float => .rg11b10_float,
        else => null,
    };
}

/// Rows of `format`'s texels, `row_pitch` bytes apart from `data`, written
/// into `pixels` as RGBA8, `width` four-byte texels a row. A float is
/// clamped to nought and one.
pub fn convert(format: types.Format, decode: Decode, data: [*]const u8, row_pitch: usize, width: u32, height: u32, pixels: []u8) void {
    const kernel = kernels.get(decode);
    const row = format.info();
    const texel: usize = row.block_bytes;
    const out_row = @as(usize, width) * 4;
    const layout = lanes[row.channels];
    for (0..height) |y| {
        const source = data[y * row_pitch ..][0 .. width * texel];
        const destination = pixels[y * out_row ..][0..out_row];
        for (0..width) |x| {
            const values = kernel(source[x * texel ..][0..texel], row.channels);
            for (layout, 0..) |lane, i| destination[x * 4 + i] = pick(lane, values);
        }
    }
}

/// Reads the first `channels` values of a texel, in the order they are stored,
/// each as eight bits.
const Kernel = *const fn (texel: []const u8, channels: u8) [4]u8;

const kernels = std.EnumArray(Decode, Kernel).init(.{
    .unorm8 = decodeUnorm8,
    .bgra8 = decodeBgra8,
    .float16 = decodeFloat16,
    .float32 = decodeFloat32,
    .rgb10a2 = decodeRgb10a2,
    .rg11b10_float = decodeRg11b10,
});

/// Which stored value goes to red, green, blue and alpha, for a format that has
/// this many channels: one is grey, two are red and green, and a format that
/// has no alpha is opaque.
const Lane = enum { c0, c1, c2, c3, zero, one };
const lanes = [_][4]Lane{
    .{ .zero, .zero, .zero, .one }, // no channels: never asked
    .{ .c0, .c0, .c0, .one },
    .{ .c0, .c1, .zero, .one },
    .{ .c0, .c1, .c2, .one },
    .{ .c0, .c1, .c2, .c3 },
};

fn pick(lane: Lane, values: [4]u8) u8 {
    return switch (lane) {
        .zero => 0,
        .one => 255,
        .c0, .c1, .c2, .c3 => values[@intFromEnum(lane)],
    };
}

fn unormFromFloat(value: f32) u8 {
    // NaN is what `@max` drops, so it reads as zero.
    return @intFromFloat(@round(@min(@max(value, 0), 1) * 255));
}

fn decodeUnorm8(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| values[c] = texel[c];
    return values;
}

fn decodeBgra8(texel: []const u8, channels: u8) [4]u8 {
    var values = decodeUnorm8(texel, channels);
    std.mem.swap(u8, &values[0], &values[2]);
    return values;
}

fn decodeFloat16(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| {
        const half: f16 = @bitCast(std.mem.readInt(u16, texel[c * 2 ..][0..2], .little));
        values[c] = unormFromFloat(half);
    }
    return values;
}

fn decodeFloat32(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| {
        const single: f32 = @bitCast(std.mem.readInt(u32, texel[c * 4 ..][0..4], .little));
        values[c] = unormFromFloat(single);
    }
    return values;
}

fn decodeRgb10a2(texel: []const u8, channels: u8) [4]u8 {
    _ = channels;
    const packed_texel = std.mem.readInt(u32, texel[0..4], .little);
    const ten_bit_max = (1 << 10) - 1;
    const two_bit_max = (1 << 2) - 1;
    var values: [4]u8 = undefined;
    inline for (0..3) |c| {
        const raw: u32 = (packed_texel >> (c * 10)) & ten_bit_max;
        values[c] = @intCast((raw * 255 + ten_bit_max / 2) / ten_bit_max);
    }
    const alpha: u32 = packed_texel >> 30;
    values[3] = @intCast((alpha * 255 + two_bit_max / 2) / two_bit_max);
    return values;
}

/// The unsigned small floats of `R11G11B10_FLOAT`: five bits of exponent, biased
/// by 15, and what is left is mantissa, with the special values IEEE gives them.
fn smallFloat(bits: u32, comptime mantissa_bits: u5) f32 {
    const exponent_bias = 15;
    const exponent_all_ones = 31;
    const exponent = bits >> mantissa_bits;
    const mantissa: f32 = @floatFromInt(bits & ((1 << mantissa_bits) - 1));
    const scale: f32 = @floatFromInt(@as(u32, 1) << mantissa_bits);
    if (exponent == 0) return mantissa / scale * @exp2(@as(f32, 1 - exponent_bias));
    if (exponent == exponent_all_ones) return if (mantissa == 0) std.math.inf(f32) else std.math.nan(f32);
    return (1 + mantissa / scale) * @exp2(@as(f32, @floatFromInt(@as(i32, @intCast(exponent)) - exponent_bias)));
}

fn decodeRg11b10(texel: []const u8, channels: u8) [4]u8 {
    _ = channels;
    const packed_texel = std.mem.readInt(u32, texel[0..4], .little);
    const red_and_green_bits = 11;
    const red_mantissa = 6;
    const blue_mantissa = 5;
    const field = (1 << red_and_green_bits) - 1;
    return .{
        unormFromFloat(smallFloat(packed_texel & field, red_mantissa)),
        unormFromFloat(smallFloat((packed_texel >> red_and_green_bits) & field, red_mantissa)),
        unormFromFloat(smallFloat(packed_texel >> (2 * red_and_green_bits), blue_mantissa)),
        0,
    };
}

test "every format a texel is read from has a kernel, and nothing else does" {
    try std.testing.expectEqual(Decode.float16, decodeOf(.rgba16_float).?);
    try std.testing.expectEqual(Decode.rg11b10_float, decodeOf(.rg11b10_float).?);
    try std.testing.expectEqual(@as(?Decode, null), decodeOf(.depth32_float));
    try std.testing.expectEqual(@as(?Decode, null), decodeOf(.bc1_rgba_unorm));
}

test "half floats are clamped into eight bits, a channel missing is filled in" {
    // Red 2.0 (clamped), green 0.5, blue 0, alpha 1.
    const halves = [_]f16{ 2.0, 0.5, 0.0, 1.0 };
    var pixels: [4]u8 = undefined;
    convert(.rgba16_float, .float16, std.mem.sliceAsBytes(&halves).ptr, 8, 1, 1, &pixels);
    try std.testing.expectEqual([4]u8{ 255, 128, 0, 255 }, pixels);
    const one = [_]f32{0.25};
    convert(.r32_float, .float32, std.mem.sliceAsBytes(&one).ptr, 4, 1, 1, &pixels);
    try std.testing.expectEqual([4]u8{ 64, 64, 64, 255 }, pixels);
}
