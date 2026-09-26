// SPDX-License-Identifier: BSD-2-Clause

//! The backend that draws nothing.
//!
//! Every call succeeds, every resource is a real handle, every submit walks
//! the list and does nothing with it. What it is for:
//!
//!   * a test of render code on a machine with no GPU and no display, which
//!     is every build server;
//!   * a headless server that shares code with a client;
//!   * checking that a third backend's absence is an ordinary condition and
//!     not a crash.
//!
//! `readTexture` answers zeros of the right size, so a test that reads pixels
//! back gets a black picture rather than a failure - and can tell the two
//! apart by asking `Device.info`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const types = @import("../types.zig");
const backend = @import("../backend.zig");
const commands = @import("../commands.zig");
const Device = @import("../Device.zig");

const None = struct {
    gpa: Allocator,
};

/// Every resource on this backend is one of these, so that `native` is a
/// pointer to something rather than a lie.
const Resource = struct {
    width: u32 = 0,
    height: u32 = 0,
    format: types.Format = .rgba8_unorm,
};

pub fn open(gpa: Allocator, desc: types.DeviceDesc) backend.Error!backend.Opened {
    _ = desc;
    const self = try gpa.create(None);
    self.* = .{ .gpa = gpa };
    return .{ self, &vtable };
}

pub const vtable: backend.Vtable = .{
    .deinit = deinit,
    .info = info,
    .caps = caps,
    .createBuffer = createBuffer,
    .destroyBuffer = destroyResource,
    .updateBuffer = updateBuffer,
    .createTexture = createTexture,
    .destroyTexture = destroyResource,
    .writeTexture = writeTexture,
    .readTexture = readTexture,
    .createSampler = createSampler,
    .destroySampler = destroyResource,
    .createShader = createShader,
    .destroyShader = destroyResource,
    .createPipeline = createPipeline,
    .destroyPipeline = destroyResource,
    .createSurface = createSurface,
    .destroySurface = destroyResource,
    .resizeSurface = resizeSurface,
    .surfaceSize = surfaceSize,
    .present = present,
    .submit = submit,
};

fn cast(impl: backend.Impl) *None {
    return @ptrCast(@alignCast(impl));
}

fn resource(native: backend.Native) *Resource {
    return @ptrCast(@alignCast(native));
}

fn make(self: *None, value: Resource) backend.Error!backend.Native {
    const r = try self.gpa.create(Resource);
    r.* = value;
    return r;
}

fn deinit(impl: backend.Impl) void {
    const self = cast(impl);
    self.gpa.destroy(self);
}

fn info(impl: backend.Impl) types.Info {
    _ = impl;
    return .{ .backend = .none, .renderer = "nothing at all" };
}

/// It accepts everything, so it says it can do everything a real device could
/// be asked - which is what lets a test of render code run here and mean the
/// same on every backend that says yes.
fn caps(impl: backend.Impl) types.Caps {
    _ = impl;
    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = 16384,
            .max_texture_3d = 2048,
            .max_texture_cube = 16384,
            .max_texture_layers = 2048,
            .max_anisotropy = 16,
            .max_color_attachments = 8,
        },
        .features = .{ .sampler_border = true, .sampler_lod_bias = true, .compressed_partial_blocks = true },
    };
    for (std.enums.values(types.Format)) |format| {
        const plain = !format.isCompressed();
        answer.formats.set(format, .{
            .sampled = true,
            .filterable = true,
            .render_target = plain,
            .blendable = plain and !format.isDepth(),
            .generate_mips = plain and !format.isDepth(),
            .sample_counts = if (plain) 0b1111 else 0b1,
        });
    }
    return answer;
}

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) backend.Error!backend.Native {
    _ = desc;
    return make(cast(impl), .{});
}

fn destroyResource(impl: backend.Impl, native: backend.Native) void {
    cast(impl).gpa.destroy(resource(native));
}

fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) backend.Error!void {
    _ = impl;
    _ = native;
    _ = offset;
    _ = bytes;
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) backend.Error!backend.Native {
    return make(cast(impl), .{ .width = desc.width, .height = desc.height, .format = desc.format });
}

fn writeTexture(
    impl: backend.Impl,
    native: backend.Native,
    region: types.TextureRegion,
    bytes: []const u8,
    row_pitch: usize,
    slice_pitch: usize,
) backend.Error!void {
    _ = impl;
    _ = native;
    _ = region;
    _ = bytes;
    _ = row_pitch;
    _ = slice_pitch;
}

fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) backend.Error![]u8 {
    _ = impl;
    const r = resource(native);
    const width = types.mipExtent(r.width, sub.mip);
    const height = types.mipExtent(r.height, sub.mip);
    const pixels = try gpa.alloc(u8, @as(usize, width) * height * 4);
    @memset(pixels, 0);
    return pixels;
}

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) backend.Error!backend.Native {
    _ = desc;
    return make(cast(impl), .{});
}

fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) backend.Error!backend.Native {
    _ = desc;
    _ = log;
    return make(cast(impl), .{});
}

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) backend.Error!backend.Native {
    _ = desc;
    _ = shader;
    _ = log;
    return make(cast(impl), .{});
}

fn createSurface(impl: backend.Impl, desc: types.SurfaceDesc) backend.Error!backend.Native {
    return make(cast(impl), .{ .width = desc.width, .height = desc.height });
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) backend.Error!void {
    _ = impl;
    const r = resource(native);
    r.width = width;
    r.height = height;
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const r = resource(native);
    return .{ r.width, r.height };
}

fn present(impl: backend.Impl, native: backend.Native, mode: types.PresentMode) backend.Error!void {
    _ = impl;
    _ = native;
    _ = mode;
}

fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) backend.Error!void {
    _ = impl;
    _ = device;
    // Walked, so that a list a program builds is at least the shape the
    // real backends would see - and so that a debugger can stop here.
    for (list) |command| {
        switch (command) {
            else => {},
        }
    }
}
