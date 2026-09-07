// SPDX-License-Identifier: BSD-2-Clause

//! What a device keeps about each resource, and the handles a program holds
//! instead.
//!
//! A handle is eight bytes from `fluxion-id`: an index and a generation. A
//! stale one - destroyed, or never made by this device - answers
//! `error.InvalidHandle` rather than the wrong resource, which is the whole
//! reason not to hand out pointers. The backend's own object sits behind
//! `native`, and nothing outside the backend knows what it is.

const std = @import("std");
const ids = @import("fluxion_id");

const types = @import("types.zig");

pub const BufferEntry = struct {
    native: *anyopaque,
    kind: types.BufferKind,
    size: usize,
    dynamic: bool,
};

pub const TextureEntry = struct {
    native: *anyopaque,
    width: u32,
    height: u32,
    format: types.Format,
    usage: types.TextureUsage,
};

pub const SamplerEntry = struct {
    native: *anyopaque,
};

pub const ShaderEntry = struct {
    native: *anyopaque,
};

pub const PipelineEntry = struct {
    native: *anyopaque,
    topology: types.Topology,
    /// How many vertex buffer slots the pipeline reads, so a draw with fewer
    /// bound is caught before the driver reads memory that is not there.
    buffer_slots: u32,
    color_format: types.Format,
    depth_format: ?types.Format,
};

pub const SurfaceEntry = struct {
    native: *anyopaque,
    vsync: bool,
};

pub const Buffer = ids.handle.Handle(BufferEntry);
pub const Texture = ids.handle.Handle(TextureEntry);
pub const Sampler = ids.handle.Handle(SamplerEntry);
pub const Shader = ids.handle.Handle(ShaderEntry);
pub const Pipeline = ids.handle.Handle(PipelineEntry);
pub const Surface = ids.handle.Handle(SurfaceEntry);

pub const BufferTable = ids.handle.Table(BufferEntry);
pub const TextureTable = ids.handle.Table(TextureEntry);
pub const SamplerTable = ids.handle.Table(SamplerEntry);
pub const ShaderTable = ids.handle.Table(ShaderEntry);
pub const PipelineTable = ids.handle.Table(PipelineEntry);
pub const SurfaceTable = ids.handle.Table(SurfaceEntry);

test "handles are eight bytes and none is all zero" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Buffer));
    try std.testing.expect(Buffer.none.isNone());
    try std.testing.expectEqual(@as(u64, 0), Texture.none.toInt());
}
