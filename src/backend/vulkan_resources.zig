// SPDX-License-Identifier: BSD-2-Clause

//! Buffers, textures, samplers, shaders and pipelines: the resource half of
//! the Vulkan backend. `vulkan.zig` owns the device and the frame; this file
//! only ever receives a `*Vk` and hands back a `Native` it made.
//!
//! **Buffers are simple on purpose.** Every one of them - vertex, index or
//! uniform - is host-visible, host-coherent memory, mapped once at creation
//! and never unmapped until it is destroyed. `updateBuffer` is a `memcpy`.
//! Nothing here is the fastest a driver could do; it is the smallest thing
//! that is unambiguously correct.
//!
//! **Textures are device-local** and always `shader_read_only_optimal`
//! between one command buffer and the next, every level of them: a write or
//! a read moves one level to a transfer layout and back inside its own
//! command buffer, `generateMips` each pair of levels in turn, and a pass
//! into one ends it where it began. Uploads and readbacks record into the one
//! command buffer the backend owns (see `vulkan.zig`'s `record`/`finish`),
//! because the backend is fully synchronous and the two are never in flight
//! at once.
//!
//! **A pipeline is made again for each format it draws into.** A Vulkan
//! pipeline is tied to the format of its render pass, and a swapchain's may be
//! BGRA where the pipeline said RGBA, so a pipeline keeps what it was made
//! from, with shader modules of its own, and makes a variant the first time it
//! is bound in a pass of another format.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const vk = @import("fluxion_vulkan");
const types = @import("../types.zig");
const backend = @import("../backend.zig");

const vulkan = @import("vulkan.zig");
const Vk = vulkan.Vk;
const swapchain = @import("vulkan_swapchain.zig");
const readback_texels = @import("readback.zig");

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

// -------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------

/// The index of a memory type that satisfies `allowed` (a
/// `MemoryRequirements.memory_type_bits`) and has every property in `wanted`.
/// Vulkan requires drivers to list the more desirable types first, so the
/// first match is the conventional choice.
fn memoryTypeIndex(self: *Vk, allowed: u32, wanted: vk.gen.types.MemoryPropertyFlags) ?u32 {
    for (self.runtime.memory_properties.types(), 0..) |kind, index| {
        if (allowed & (@as(u32, 1) << @intCast(index)) == 0) continue;
        if (!kind.property_flags.contains(wanted)) continue;
        return @intCast(index);
    }
    return null;
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

/// One copy of a buffer's memory, mapped for its life, and the serial of
/// the last submission that recorded a use of it.
pub const Version = struct {
    buffer: vk.gen.types.Buffer,
    memory: vk.gen.types.DeviceMemory,
    mapped: [*]u8,
    busy: u64 = 0,
};

/// A buffer: the version draws are recorded with now, the ones a write
/// moved it off, and its bytes on the CPU, which a new version starts as.
/// See the module doc of `vulkan.zig`.
pub const BufferRes = struct {
    current: Version,
    spare: std.ArrayListUnmanaged(Version) = .empty,
    shadow: []u8,
    /// How far anything was ever written: a new version copies no further.
    written: usize,
    size: usize,
    usage: vk.gen.types.BufferUsageFlags,
};

pub fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    var usage: vk.gen.types.BufferUsageFlags = .{};
    switch (desc.kind) {
        .vertex => usage.vertex_buffer = true,
        .index => usage.index_buffer = true,
        .uniform => usage.uniform_buffer = true,
    }
    // A uniform block is read in whole vec4s, as on the other backends.
    const size = if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 16) else desc.size;
    const shadow = try self.gpa.alloc(u8, @max(size, 1));
    errdefer self.gpa.free(shadow);
    const version = try hostBuffer(self, size, usage, .{});
    errdefer freeVersion(self, version);
    const res = try self.gpa.create(BufferRes);
    res.* = .{ .current = version, .shadow = shadow, .written = 0, .size = size, .usage = usage };
    if (desc.data) |data| {
        @memcpy(version.mapped[0..data.len], data);
        @memcpy(shadow[0..data.len], data);
        res.written = data.len;
    }
    return res;
}

/// A buffer's memory the CPU writes and reads through a mapping kept for its
/// life: a version of every buffer here, and the staging and readback ones.
/// `cached` is asked for where the CPU reads it back, when the driver has
/// such memory.
fn hostBuffer(self: *Vk, size: usize, usage: vk.gen.types.BufferUsageFlags, extra: vk.gen.types.MemoryPropertyFlags) types.Error!Version {
    const vkd = self.runtime.vkd;
    var buffer: vk.gen.types.Buffer = .none;
    _ = vkd.createBuffer(self.runtime.device, &.{
        .size = @max(size, 1),
        .usage = usage,
        .sharing_mode = .exclusive,
    }, null, &buffer).check() catch return error.Failed;
    errdefer vkd.destroyBuffer(self.runtime.device, buffer, null);

    var requirements: vk.gen.types.MemoryRequirements = undefined;
    vkd.getBufferMemoryRequirements(self.runtime.device, buffer, &requirements);
    var wanted: vk.gen.types.MemoryPropertyFlags = .{ .host_visible = true, .host_coherent = true };
    const plain = wanted;
    if (extra.host_cached) wanted.host_cached = true;
    const type_index = memoryTypeIndex(self, requirements.memory_type_bits, wanted) orelse
        memoryTypeIndex(self, requirements.memory_type_bits, plain) orelse return error.NoDevice;

    var memory: vk.gen.types.DeviceMemory = .none;
    _ = vkd.allocateMemory(self.runtime.device, &.{
        .allocation_size = requirements.size,
        .memory_type_index = type_index,
    }, null, &memory).check() catch return error.OutOfMemory;
    errdefer vkd.freeMemory(self.runtime.device, memory, null);
    _ = vkd.bindBufferMemory(self.runtime.device, buffer, memory, 0).check() catch return error.Failed;

    var mapped: ?*anyopaque = null;
    _ = vkd.mapMemory(self.runtime.device, memory, 0, vk.gen.types.whole_size, .{}, &mapped).check() catch return error.Failed;
    return .{ .buffer = buffer, .memory = memory, .mapped = @ptrCast(mapped.?) };
}

pub fn freeVersion(self: *Vk, version: Version) void {
    self.runtime.vkd.unmapMemory(self.runtime.device, version.memory);
    self.runtime.vkd.destroyBuffer(self.runtime.device, version.buffer, null);
    self.runtime.vkd.freeMemory(self.runtime.device, version.memory, null);
}

pub fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(BufferRes, native);
    vulkan.bury(self, .{ .buffer = res.current });
    for (res.spare.items) |version| vulkan.bury(self, .{ .buffer = version });
    res.spare.deinit(self.gpa);
    self.gpa.free(res.shadow);
    self.gpa.destroy(res);
}

/// Write into the version the GPU is not reading: the current one when no
/// submission it may still be running used it, another when one did.
pub fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) types.Error!void {
    const self = vulkan.cast(impl);
    const res = as(BufferRes, native);
    const end = offset + bytes.len;
    if (res.current.busy > self.completed) vulkan.poll(self);
    if (res.current.busy > self.completed) {
        try res.spare.ensureUnusedCapacity(self.gpa, 1);
        const fresh = try spareVersion(self, res);
        res.spare.appendAssumeCapacity(res.current);
        res.current = fresh;
        // Everything else written is in the new one too.
        const before = @min(offset, res.written);
        @memcpy(fresh.mapped[0..before], res.shadow[0..before]);
        if (res.written > end) @memcpy(fresh.mapped[end..res.written], res.shadow[end..res.written]);
    }
    @memcpy(res.current.mapped[offset..end], bytes);
    @memcpy(res.shadow[offset..end], bytes);
    res.written = @max(res.written, end);
}

/// A version of `res` the GPU is done with, or a new one.
fn spareVersion(self: *Vk, res: *BufferRes) types.Error!Version {
    for (res.spare.items, 0..) |version, i| {
        if (version.busy <= self.completed) return res.spare.swapRemove(i);
    }
    return hostBuffer(self, res.size, res.usage, .{});
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

pub const TextureRes = struct {
    image: vk.gen.types.Image,
    memory: vk.gen.types.DeviceMemory,
    view: vk.gen.types.ImageView,
    width: u32,
    height: u32,
    format: types.Format,
    vk_format: vk.gen.types.Format,
    /// For a texture made to be drawn into: the framebuffer a pass begins.
    /// A multisampled one has none: each pass that resolves it makes its own.
    /// A chain of levels has one a level, in `level_framebuffers`, instead.
    framebuffer: vk.gen.types.Framebuffer = .none,
    /// Samples a pixel.
    samples: u8 = 1,
    /// A depth texture that is sampled as well - a shadow map - kept where
    /// a shader reads it between the passes that draw into it.
    sampled_depth: bool = false,
    /// What a shader reads it through: `view`, but for a depth format with
    /// stencil, a view of the depth alone, which is all a sampler can read.
    read_view: vk.gen.types.ImageView = .none,
    /// Levels in its chain. `view` covers them all, which is what a sampler
    /// picks its level from.
    mip_levels: u32 = 1,
    /// A chain drawn into, a level at a time: a view of each level, and a
    /// framebuffer of each, for the pass that names it.
    level_views: []vk.gen.types.ImageView = &.{},
    level_framebuffers: []vk.gen.types.Framebuffer = &.{},

    fn aspect(self: *const TextureRes) vk.gen.types.ImageAspectFlags {
        if (!self.format.isDepth()) return .{ .color = true };
        return .{ .depth = true, .stencil = self.format.hasStencil() };
    }

    /// The view a descriptor names.
    pub fn sampledView(self: *const TextureRes) vk.gen.types.ImageView {
        return if (self.read_view != .none) self.read_view else self.view;
    }

    /// How big level `mip` is, across and down.
    pub fn levelSize(self: *const TextureRes, mip: u32) [2]u32 {
        return .{ types.mipExtent(self.width, mip), types.mipExtent(self.height, mip) };
    }

    /// The view and the framebuffer a pass into level `mip` draws through.
    pub fn target(self: *const TextureRes, mip: u32) ?struct { view: vk.gen.types.ImageView, framebuffer: vk.gen.types.Framebuffer } {
        if (self.level_framebuffers.len > 0) {
            if (mip >= self.level_framebuffers.len) return null;
            return .{ .view = self.level_views[mip], .framebuffer = self.level_framebuffers[mip] };
        }
        if (mip != 0 or self.framebuffer == .none) return null;
        return .{ .view = self.view, .framebuffer = self.framebuffer };
    }
};

pub fn createTexture(impl: backend.Impl, desc: types.TextureDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const vkd = self.runtime.vkd;

    // `Device` already filters the dimension, the usage, the format and the
    // samples through `caps`, and resolves a whole chain to its count; a
    // chain of depth is not something a pass here draws.
    const levels = desc.mip_levels;
    if (levels > 1 and desc.format.isDepth()) return error.Unsupported;
    const vk_format = vulkan.toVkFormat(desc.format) orelse return error.Unsupported;
    const samples: u8 = @intCast(desc.samples);

    var image: vk.gen.types.Image = .none;
    _ = vkd.createImage(self.runtime.device, &.{
        .image_type = .@"2d",
        .format = vk_format,
        .extent = .{ .width = desc.width, .height = desc.height, .depth = 1 },
        .mip_levels = levels,
        .array_layers = 1,
        .samples = swapchain.sampleFlags(samples),
        .tiling = .optimal,
        // A depth texture is drawn into, and kept for the passes that load
        // it - and read too, if it is sampled; a multisampled one is only
        // drawn into, resolved where it is read.
        .usage = if (desc.format.isDepth())
            .{ .depth_stencil_attachment = true, .sampled = desc.usage.sampled }
        else if (samples > 1)
            .{ .color_attachment = true }
        else
            .{ .transfer_src = true, .transfer_dst = true, .sampled = true, .color_attachment = desc.usage.render_target },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null, &image).check() catch return error.Failed;
    errdefer vkd.destroyImage(self.runtime.device, image, null);

    var requirements: vk.gen.types.MemoryRequirements = undefined;
    vkd.getImageMemoryRequirements(self.runtime.device, image, &requirements);
    const type_index = memoryTypeIndex(self, requirements.memory_type_bits, .{ .device_local = true }) orelse
        memoryTypeIndex(self, requirements.memory_type_bits, .{}) orelse return error.NoDevice;

    var memory: vk.gen.types.DeviceMemory = .none;
    _ = vkd.allocateMemory(self.runtime.device, &.{
        .allocation_size = requirements.size,
        .memory_type_index = type_index,
    }, null, &memory).check() catch return error.OutOfMemory;
    errdefer vkd.freeMemory(self.runtime.device, memory, null);
    _ = vkd.bindImageMemory(self.runtime.device, image, memory, 0).check() catch return error.Failed;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);
    res.* = .{ .image = image, .memory = memory, .view = .none, .width = desc.width, .height = desc.height, .format = desc.format, .vk_format = vk_format, .samples = samples, .sampled_depth = desc.format.isDepth() and desc.usage.sampled, .mip_levels = levels };

    res.view = try imageView(self, image, vk_format, rangeOf(res.aspect(), 0, levels));
    errdefer vkd.destroyImageView(self.runtime.device, res.view, null);
    const view = res.view;

    if (res.sampled_depth and desc.format.hasStencil()) {
        res.read_view = try imageView(self, image, vk_format, rangeOf(.{ .depth = true }, 0, 1));
    }
    errdefer if (res.read_view != .none) vkd.destroyImageView(self.runtime.device, res.read_view, null);

    // A sampled depth texture into where a shader reads it, where every pass
    // that draws into it leaves it.
    if (res.sampled_depth) {
        try vulkan.ensureRecording(self);
        barrierOf(self, image, rangeOf(res.aspect(), 0, 1), .undefined, .shader_read_only_optimal, .{}, .{ .shader_read = true }, .{ .top_of_pipe = true }, .{ .fragment_shader = true });
        self.has_work = true;
        return res;
    }

    // Into where depth is written, the layout every pass that loads it
    // expects; the framebuffer is the pass's own, with its colour target.
    if (desc.format.isDepth()) {
        try vulkan.ensureRecording(self);
        const to: vk.gen.types.AccessFlags = .{ .depth_stencil_attachment_read = true, .depth_stencil_attachment_write = true };
        barrierOf(self, image, rangeOf(res.aspect(), 0, 1), .undefined, .depth_stencil_attachment_optimal, .{}, to, .{ .top_of_pipe = true }, .{ .early_fragment_tests = true });
        self.has_work = true;
        return res;
    }

    // Multisampled: into where colour is written, where every pass that
    // draws into it leaves it.
    if (samples > 1) {
        try vulkan.ensureRecording(self);
        barrier(self, image, 0, 1, .undefined, .color_attachment_optimal, .{}, .{ .color_attachment_write = true }, .{ .top_of_pipe = true }, .{ .color_attachment_output = true });
        self.has_work = true;
        return res;
    }

    // Drawn into: a framebuffer for the one level, or a view and a
    // framebuffer for each level of a chain, the size that level is.
    errdefer freeTargets(self, res);
    if (desc.usage.render_target) {
        const render_pass = try swapchain.getRenderPass(self, vk_format, .clear, .undefined, .shader_read_only_optimal, .none, 1, null);
        if (levels == 1) {
            res.framebuffer = try framebufferOf(self, render_pass, view, desc.width, desc.height);
        } else {
            res.level_views = try self.gpa.alloc(vk.gen.types.ImageView, levels);
            @memset(res.level_views, .none);
            res.level_framebuffers = try self.gpa.alloc(vk.gen.types.Framebuffer, levels);
            @memset(res.level_framebuffers, .none);
            for (0..levels) |mip| {
                const size = res.levelSize(@intCast(mip));
                res.level_views[mip] = try imageView(self, image, vk_format, rangeOf(.{ .color = true }, @intCast(mip), 1));
                res.level_framebuffers[mip] = try framebufferOf(self, render_pass, res.level_views[mip], size[0], size[1]);
            }
        }
    }

    // Every level into the layout it is sampled in, then level zero's
    // texels if it was given any.
    try vulkan.ensureRecording(self);
    barrier(self, image, 0, levels, .undefined, .shader_read_only_optimal, .{}, .{ .shader_read = true }, .{ .top_of_pipe = true }, .{ .fragment_shader = true });
    self.has_work = true;
    if (desc.data) |data| {
        try write(self, res, .{ .width = desc.width, .height = desc.height, .depth = 1 }, data, desc.effectiveRowPitch());
    }
    return res;
}

fn imageView(self: *Vk, image: vk.gen.types.Image, format: vk.gen.types.Format, range: vk.gen.types.ImageSubresourceRange) types.Error!vk.gen.types.ImageView {
    var view: vk.gen.types.ImageView = .none;
    _ = self.runtime.vkd.createImageView(self.runtime.device, &.{
        .image = image,
        .view_type = .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = range,
    }, null, &view).check() catch return error.Failed;
    return view;
}

fn framebufferOf(self: *Vk, render_pass: vk.gen.types.RenderPass, view: vk.gen.types.ImageView, width: u32, height: u32) types.Error!vk.gen.types.Framebuffer {
    const attachments = [_]vk.gen.types.ImageView{view};
    var framebuffer: vk.gen.types.Framebuffer = .none;
    _ = self.runtime.vkd.createFramebuffer(self.runtime.device, &.{
        .render_pass = render_pass,
        .attachment_count = attachments.len,
        .attachments = &attachments,
        .width = width,
        .height = height,
        .layers = 1,
    }, null, &framebuffer).check() catch return error.Failed;
    return framebuffer;
}

/// The framebuffers and the level views a texture drawn into has.
fn freeTargets(self: *Vk, res: *TextureRes) void {
    const vkd = self.runtime.vkd;
    if (res.framebuffer != .none) vkd.destroyFramebuffer(self.runtime.device, res.framebuffer, null);
    res.framebuffer = .none;
    for (res.level_framebuffers) |framebuffer| if (framebuffer != .none) vkd.destroyFramebuffer(self.runtime.device, framebuffer, null);
    for (res.level_views) |view| if (view != .none) vkd.destroyImageView(self.runtime.device, view, null);
    self.gpa.free(res.level_framebuffers);
    self.gpa.free(res.level_views);
    res.level_framebuffers = &.{};
    res.level_views = &.{};
}

/// `count` levels from `base`, of the one layer every texture here has.
fn rangeOf(aspect: vk.gen.types.ImageAspectFlags, base: u32, count: u32) vk.gen.types.ImageSubresourceRange {
    return .{ .aspect_mask = aspect, .base_mip_level = base, .level_count = count, .base_array_layer = 0, .layer_count = 1 };
}

/// One layout barrier on `count` colour levels from `base`, recorded into
/// the backend's command buffer.
fn barrier(
    self: *Vk,
    image: vk.gen.types.Image,
    base: u32,
    count: u32,
    old: vk.gen.types.ImageLayout,
    new: vk.gen.types.ImageLayout,
    src_access: vk.gen.types.AccessFlags,
    dst_access: vk.gen.types.AccessFlags,
    src_stage: vk.gen.types.PipelineStageFlags,
    dst_stage: vk.gen.types.PipelineStageFlags,
) void {
    barrierOf(self, image, rangeOf(.{ .color = true }, base, count), old, new, src_access, dst_access, src_stage, dst_stage);
}

fn barrierOf(
    self: *Vk,
    image: vk.gen.types.Image,
    range: vk.gen.types.ImageSubresourceRange,
    old: vk.gen.types.ImageLayout,
    new: vk.gen.types.ImageLayout,
    src_access: vk.gen.types.AccessFlags,
    dst_access: vk.gen.types.AccessFlags,
    src_stage: vk.gen.types.PipelineStageFlags,
    dst_stage: vk.gen.types.PipelineStageFlags,
) void {
    const barriers = [_]vk.gen.types.ImageMemoryBarrier{.{
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
        .old_layout = old,
        .new_layout = new,
        .src_queue_family_index = vk.gen.types.queue_family_ignored,
        .dst_queue_family_index = vk.gen.types.queue_family_ignored,
        .image = image,
        .subresource_range = range,
    }};
    self.runtime.vkd.cmdPipelineBarrier(self.command_buffer, src_stage, dst_stage, .{}, 0, null, 0, null, barriers.len, &barriers);
}

pub fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    vulkan.bury(self, .{ .texture = as(TextureRes, native) });
}

pub fn freeTexture(self: *Vk, res: *TextureRes) void {
    freeTargets(self, res);
    if (res.read_view != .none) self.runtime.vkd.destroyImageView(self.runtime.device, res.read_view, null);
    self.runtime.vkd.destroyImageView(self.runtime.device, res.view, null);
    self.runtime.vkd.destroyImage(self.runtime.device, res.image, null);
    self.runtime.vkd.freeMemory(self.runtime.device, res.memory, null);
    self.gpa.destroy(res);
}

pub fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) types.Error!void {
    _ = slice_pitch;
    const self = vulkan.cast(impl);
    return write(self, as(TextureRes, native), region, bytes, row_pitch);
}

/// A box of one level, through a staging buffer its rows are packed into
/// tightly, copied onto the texture; that level is left sampled-from, as
/// every level is between commands.
fn write(self: *Vk, res: *TextureRes, region: types.TextureRegion, bytes: []const u8, row_pitch: usize) types.Error!void {
    const row_bytes = res.format.rowBytes(region.width);
    const staging = try hostBuffer(self, row_bytes * region.height, .{ .transfer_src = true }, .{});
    var buried = false;
    defer if (!buried) freeVersion(self, staging);
    for (0..region.height) |y| @memcpy(staging.mapped[y * row_bytes ..][0..row_bytes], bytes[y * row_pitch ..][0..row_bytes]);

    try vulkan.ensureRecording(self);
    barrier(self, res.image, region.mip, 1, .shader_read_only_optimal, .transfer_dst_optimal, .{}, .{ .transfer_write = true }, .{ .fragment_shader = true, .color_attachment_output = true }, .{ .transfer = true });
    const copy = [_]vk.gen.types.BufferImageCopy{.{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = region.mip, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = @intCast(region.x), .y = @intCast(region.y), .z = 0 },
        .image_extent = .{ .width = region.width, .height = region.height, .depth = 1 },
    }};
    self.runtime.vkd.cmdCopyBufferToImage(self.command_buffer, staging.buffer, res.image, .transfer_dst_optimal, copy.len, &copy);
    barrier(self, res.image, region.mip, 1, .transfer_dst_optimal, .shader_read_only_optimal, .{ .transfer_write = true }, .{ .shader_read = true }, .{ .transfer = true }, .{ .fragment_shader = true });
    self.has_work = true;
    // Read by the copy when the recording runs: freed once it has.
    vulkan.bury(self, .{ .buffer = staging });
    buried = true;
}

/// One level, copied into a buffer the CPU reads and handed back as RGBA,
/// eight bits a channel, top row first - a Vulkan image is stored top row
/// first, and a pass draws into it that way up, see `vulkan.zig`'s viewport -
/// with one channel repeated into red, green and blue as the Direct3D 11
/// backend does.
pub fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) types.Error![]u8 {
    const self = vulkan.cast(impl);
    const res = as(TextureRes, native);
    const decode = readback_texels.decodeOf(res.format) orelse return error.Unsupported;
    const size = res.levelSize(sub.mip);
    const row_bytes = res.format.rowBytes(size[0]);
    const readback = try hostBuffer(self, row_bytes * size[1], .{ .transfer_dst = true }, .{ .host_cached = true });
    defer freeVersion(self, readback);

    try vulkan.ensureRecording(self);
    barrier(self, res.image, sub.mip, 1, .shader_read_only_optimal, .transfer_src_optimal, .{ .color_attachment_write = true, .transfer_write = true }, .{ .transfer_read = true }, .{ .color_attachment_output = true, .fragment_shader = true, .transfer = true }, .{ .transfer = true });
    const copy = [_]vk.gen.types.BufferImageCopy{.{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = sub.mip, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = size[0], .height = size[1], .depth = 1 },
    }};
    self.runtime.vkd.cmdCopyImageToBuffer(self.command_buffer, res.image, .transfer_src_optimal, readback.buffer, copy.len, &copy);
    barrier(self, res.image, sub.mip, 1, .transfer_src_optimal, .shader_read_only_optimal, .{}, .{ .shader_read = true }, .{ .transfer = true }, .{ .fragment_shader = true });
    const to_host = [_]vk.gen.types.BufferMemoryBarrier{.{
        .src_access_mask = .{ .transfer_write = true },
        .dst_access_mask = .{ .host_read = true },
        .src_queue_family_index = vk.gen.types.queue_family_ignored,
        .dst_queue_family_index = vk.gen.types.queue_family_ignored,
        .buffer = readback.buffer,
        .offset = 0,
        .size = vk.gen.types.whole_size,
    }};
    self.runtime.vkd.cmdPipelineBarrier(self.command_buffer, .{ .transfer = true }, .{ .host = true }, .{}, 0, null, to_host.len, &to_host, 0, null);
    try vulkan.finish(self);
    // The one wait a readback has: the bytes are wanted now.
    try vulkan.waitFor(self, self.submitted);

    const pixels = try gpa.alloc(u8, @as(usize, size[0]) * 4 * size[1]);
    readback_texels.convert(res.format, decode, readback.mapped, row_bytes, size[0], size[1], pixels);
    return pixels;
}

/// Every level below the first, each filtered down from the one above it
/// by a blit: the level above made a source and this one a destination,
/// and both back to where a shader reads them once it is filled.
/// Recorded into the open recording, outside any pass; `caps` says which
/// formats the device blits with a linear filter, and `Device` refuses the
/// rest.
pub fn generateMips(self: *Vk, res: *TextureRes) void {
    const drawn: vk.gen.types.AccessFlags = .{ .color_attachment_write = true, .transfer_write = true };
    const any_stage: vk.gen.types.PipelineStageFlags = .{ .color_attachment_output = true, .fragment_shader = true, .transfer = true };
    for (1..res.mip_levels) |level| {
        const mip: u32 = @intCast(level);
        barrier(self, res.image, mip - 1, 1, .shader_read_only_optimal, .transfer_src_optimal, drawn, .{ .transfer_read = true }, any_stage, .{ .transfer = true });
        barrier(self, res.image, mip, 1, .shader_read_only_optimal, .transfer_dst_optimal, .{ .shader_read = true }, .{ .transfer_write = true }, any_stage, .{ .transfer = true });
        const above = res.levelSize(mip - 1);
        const here = res.levelSize(mip);
        const blit = [_]vk.gen.types.ImageBlit{.{
            .src_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = mip - 1, .base_array_layer = 0, .layer_count = 1 },
            .src_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{ .x = @intCast(above[0]), .y = @intCast(above[1]), .z = 1 } },
            .dst_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = mip, .base_array_layer = 0, .layer_count = 1 },
            .dst_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{ .x = @intCast(here[0]), .y = @intCast(here[1]), .z = 1 } },
        }};
        self.runtime.vkd.cmdBlitImage(self.command_buffer, res.image, .transfer_src_optimal, res.image, .transfer_dst_optimal, blit.len, &blit, .linear);
        barrier(self, res.image, mip - 1, 1, .transfer_src_optimal, .shader_read_only_optimal, .{}, .{ .shader_read = true }, .{ .transfer = true }, .{ .fragment_shader = true });
        barrier(self, res.image, mip, 1, .transfer_dst_optimal, .shader_read_only_optimal, .{ .transfer_write = true }, .{ .shader_read = true, .transfer_read = true }, .{ .transfer = true }, .{ .fragment_shader = true, .transfer = true });
    }
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

pub const SamplerRes = struct {
    sampler: vk.gen.types.Sampler,
};

pub fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    var sampler: vk.gen.types.Sampler = .none;
    _ = self.runtime.vkd.createSampler(self.runtime.device, &.{
        .mag_filter = vkFilter(desc.mag_filter),
        .min_filter = vkFilter(desc.min_filter),
        .mipmap_mode = if (desc.mip_filter == .linear) .linear else .nearest,
        .address_mode_u = vkWrap(desc.wrap_u),
        .address_mode_v = vkWrap(desc.wrap_v),
        .address_mode_w = vkWrap(desc.wrap_w),
        .mip_lod_bias = desc.lod_bias,
        // Anisotropy only where both filters are linear, as on Direct3D: a
        // way of filtering linearly along a stretched footprint. `Device`
        // has clamped it to what `caps` said the device takes.
        .anisotropy_enable = if (anisotropic(desc)) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .max_anisotropy = if (anisotropic(desc)) @floatFromInt(desc.max_anisotropy) else 1,
        .compare_enable = if (desc.compare != null) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .compare_op = vkCompare(desc.compare orelse .always),
        // "Level zero only" is a range with one level in it, as on Direct3D.
        .min_lod = if (desc.mip_filter == .none) 0 else desc.lod_min,
        .max_lod = if (desc.mip_filter == .none) 0 else desc.lod_max,
        .border_color = switch (desc.border) {
            .transparent_black => .float_transparent_black,
            .opaque_black => .float_opaque_black,
            .opaque_white => .float_opaque_white,
        },
        .unnormalized_coordinates = vk.gen.types.vk_false,
    }, null, &sampler).check() catch return error.Failed;

    const res = try self.gpa.create(SamplerRes);
    res.* = .{ .sampler = sampler };
    return res;
}

fn anisotropic(desc: types.SamplerDesc) bool {
    return desc.max_anisotropy > 1 and desc.min_filter == .linear and desc.mag_filter == .linear;
}

pub fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    vulkan.bury(self, .{ .sampler = as(SamplerRes, native) });
}

pub fn freeSampler(self: *Vk, res: *SamplerRes) void {
    self.runtime.vkd.destroySampler(self.runtime.device, res.sampler, null);
    self.gpa.destroy(res);
}

fn vkFilter(f: types.Filter) vk.gen.types.Filter {
    return switch (f) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

fn vkWrap(w: types.Wrap) vk.gen.types.SamplerAddressMode {
    return switch (w) {
        .repeat => .repeat,
        .clamp_to_edge => .clamp_to_edge,
        .mirror => .mirrored_repeat,
        .border => .clamp_to_border,
    };
}

fn vkCompare(c: types.CompareFn) vk.gen.types.CompareOp {
    return switch (c) {
        .never => .never,
        .less => .less,
        .equal => .equal,
        .less_equal => .less_or_equal,
        .greater => .greater,
        .not_equal => .not_equal,
        .greater_equal => .greater_or_equal,
        .always => .always,
    };
}

// -------------------------------------------------------------------------
// Shaders
// -------------------------------------------------------------------------

/// The two stages' SPIR-V, kept: a pipeline makes modules of its own from
/// them, so that it can make variants after the shader is gone.
pub const ShaderRes = struct {
    vertex: []u32,
    fragment: []u32,
};

pub fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const words = desc.spirv orelse {
        log.writeAll("fluxion-rhi: the Vulkan backend needs ShaderDesc.spirv") catch {};
        return error.ShaderFailed;
    };

    // Made once here, so that SPIR-V the driver will not take is said now
    // rather than when a pipeline is.
    const vertex = createModule(self, words.vertex) catch {
        log.writeAll("fluxion-rhi: vkCreateShaderModule failed for the vertex stage") catch {};
        return error.ShaderFailed;
    };
    self.runtime.vkd.destroyShaderModule(self.runtime.device, vertex, null);
    const fragment = createModule(self, words.fragment) catch {
        log.writeAll("fluxion-rhi: vkCreateShaderModule failed for the fragment stage") catch {};
        return error.ShaderFailed;
    };
    self.runtime.vkd.destroyShaderModule(self.runtime.device, fragment, null);

    const res = try self.gpa.create(ShaderRes);
    errdefer self.gpa.destroy(res);
    const kept_vertex = try self.gpa.dupe(u32, words.vertex);
    errdefer self.gpa.free(kept_vertex);
    res.* = .{ .vertex = kept_vertex, .fragment = try self.gpa.dupe(u32, words.fragment) };
    return res;
}

fn createModule(self: *Vk, words: []const u32) !vk.gen.types.ShaderModule {
    var module: vk.gen.types.ShaderModule = .none;
    _ = try self.runtime.vkd.createShaderModule(self.runtime.device, &.{
        .code_size = words.len * @sizeOf(u32),
        .code = words.ptr,
    }, null, &module).check();
    return module;
}

pub fn destroyShader(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(ShaderRes, native);
    self.gpa.free(res.vertex);
    self.gpa.free(res.fragment);
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Pipelines
// -------------------------------------------------------------------------

const max_vertex_bindings = 8;
const max_vertex_attributes = 16;
/// Formats one pipeline is made for before it refuses another: RGBA and
/// BGRA and a one-channel target leave room.
const max_variants = 4;

pub const PipelineRes = struct {
    vertex: vk.gen.types.ShaderModule,
    fragment: vk.gen.types.ShaderModule,
    bindings: [max_vertex_bindings]vk.gen.types.VertexInputBindingDescription,
    binding_count: u32,
    attributes: [max_vertex_attributes]vk.gen.types.VertexInputAttributeDescription,
    attribute_count: u32,
    topology: vk.gen.types.PrimitiveTopology,
    cull: vk.gen.types.CullModeFlags,
    front_face: vk.gen.types.FrontFace,
    blend: vk.gen.types.PipelineColorBlendAttachmentState,
    depth: types.DepthState,
    /// Samples a pixel of the passes it draws in.
    samples: u8,
    variants: [max_variants]?Variant = @splat(null),

    const Variant = struct { format: vk.gen.types.Format, depth: vk.gen.types.Format, pipeline: vk.gen.types.Pipeline };
};

pub fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader_native: backend.Native, log: *Io.Writer) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const vkd = self.runtime.vkd;
    if (desc.extra_color_formats.len > 0) {
        log.writeAll("fluxion-rhi: the Vulkan backend has no multiple render targets yet") catch {};
        return error.Unsupported;
    }
    const vk_format: vk.gen.types.Format = if (desc.color_format) |format| vulkan.toVkFormat(format) orelse return error.Unsupported else .undefined;
    const depth_format: vk.gen.types.Format = if (desc.depth_format) |format| vulkan.toVkFormat(format) orelse return error.Unsupported else .undefined;
    if (desc.buffers.len > max_vertex_bindings or desc.attributes.len > max_vertex_attributes) return error.Unsupported;

    const shader = as(ShaderRes, shader_native);
    const vertex = createModule(self, shader.vertex) catch return error.PipelineFailed;
    errdefer vkd.destroyShaderModule(self.runtime.device, vertex, null);
    const fragment = createModule(self, shader.fragment) catch return error.PipelineFailed;
    errdefer vkd.destroyShaderModule(self.runtime.device, fragment, null);

    const res = try self.gpa.create(PipelineRes);
    errdefer self.gpa.destroy(res);
    res.* = .{
        .vertex = vertex,
        .fragment = fragment,
        .bindings = undefined,
        .binding_count = @intCast(desc.buffers.len),
        .attributes = undefined,
        .attribute_count = @intCast(desc.attributes.len),
        .topology = vkTopology(desc.topology),
        .cull = switch (desc.cull) {
            .none => .{},
            .back => .{ .back = true },
            .front => .{ .front = true },
        },
        .front_face = if (desc.front_face == .ccw) .counter_clockwise else .clockwise,
        .depth = desc.depth,
        .samples = @intCast(desc.samples),
        .blend = .{
            .blend_enable = if (desc.blend.enabled) vk.gen.types.vk_true else vk.gen.types.vk_false,
            .src_color_blend_factor = vkBlendFactor(desc.blend.src_rgb),
            .dst_color_blend_factor = vkBlendFactor(desc.blend.dst_rgb),
            .color_blend_op = vkBlendOp(desc.blend.op_rgb),
            .src_alpha_blend_factor = vkBlendFactor(desc.blend.src_alpha),
            .dst_alpha_blend_factor = vkBlendFactor(desc.blend.dst_alpha),
            .alpha_blend_op = vkBlendOp(desc.blend.op_alpha),
            .color_write_mask = .{ .r = true, .g = true, .b = true, .a = true },
        },
    };
    for (desc.buffers, 0..) |buf, i| res.bindings[i] = .{
        .binding = @intCast(i),
        .stride = buf.stride,
        .input_rate = if (buf.step == .instance) .instance else .vertex,
    };
    for (desc.attributes, 0..) |attr, i| res.attributes[i] = .{
        .location = attr.location,
        .binding = attr.buffer,
        .format = vertexFormat(attr.format),
        .offset = attr.offset,
    };

    // The format it said it draws into is made now, so that a pipeline the
    // driver will not make is said at once.
    _ = pipelineFor(self, res, vk_format, depth_format) catch {
        log.writeAll("fluxion-rhi: vkCreateGraphicsPipelines failed") catch {};
        return error.PipelineFailed;
    };
    return res;
}

/// The pipeline `res` is for a pass into `format` - `undefined` for none -
/// with a depth attachment of `depth` - `undefined` for none - made the
/// first time it is asked for.
pub fn pipelineFor(self: *Vk, res: *PipelineRes, format: vk.gen.types.Format, depth: vk.gen.types.Format) types.Error!vk.gen.types.Pipeline {
    var free: ?usize = null;
    for (&res.variants, 0..) |*slot, i| {
        if (slot.*) |variant| {
            if (variant.format == format and variant.depth == depth) return variant.pipeline;
        } else if (free == null) free = i;
    }
    const index = free orelse return error.Unsupported;

    const vertex_input: vk.gen.types.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = res.binding_count,
        .vertex_binding_descriptions = if (res.binding_count > 0) &res.bindings else null,
        .vertex_attribute_description_count = res.attribute_count,
        .vertex_attribute_descriptions = if (res.attribute_count > 0) &res.attributes else null,
    };
    const input_assembly: vk.gen.types.PipelineInputAssemblyStateCreateInfo = .{
        .topology = res.topology,
        .primitive_restart_enable = vk.gen.types.vk_false,
    };
    // Real viewport/scissor come from `vkCmdSetViewport`/`vkCmdSetScissor`,
    // set dynamically per `commands.Command.set_viewport`/`.set_scissor` -
    // this is only the count `PipelineDynamicStateCreateInfo` promises.
    const viewport_state: vk.gen.types.PipelineViewportStateCreateInfo = .{ .viewport_count = 1, .scissor_count = 1 };
    const rasterization: vk.gen.types.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = vk.gen.types.vk_false,
        .rasterizer_discard_enable = vk.gen.types.vk_false,
        .polygon_mode = .fill,
        .cull_mode = res.cull,
        .front_face = res.front_face,
        .depth_bias_enable = if (res.depth.biased()) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .depth_bias_constant_factor = @floatFromInt(res.depth.bias),
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = res.depth.slope_bias,
        .line_width = 1,
    };
    const multisample: vk.gen.types.PipelineMultisampleStateCreateInfo = .{
        .rasterization_samples = swapchain.sampleFlags(res.samples),
        .sample_shading_enable = vk.gen.types.vk_false,
        .min_sample_shading = 0,
        .alpha_to_coverage_enable = vk.gen.types.vk_false,
        .alpha_to_one_enable = vk.gen.types.vk_false,
    };
    const blend_attachment = [_]vk.gen.types.PipelineColorBlendAttachmentState{res.blend};
    const color_blend: vk.gen.types.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = vk.gen.types.vk_false,
        .logic_op = .copy,
        .attachment_count = if (format != .undefined) blend_attachment.len else 0,
        .attachments = &blend_attachment,
        .blend_constants = .{ 0, 0, 0, 0 },
    };
    const keep: vk.gen.types.StencilOpState = .{ .fail_op = .keep, .pass_op = .keep, .depth_fail_op = .keep, .compare_op = .always, .compare_mask = 0, .write_mask = 0, .reference = 0 };
    const depth_stencil: vk.gen.types.PipelineDepthStencilStateCreateInfo = .{
        .depth_test_enable = if (res.depth.test_enabled) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .depth_write_enable = if (res.depth.write) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .depth_compare_op = vkCompare(res.depth.compare),
        .depth_bounds_test_enable = vk.gen.types.vk_false,
        .stencil_test_enable = vk.gen.types.vk_false,
        .front = keep,
        .back = keep,
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
    const dynamic_states = [_]vk.gen.types.DynamicState{ .viewport, .scissor };
    const dynamic_state: vk.gen.types.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .dynamic_states = &dynamic_states,
    };

    // Any render pass whose attachments have these formats and this many
    // samples is compatible at `vkCmdBeginRenderPass` time; the load op, the
    // layouts and - with one subpass - a resolve do not matter.
    const render_pass = try swapchain.getRenderPass(self, format, .clear, .undefined, if (res.samples > 1) .color_attachment_optimal else .shader_read_only_optimal, .{ .format = depth, .load = if (depth != .undefined) .clear else .dont_care }, res.samples, null);

    const stages = [_]vk.gen.types.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex = true }, .module = res.vertex, .name = "main" },
        .{ .stage = .{ .fragment = true }, .module = res.fragment, .name = "main" },
    };
    const pipeline_info = [_]vk.gen.types.GraphicsPipelineCreateInfo{.{
        .stage_count = stages.len,
        .stages = &stages,
        .vertex_input_state = &vertex_input,
        .input_assembly_state = &input_assembly,
        .viewport_state = &viewport_state,
        .rasterization_state = &rasterization,
        .multisample_state = &multisample,
        .depth_stencil_state = if (depth != .undefined) &depth_stencil else null,
        .color_blend_state = &color_blend,
        .dynamic_state = &dynamic_state,
        .layout = self.pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_index = -1,
    }};

    var pipeline: vk.gen.types.Pipeline = .none;
    _ = self.runtime.vkd.createGraphicsPipelines(self.runtime.device, .none, 1, &pipeline_info, null, @ptrCast(&pipeline)).check() catch return error.PipelineFailed;
    res.variants[index] = .{ .format = format, .depth = depth, .pipeline = pipeline };
    return pipeline;
}

pub fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(PipelineRes, native);
    if (self.current_pipeline == res) self.current_pipeline = null;
    vulkan.bury(self, .{ .pipeline = res });
}

pub fn freePipeline(self: *Vk, res: *PipelineRes) void {
    for (res.variants) |maybe| if (maybe) |variant| self.runtime.vkd.destroyPipeline(self.runtime.device, variant.pipeline, null);
    self.runtime.vkd.destroyShaderModule(self.runtime.device, res.vertex, null);
    self.runtime.vkd.destroyShaderModule(self.runtime.device, res.fragment, null);
    self.gpa.destroy(res);
}

fn vertexFormat(f: types.VertexFormat) vk.gen.types.Format {
    return switch (f) {
        .float => .r32_sfloat,
        .float2 => .r32g32_sfloat,
        .float3 => .r32g32b32_sfloat,
        .float4 => .r32g32b32a32_sfloat,
        .ubyte4_norm => .r8g8b8a8_unorm,
        .ubyte4 => .r8g8b8a8_uint,
        .uint => .r32_uint,
        .int => .r32_sint,
    };
}

fn vkTopology(t: types.Topology) vk.gen.types.PrimitiveTopology {
    return switch (t) {
        .triangles => .triangle_list,
        .triangle_strip => .triangle_strip,
        .lines => .line_list,
        .line_strip => .line_strip,
        .points => .point_list,
    };
}

fn vkBlendFactor(f: types.BlendFactor) vk.gen.types.BlendFactor {
    return switch (f) {
        .zero => .zero,
        .one => .one,
        .src_color => .src_color,
        .one_minus_src_color => .one_minus_src_color,
        .src_alpha => .src_alpha,
        .one_minus_src_alpha => .one_minus_src_alpha,
        .dst_color => .dst_color,
        .one_minus_dst_color => .one_minus_dst_color,
        .dst_alpha => .dst_alpha,
        .one_minus_dst_alpha => .one_minus_dst_alpha,
    };
}

fn vkBlendOp(o: types.BlendOp) vk.gen.types.BlendOp {
    return switch (o) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .reverse_subtract,
        .min => .min,
        .max => .max,
    };
}

// -------------------------------------------------------------------------
// Dummy resources - what an unbound uniform or texture slot reads from
// -------------------------------------------------------------------------

/// A pipeline's shared layout always has four uniform and four texture
/// bindings, whether or not a particular draw's shader statically uses all
/// of them - and Vulkan wants every binding a bound pipeline could use to
/// hold a live descriptor. These fill the ones a program never called
/// `setUniformBuffer`/`setTexture` on.
pub fn createDummyBuffer(self: *Vk) !*BufferRes {
    const native = try createBuffer(self, .{ .kind = .uniform, .size = 16 });
    return as(BufferRes, native);
}

pub fn createDummyTexture(self: *Vk) !*TextureRes {
    const pixel = [4]u8{ 255, 255, 255, 255 };
    const native = try createTexture(self, .{ .width = 1, .height = 1, .format = .rgba8_unorm, .data = &pixel });
    return as(TextureRes, native);
}

pub fn createDummySampler(self: *Vk) !*SamplerRes {
    const native = try createSampler(self, .{});
    return as(SamplerRes, native);
}

pub fn destroyDummyResources(self: *Vk) void {
    destroyBuffer(self, self.dummy_buffer);
    destroyTexture(self, self.dummy_texture);
    destroySampler(self, self.dummy_sampler);
}
