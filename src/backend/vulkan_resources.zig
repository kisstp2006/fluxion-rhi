// SPDX-License-Identifier: BSD-2-Clause

//! Buffers, textures, samplers, shaders and pipelines: the resource half of
//! the Vulkan backend. `vulkan.zig` owns the device and the frame; this file
//! only ever receives a `*Vk` and hands back a `Native` it made.
//!
//! **Buffers are simple on purpose.** Every one of them - vertex, index or
//! uniform - is host-visible, host-coherent memory, mapped once at creation
//! and never unmapped until it is destroyed. `updateBuffer` is a `memcpy`.
//! Nothing here is the fastest a driver could do; it is the smallest thing
//! that is unambiguously correct, which is what the MVP scope in the plan
//! asks for.
//!
//! **Textures are device-local**, uploaded once through a staging buffer and
//! a one-shot command buffer the backend already owns (see `vulkan.zig`'s
//! `command_buffer`/`fence`) - reused here exactly as `submit` reuses it,
//! because the backend is fully synchronous and the two are never in flight
//! at once.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const vk = @import("fluxion_vulkan");
const types = @import("../types.zig");
const backend = @import("../backend.zig");

const vulkan = @import("vulkan.zig");
const Vk = vulkan.Vk;
const swapchain = @import("vulkan_swapchain.zig");

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

pub const BufferRes = struct {
    buffer: vk.gen.types.Buffer,
    memory: vk.gen.types.DeviceMemory,
    /// Mapped for the whole life of the buffer.
    mapped: [*]u8,
    size: usize,
};

pub fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const vkd = self.runtime.vkd;

    var usage: vk.gen.types.BufferUsageFlags = .{};
    switch (desc.kind) {
        .vertex => usage.vertex_buffer = true,
        .index => usage.index_buffer = true,
        .uniform => usage.uniform_buffer = true,
    }

    var buffer: vk.gen.types.Buffer = .none;
    _ = vkd.createBuffer(self.runtime.device, &.{
        .size = desc.size,
        .usage = usage,
        .sharing_mode = .exclusive,
    }, null, &buffer).check() catch return error.Failed;
    errdefer vkd.destroyBuffer(self.runtime.device, buffer, null);

    var requirements: vk.gen.types.MemoryRequirements = undefined;
    vkd.getBufferMemoryRequirements(self.runtime.device, buffer, &requirements);
    const type_index = memoryTypeIndex(self, requirements.memory_type_bits, .{
        .host_visible = true,
        .host_coherent = true,
    }) orelse return error.NoDevice;

    var memory: vk.gen.types.DeviceMemory = .none;
    _ = vkd.allocateMemory(self.runtime.device, &.{
        .allocation_size = requirements.size,
        .memory_type_index = type_index,
    }, null, &memory).check() catch return error.OutOfMemory;
    errdefer vkd.freeMemory(self.runtime.device, memory, null);
    _ = vkd.bindBufferMemory(self.runtime.device, buffer, memory, 0).check() catch return error.Failed;

    var mapped: ?*anyopaque = null;
    _ = vkd.mapMemory(self.runtime.device, memory, 0, vk.gen.types.whole_size, .{}, &mapped).check() catch return error.Failed;

    const res = try self.gpa.create(BufferRes);
    res.* = .{ .buffer = buffer, .memory = memory, .mapped = @ptrCast(mapped.?), .size = desc.size };
    if (desc.data) |data| @memcpy(res.mapped[0..data.len], data);
    return res;
}

pub fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(BufferRes, native);
    self.runtime.vkd.unmapMemory(self.runtime.device, res.memory);
    self.runtime.vkd.destroyBuffer(self.runtime.device, res.buffer, null);
    self.runtime.vkd.freeMemory(self.runtime.device, res.memory, null);
    self.gpa.destroy(res);
}

pub fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) types.Error!void {
    _ = impl;
    const res = as(BufferRes, native);
    @memcpy(res.mapped[offset..][0..bytes.len], bytes);
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
};

pub fn createTexture(impl: backend.Impl, desc: types.TextureDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const vkd = self.runtime.vkd;

    // `Device` already filters most of the MVP scope (dimension, usage,
    // format) through `caps`; a mip chain and multisampling are not caps,
    // so they are refused here.
    if (desc.mip_levels != 1) return error.Unsupported;
    if (desc.samples != 1) return error.Unsupported;
    if (desc.usage.render_target) return error.Unsupported;
    const vk_format = vulkan.toVkFormat(desc.format) orelse return error.Unsupported;

    var image: vk.gen.types.Image = .none;
    _ = vkd.createImage(self.runtime.device, &.{
        .image_type = .@"2d",
        .format = vk_format,
        .extent = .{ .width = desc.width, .height = desc.height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .x1 = true },
        .tiling = .optimal,
        .usage = .{ .transfer_dst = true, .sampled = true },
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

    var view: vk.gen.types.ImageView = .none;
    _ = vkd.createImageView(self.runtime.device, &.{
        .image = image,
        .view_type = .@"2d",
        .format = vk_format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = colorRange(),
    }, null, &view).check() catch return error.Failed;
    errdefer vkd.destroyImageView(self.runtime.device, view, null);

    try uploadTexture(self, image, desc);

    const res = try self.gpa.create(TextureRes);
    res.* = .{ .image = image, .memory = memory, .view = view, .width = desc.width, .height = desc.height };
    return res;
}

fn colorRange() vk.gen.types.ImageSubresourceRange {
    return .{ .aspect_mask = .{ .color = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 };
}

/// Uploads `desc.data`, if there is any, through a staging buffer and one
/// barrier-copy-barrier command buffer - and either way leaves `image` in
/// `shader_read_only_optimal`, the layout every draw assumes it is in.
///
/// Reuses the backend's own `command_buffer`/`fence`: safe because the
/// backend is fully synchronous, so nothing else is ever mid-flight on them.
fn uploadTexture(self: *Vk, image: vk.gen.types.Image, desc: types.TextureDesc) types.Error!void {
    const vkd = self.runtime.vkd;

    var staging_buffer: vk.gen.types.Buffer = .none;
    var staging_memory: vk.gen.types.DeviceMemory = .none;
    if (desc.data) |data| {
        const tight_size = desc.format.imageBytes(desc.width, desc.height);
        _ = vkd.createBuffer(self.runtime.device, &.{
            .size = tight_size,
            .usage = .{ .transfer_src = true },
            .sharing_mode = .exclusive,
        }, null, &staging_buffer).check() catch return error.Failed;

        var requirements: vk.gen.types.MemoryRequirements = undefined;
        vkd.getBufferMemoryRequirements(self.runtime.device, staging_buffer, &requirements);
        const type_index = memoryTypeIndex(self, requirements.memory_type_bits, .{
            .host_visible = true,
            .host_coherent = true,
        }) orelse return error.NoDevice;
        _ = vkd.allocateMemory(self.runtime.device, &.{
            .allocation_size = requirements.size,
            .memory_type_index = type_index,
        }, null, &staging_memory).check() catch return error.OutOfMemory;
        _ = vkd.bindBufferMemory(self.runtime.device, staging_buffer, staging_memory, 0).check() catch return error.Failed;

        var mapped: ?*anyopaque = null;
        _ = vkd.mapMemory(self.runtime.device, staging_memory, 0, vk.gen.types.whole_size, .{}, &mapped).check() catch return error.Failed;
        const dst: [*]u8 = @ptrCast(mapped.?);
        const pitch = desc.effectiveRowPitch();
        const tight = desc.tightRowPitch();
        var y: u32 = 0;
        while (y < desc.height) : (y += 1) {
            @memcpy(dst[y * tight ..][0..tight], data[y * pitch ..][0..tight]);
        }
        vkd.unmapMemory(self.runtime.device, staging_memory);
    }
    defer if (desc.data != null) {
        vkd.destroyBuffer(self.runtime.device, staging_buffer, null);
        vkd.freeMemory(self.runtime.device, staging_memory, null);
    };

    _ = vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
    _ = vkd.resetFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}).check() catch return error.Failed;
    _ = vkd.resetCommandBuffer(self.command_buffer, .{}).check() catch return error.Failed;
    _ = vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit = true } }).check() catch return error.Failed;

    const to_transfer = [_]vk.gen.types.ImageMemoryBarrier{.{
        .src_access_mask = .{},
        .dst_access_mask = .{ .transfer_write = true },
        .old_layout = .undefined,
        .new_layout = .transfer_dst_optimal,
        .src_queue_family_index = vk.gen.types.queue_family_ignored,
        .dst_queue_family_index = vk.gen.types.queue_family_ignored,
        .image = image,
        .subresource_range = colorRange(),
    }};
    vkd.cmdPipelineBarrier(self.command_buffer, .{ .top_of_pipe = true }, .{ .transfer = true }, .{}, 0, null, 0, null, to_transfer.len, &to_transfer);

    if (desc.data != null) {
        const region = [_]vk.gen.types.BufferImageCopy{.{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = desc.width, .height = desc.height, .depth = 1 },
        }};
        vkd.cmdCopyBufferToImage(self.command_buffer, staging_buffer, image, .transfer_dst_optimal, region.len, &region);
    }

    const to_shader_read = [_]vk.gen.types.ImageMemoryBarrier{.{
        .src_access_mask = .{ .transfer_write = true },
        .dst_access_mask = .{ .shader_read = true },
        .old_layout = .transfer_dst_optimal,
        .new_layout = .shader_read_only_optimal,
        .src_queue_family_index = vk.gen.types.queue_family_ignored,
        .dst_queue_family_index = vk.gen.types.queue_family_ignored,
        .image = image,
        .subresource_range = colorRange(),
    }};
    vkd.cmdPipelineBarrier(self.command_buffer, .{ .transfer = true }, .{ .fragment_shader = true }, .{}, 0, null, 0, null, to_shader_read.len, &to_shader_read);

    _ = vkd.endCommandBuffer(self.command_buffer).check() catch return error.Failed;
    const cmd_buffers = [_]vk.gen.types.CommandBuffer{self.command_buffer};
    const submit_info = [_]vk.gen.types.SubmitInfo{.{ .command_buffer_count = 1, .command_buffers = &cmd_buffers }};
    _ = vkd.queueSubmit(self.runtime.graphics_queue, 1, &submit_info, self.fence).check() catch return error.Failed;
    _ = vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
}

const forever: u64 = ~@as(u64, 0);

pub fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(TextureRes, native);
    self.runtime.vkd.destroyImageView(self.runtime.device, res.view, null);
    self.runtime.vkd.destroyImage(self.runtime.device, res.image, null);
    self.runtime.vkd.freeMemory(self.runtime.device, res.memory, null);
    self.gpa.destroy(res);
}

/// Out of scope for v1 - see the plan's MVP table.
pub fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) types.Error!void {
    _ = impl;
    _ = native;
    _ = region;
    _ = bytes;
    _ = row_pitch;
    _ = slice_pitch;
    return error.Unsupported;
}

/// Out of scope for v1 - see the plan's MVP table.
pub fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) types.Error![]u8 {
    _ = impl;
    _ = native;
    _ = sub;
    _ = gpa;
    return error.Unsupported;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

pub const SamplerRes = struct {
    sampler: vk.gen.types.Sampler,
};

pub fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    // `Device` already refuses `wrap == .border` and `lod_bias != 0` against
    // `caps.features`, which this backend never sets - what is left for this
    // function to refuse itself is what `Device` has no caps flag for.
    if (desc.mip_filter != .none) return error.Unsupported;
    if (desc.compare != null) return error.Unsupported;

    var sampler: vk.gen.types.Sampler = .none;
    _ = self.runtime.vkd.createSampler(self.runtime.device, &.{
        .mag_filter = vkFilter(desc.mag_filter),
        .min_filter = vkFilter(desc.min_filter),
        .mipmap_mode = .nearest,
        .address_mode_u = vkWrap(desc.wrap_u),
        .address_mode_v = vkWrap(desc.wrap_v),
        .address_mode_w = vkWrap(desc.wrap_w),
        .mip_lod_bias = 0,
        .anisotropy_enable = vk.gen.types.vk_false,
        .max_anisotropy = 1,
        .compare_enable = vk.gen.types.vk_false,
        .compare_op = .always,
        .min_lod = desc.lod_min,
        .max_lod = desc.lod_max,
        .border_color = .float_transparent_black,
        .unnormalized_coordinates = vk.gen.types.vk_false,
    }, null, &sampler).check() catch return error.Failed;

    const res = try self.gpa.create(SamplerRes);
    res.* = .{ .sampler = sampler };
    return res;
}

pub fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(SamplerRes, native);
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

// -------------------------------------------------------------------------
// Shaders
// -------------------------------------------------------------------------

pub const ShaderRes = struct {
    vertex: vk.gen.types.ShaderModule,
    fragment: vk.gen.types.ShaderModule,
};

pub fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    const words = desc.spirv orelse {
        log.writeAll("fluxion-rhi: the Vulkan backend needs ShaderDesc.spirv") catch {};
        return error.ShaderFailed;
    };

    const vertex = createModule(self, words.vertex) catch {
        log.writeAll("fluxion-rhi: vkCreateShaderModule failed for the vertex stage") catch {};
        return error.ShaderFailed;
    };
    errdefer self.runtime.vkd.destroyShaderModule(self.runtime.device, vertex, null);
    const fragment = createModule(self, words.fragment) catch {
        log.writeAll("fluxion-rhi: vkCreateShaderModule failed for the fragment stage") catch {};
        return error.ShaderFailed;
    };

    const res = try self.gpa.create(ShaderRes);
    res.* = .{ .vertex = vertex, .fragment = fragment };
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
    self.runtime.vkd.destroyShaderModule(self.runtime.device, res.vertex, null);
    self.runtime.vkd.destroyShaderModule(self.runtime.device, res.fragment, null);
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Pipelines
// -------------------------------------------------------------------------

pub const PipelineRes = struct {
    pipeline: vk.gen.types.Pipeline,
};

const max_vertex_bindings = 8;
const max_vertex_attributes = 16;

pub fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader_native: backend.Native, log: *Io.Writer) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    if (desc.depth_format != null) {
        log.writeAll("fluxion-rhi: the Vulkan backend has no depth attachments yet") catch {};
        return error.Unsupported;
    }
    if (desc.extra_color_formats.len > 0) {
        log.writeAll("fluxion-rhi: the Vulkan backend has no multiple render targets yet") catch {};
        return error.Unsupported;
    }
    if (desc.samples != 1) {
        log.writeAll("fluxion-rhi: the Vulkan backend has no multisampling yet") catch {};
        return error.Unsupported;
    }
    const color_format = desc.color_format orelse return error.Unsupported;
    const vk_format = vulkan.toVkFormat(color_format) orelse return error.Unsupported;
    if (desc.buffers.len > max_vertex_bindings or desc.attributes.len > max_vertex_attributes) return error.Unsupported;

    const shader = as(ShaderRes, shader_native);

    var bindings: [max_vertex_bindings]vk.gen.types.VertexInputBindingDescription = undefined;
    for (desc.buffers, 0..) |buf, i| bindings[i] = .{
        .binding = @intCast(i),
        .stride = buf.stride,
        .input_rate = if (buf.step == .instance) .instance else .vertex,
    };
    var attrs: [max_vertex_attributes]vk.gen.types.VertexInputAttributeDescription = undefined;
    for (desc.attributes, 0..) |attr, i| attrs[i] = .{
        .location = attr.location,
        .binding = attr.buffer,
        .format = vertexFormat(attr.format),
        .offset = attr.offset,
    };

    const vertex_input: vk.gen.types.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = @intCast(desc.buffers.len),
        .vertex_binding_descriptions = if (desc.buffers.len > 0) &bindings else null,
        .vertex_attribute_description_count = @intCast(desc.attributes.len),
        .vertex_attribute_descriptions = if (desc.attributes.len > 0) &attrs else null,
    };

    const input_assembly: vk.gen.types.PipelineInputAssemblyStateCreateInfo = .{
        .topology = vkTopology(desc.topology),
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
        .cull_mode = switch (desc.cull) {
            .none => .{},
            .back => .{ .back = true },
            .front => .{ .front = true },
        },
        .front_face = if (desc.front_face == .ccw) .counter_clockwise else .clockwise,
        .depth_bias_enable = vk.gen.types.vk_false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const multisample: vk.gen.types.PipelineMultisampleStateCreateInfo = .{
        .rasterization_samples = .{ .x1 = true },
        .sample_shading_enable = vk.gen.types.vk_false,
        .min_sample_shading = 0,
        .alpha_to_coverage_enable = vk.gen.types.vk_false,
        .alpha_to_one_enable = vk.gen.types.vk_false,
    };

    const blend_attachment = [_]vk.gen.types.PipelineColorBlendAttachmentState{.{
        .blend_enable = if (desc.blend.enabled) vk.gen.types.vk_true else vk.gen.types.vk_false,
        .src_color_blend_factor = vkBlendFactor(desc.blend.src_rgb),
        .dst_color_blend_factor = vkBlendFactor(desc.blend.dst_rgb),
        .color_blend_op = vkBlendOp(desc.blend.op_rgb),
        .src_alpha_blend_factor = vkBlendFactor(desc.blend.src_alpha),
        .dst_alpha_blend_factor = vkBlendFactor(desc.blend.dst_alpha),
        .alpha_blend_op = vkBlendOp(desc.blend.op_alpha),
        .color_write_mask = .{ .r = true, .g = true, .b = true, .a = true },
    }};
    const color_blend: vk.gen.types.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = vk.gen.types.vk_false,
        .logic_op = .copy,
        .attachment_count = blend_attachment.len,
        .attachments = &blend_attachment,
        .blend_constants = .{ 0, 0, 0, 0 },
    };

    const dynamic_states = [_]vk.gen.types.DynamicState{ .viewport, .scissor };
    const dynamic_state: vk.gen.types.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .dynamic_states = &dynamic_states,
    };

    // Only used to satisfy `vkCreateGraphicsPipelines`' render-pass argument:
    // any render pass whose attachments are format/sample compatible works
    // at `vkCmdBeginRenderPass` time, and every render pass this backend
    // makes for one format has the same one colour attachment. `.clear` is
    // an arbitrary, always-cached choice - the load op does not affect
    // compatibility.
    const render_pass = swapchain.getRenderPass(self, vk_format, .clear) catch {
        log.writeAll("fluxion-rhi: could not make a render pass for this pipeline's color_format") catch {};
        return error.PipelineFailed;
    };

    const stages = [_]vk.gen.types.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex = true }, .module = shader.vertex, .name = "main" },
        .{ .stage = .{ .fragment = true }, .module = shader.fragment, .name = "main" },
    };

    const pipeline_info = [_]vk.gen.types.GraphicsPipelineCreateInfo{.{
        .stage_count = stages.len,
        .stages = &stages,
        .vertex_input_state = &vertex_input,
        .input_assembly_state = &input_assembly,
        .viewport_state = &viewport_state,
        .rasterization_state = &rasterization,
        .multisample_state = &multisample,
        .color_blend_state = &color_blend,
        .dynamic_state = &dynamic_state,
        .layout = self.pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_index = -1,
    }};

    var pipeline: vk.gen.types.Pipeline = .none;
    _ = self.runtime.vkd.createGraphicsPipelines(self.runtime.device, .none, 1, &pipeline_info, null, @ptrCast(&pipeline)).check() catch {
        log.writeAll("fluxion-rhi: vkCreateGraphicsPipelines failed") catch {};
        return error.PipelineFailed;
    };

    const res = try self.gpa.create(PipelineRes);
    res.* = .{ .pipeline = pipeline };
    return res;
}

pub fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(PipelineRes, native);
    self.runtime.vkd.destroyPipeline(self.runtime.device, res.pipeline, null);
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
