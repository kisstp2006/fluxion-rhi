// SPDX-License-Identifier: BSD-2-Clause

//! The surface half of the Vulkan backend: swapchain, image views,
//! framebuffers, and the render-pass cache `vulkan.zig`'s `submit` and
//! `vulkan_resources.zig`'s `createPipeline` both read from.
//!
//! **A `VkRenderPass` bakes its load op in**, unlike a pipeline's viewport -
//! so a pass that clears and a pass that loads need two different render
//! pass *objects*, even though they are "the same" render pass to everything
//! else: same one colour attachment, same format, same sample count. Vulkan
//! calls that "compatible", and a compatible render pass is all
//! `vkCreateGraphicsPipelines` or a framebuffer ever needs - so `getRenderPass`
//! caches one object per `(format, load op)` pair and every framebuffer this
//! file makes is happy to begin with any of them.

const std = @import("std");
const Allocator = std.mem.Allocator;

const vk = @import("fluxion_vulkan");
const types = @import("../types.zig");
const backend = @import("../backend.zig");

const vulkan = @import("vulkan.zig");
const Vk = vulkan.Vk;

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

const forever: u64 = ~@as(u64, 0);

/// At most this many images or formats a driver reports - generous for any
/// real swapchain (two to four images, one to a handful of formats), and a
/// fixed bound keeps this file's swapchain rebuilding allocation-free.
const max_swapchain_images = 16;
const max_surface_formats = 32;

pub const SurfaceRes = struct {
    surface: vk.gen.types.SurfaceKHR,
    swapchain: vk.gen.types.SwapchainKHR = .none,
    format: vk.gen.types.Format = .undefined,
    width: u32 = 0,
    height: u32 = 0,
    views: []vk.gen.types.ImageView = &.{},
    framebuffers: []vk.gen.types.Framebuffer = &.{},
    /// The image `acquire` last chose; what `present` presents.
    image_index: u32 = 0,
};

pub fn createSurface(impl: backend.Impl, desc: types.SurfaceDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    if (desc.vulkan_surface == 0) return error.InvalidArgument;
    const surface: vk.gen.types.SurfaceKHR = @enumFromInt(desc.vulkan_surface);

    const res = try self.gpa.create(SurfaceRes);
    errdefer self.gpa.destroy(res);
    res.* = .{ .surface = surface };
    errdefer destroySwapchainObjects(self, res);

    try buildSwapchain(self, res, desc.width, desc.height, .none);
    return res;
}

pub fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    _ = self.runtime.vkd.deviceWaitIdle(self.runtime.device);
    destroySwapchainObjects(self, res);
    if (self.runtime.vki.destroySurfaceKHR) |destroy| destroy(self.runtime.instance, res.surface, null);
    self.gpa.destroy(res);
}

pub fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) types.Error!void {
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    _ = self.runtime.vkd.deviceWaitIdle(self.runtime.device);
    destroySwapchainViews(self, res);
    try buildSwapchain(self, res, width, height, res.swapchain);
}

pub fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

/// Acquires the next image, if this surface has not already had one
/// acquired into it this `submit`. Called from `vulkan.zig`'s `beginPass`.
pub fn acquire(self: *Vk, res: *SurfaceRes) types.Error!void {
    const acquireFn = self.runtime.vkd.acquireNextImageKHR orelse return error.Unsupported;
    var index: u32 = 0;
    _ = acquireFn(self.runtime.device, res.swapchain, forever, self.sem_image_available, .none, &index).check() catch |err| switch (err) {
        error.DeviceLost => return error.DeviceLost,
        else => return error.Failed,
    };
    res.image_index = index;
}

/// `vsync` is not read: `buildSwapchain` always picks `.fifo`, which is
/// vsync-on and the one present mode every implementation has - see the
/// plan's MVP scope.
pub fn present(impl: backend.Impl, native: backend.Native, vsync: bool) types.Error!void {
    _ = vsync;
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    const presentFn = self.runtime.vkd.queuePresentKHR orelse return error.Unsupported;

    const swapchains = [_]vk.gen.types.SwapchainKHR{res.swapchain};
    const indices = [_]u32{res.image_index};
    // `submit` already waited on its fence before this is ever called, so
    // the GPU has finished drawing - `sem_render_finished` is only here
    // because `vkQueuePresentKHR` wants a wait semaphore to check, and it is
    // already signalled by the time it is checked.
    const wait_semaphores = [_]vk.gen.types.Semaphore{self.sem_render_finished};
    _ = presentFn(self.runtime.graphics_queue, &.{
        .wait_semaphore_count = wait_semaphores.len,
        .wait_semaphores = &wait_semaphores,
        .swapchain_count = swapchains.len,
        .swapchains = &swapchains,
        .image_indices = &indices,
    }).check() catch |err| switch (err) {
        error.DeviceLost => return error.DeviceLost,
        else => return error.Failed,
    };

    // Fully synchronous, like every other step: block until the present -
    // and everything queued before it - has actually happened, so the next
    // frame's `submit` never overlaps this one.
    _ = self.runtime.vkd.queueWaitIdle(self.runtime.graphics_queue).check() catch return error.DeviceLost;
}

pub fn mapLoadOp(load: types.LoadOp) vk.gen.types.AttachmentLoadOp {
    return switch (load) {
        .clear => .clear,
        .load => .load,
        .dont_care => .dont_care,
    };
}

// -------------------------------------------------------------------------
// Building and tearing down the swapchain itself
// -------------------------------------------------------------------------

/// Creates (or, with `old != .none`, recreates in place) `res`'s swapchain,
/// image views and framebuffers. `res.surface` and `res.swapchain` (as
/// `old`) are the only fields read; every other field is written.
fn buildSwapchain(self: *Vk, res: *SurfaceRes, want_width: u32, want_height: u32, old: vk.gen.types.SwapchainKHR) types.Error!void {
    const vki = self.runtime.vki;
    const vkd = self.runtime.vkd;
    const phys = self.runtime.physical_device;

    var caps: vk.gen.types.SurfaceCapabilitiesKHR = undefined;
    const getCaps = vki.getPhysicalDeviceSurfaceCapabilitiesKHR orelse return error.Unsupported;
    _ = getCaps(phys, res.surface, &caps).check() catch return error.Failed;

    const getFormats = vki.getPhysicalDeviceSurfaceFormatsKHR orelse return error.Unsupported;
    var format_count: u32 = 0;
    _ = getFormats(phys, res.surface, &format_count, null).check() catch return error.Failed;
    if (format_count == 0 or format_count > max_surface_formats) return error.Unsupported;
    var format_buf: [max_surface_formats]vk.gen.types.SurfaceFormatKHR = undefined;
    _ = getFormats(phys, res.surface, &format_count, &format_buf).check() catch return error.Failed;
    const formats = format_buf[0..format_count];

    var chosen_format = formats[0].format;
    var chosen_space = formats[0].color_space;
    for (formats) |f| {
        if ((f.format == .b8g8r8a8_unorm or f.format == .r8g8b8a8_unorm) and f.color_space == .srgb_nonlinear) {
            chosen_format = f.format;
            chosen_space = f.color_space;
            break;
        }
    }
    if (vulkan.fromVkFormat(chosen_format) == null) return error.Unsupported;

    // FIFO is required by the spec on every implementation, and is what
    // `vsync = true` means everywhere else in this library.
    const present_mode: vk.gen.types.PresentModeKHR = .fifo;

    const extent: vk.gen.types.Extent2D = if (caps.current_extent.width != 0xFFFF_FFFF)
        caps.current_extent
    else .{
        .width = std.math.clamp(if (want_width != 0) want_width else 1, caps.min_image_extent.width, @max(1, caps.max_image_extent.width)),
        .height = std.math.clamp(if (want_height != 0) want_height else 1, caps.min_image_extent.height, @max(1, caps.max_image_extent.height)),
    };
    if (extent.width == 0 or extent.height == 0) return error.Unsupported;

    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count != 0 and image_count > caps.max_image_count) image_count = caps.max_image_count;

    const usage: vk.gen.types.ImageUsageFlags = .{ .color_attachment = true };
    if (!caps.supported_usage_flags.contains(usage)) return error.Unsupported;

    const composite_alpha: vk.gen.types.CompositeAlphaFlagsKHR = if (caps.supported_composite_alpha.@"opaque")
        .{ .@"opaque" = true }
    else
        caps.supported_composite_alpha;

    var swapchain: vk.gen.types.SwapchainKHR = .none;
    const create = vkd.createSwapchainKHR orelse return error.Unsupported;
    _ = create(self.runtime.device, &.{
        .surface = res.surface,
        .min_image_count = image_count,
        .image_format = chosen_format,
        .image_color_space = chosen_space,
        .image_extent = extent,
        .image_array_layers = 1,
        .image_usage = usage,
        .image_sharing_mode = .exclusive,
        .pre_transform = caps.current_transform,
        .composite_alpha = composite_alpha,
        .present_mode = present_mode,
        .clipped = vk.gen.types.vk_true,
        .old_swapchain = old,
    }, null, &swapchain).check() catch return error.Failed;
    errdefer if (vkd.destroySwapchainKHR) |destroy| destroy(self.runtime.device, swapchain, null);

    // The old swapchain is retired, not destroyed, by `vkCreateSwapchainKHR`
    // taking it as `oldSwapchain` - it is safe to destroy only now, after
    // the new one exists, and only here.
    if (old != .none) {
        if (vkd.destroySwapchainKHR) |destroy| destroy(self.runtime.device, old, null);
    }

    var image_count_actual: u32 = 0;
    const getImages = vkd.getSwapchainImagesKHR orelse return error.Unsupported;
    _ = getImages(self.runtime.device, swapchain, &image_count_actual, null).check() catch return error.Failed;
    if (image_count_actual == 0 or image_count_actual > max_swapchain_images) return error.Unsupported;
    var images_buf: [max_swapchain_images]vk.gen.types.Image = undefined;
    _ = getImages(self.runtime.device, swapchain, &image_count_actual, &images_buf).check() catch return error.Failed;
    const images = images_buf[0..image_count_actual];

    const render_pass = try getRenderPass(self, chosen_format, .clear);

    const views = try self.gpa.alloc(vk.gen.types.ImageView, images.len);
    errdefer self.gpa.free(views);
    const framebuffers = try self.gpa.alloc(vk.gen.types.Framebuffer, images.len);
    errdefer self.gpa.free(framebuffers);

    for (images, 0..) |image, i| {
        _ = vkd.createImageView(self.runtime.device, &.{
            .image = image,
            .view_type = .@"2d",
            .format = chosen_format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .color = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        }, null, &views[i]).check() catch return error.Failed;

        const attachments = [_]vk.gen.types.ImageView{views[i]};
        _ = vkd.createFramebuffer(self.runtime.device, &.{
            .render_pass = render_pass,
            .attachment_count = attachments.len,
            .attachments = &attachments,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        }, null, &framebuffers[i]).check() catch return error.Failed;
    }

    res.swapchain = swapchain;
    res.format = chosen_format;
    res.width = extent.width;
    res.height = extent.height;
    res.views = views;
    res.framebuffers = framebuffers;
    res.image_index = 0;
}

fn destroySwapchainViews(self: *Vk, res: *SurfaceRes) void {
    for (res.framebuffers) |fb| self.runtime.vkd.destroyFramebuffer(self.runtime.device, fb, null);
    for (res.views) |view| self.runtime.vkd.destroyImageView(self.runtime.device, view, null);
    self.gpa.free(res.framebuffers);
    self.gpa.free(res.views);
    res.framebuffers = &.{};
    res.views = &.{};
}

fn destroySwapchainObjects(self: *Vk, res: *SurfaceRes) void {
    destroySwapchainViews(self, res);
    if (res.swapchain != .none) {
        if (self.runtime.vkd.destroySwapchainKHR) |destroy| destroy(self.runtime.device, res.swapchain, null);
    }
}

// -------------------------------------------------------------------------
// The render pass cache
// -------------------------------------------------------------------------

pub const RenderPassEntry = struct {
    format: vk.gen.types.Format,
    load: vk.gen.types.AttachmentLoadOp,
    pass: vk.gen.types.RenderPass,
};

/// The render pass for one `(format, load op)` pair, making it the first
/// time it is asked for. `vulkan.zig`'s `Vk.render_passes` is the cache;
/// this is the only function that reads or writes it.
pub fn getRenderPass(self: *Vk, format: vk.gen.types.Format, load: vk.gen.types.AttachmentLoadOp) types.Error!vk.gen.types.RenderPass {
    for (self.render_passes) |maybe| {
        if (maybe) |entry| {
            if (entry.format == format and entry.load == load) return entry.pass;
        }
    }

    var free_index: ?usize = null;
    for (self.render_passes, 0..) |maybe, i| {
        if (maybe == null) {
            free_index = i;
            break;
        }
    }
    const index = free_index orelse return error.OutOfMemory;

    const attachment = [_]vk.gen.types.AttachmentDescription{.{
        .format = format,
        .samples = .{ .x1 = true },
        .load_op = load,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = if (load == .load) .present_src_khr else .undefined,
        .final_layout = .present_src_khr,
    }};
    const color_ref = [_]vk.gen.types.AttachmentReference{.{ .attachment = 0, .layout = .color_attachment_optimal }};
    const subpass = [_]vk.gen.types.SubpassDescription{.{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = color_ref.len,
        .color_attachments = &color_ref,
    }};

    // Without this, the implicit external dependency Vulkan adds by default
    // waits at TOP_OF_PIPE - a stage `sem_image_available` does not gate -
    // so the attachment's layout transition could start before the image is
    // actually acquired. Tying it to COLOR_ATTACHMENT_OUTPUT, the same stage
    // the semaphore blocks, closes that race.
    const dependency = [_]vk.gen.types.SubpassDependency{.{
        .src_subpass = vk.gen.types.subpass_external,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output = true },
        .dst_stage_mask = .{ .color_attachment_output = true },
        .dst_access_mask = .{ .color_attachment_write = true },
    }};

    var render_pass: vk.gen.types.RenderPass = .none;
    _ = self.runtime.vkd.createRenderPass(self.runtime.device, &.{
        .attachment_count = attachment.len,
        .attachments = &attachment,
        .subpass_count = subpass.len,
        .subpasses = &subpass,
        .dependency_count = dependency.len,
        .dependencies = &dependency,
    }, null, &render_pass).check() catch return error.Failed;

    self.render_passes[index] = .{ .format = format, .load = load, .pass = render_pass };
    return render_pass;
}
