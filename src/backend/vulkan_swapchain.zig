// SPDX-License-Identifier: BSD-2-Clause

//! The surface half of the Vulkan backend: swapchain, image views,
//! framebuffers, and the render-pass cache `vulkan.zig`'s `submit` and
//! `vulkan_resources.zig`'s textures and pipelines all read from.
//!
//! **A `VkRenderPass` bakes its load op in**, and the layouts its attachment
//! starts and ends in, unlike a pipeline's viewport - so a pass that clears
//! and a pass that loads need two different render pass *objects*, even
//! though they are "the same" render pass to everything else: same one colour
//! attachment, same format, same sample count. Vulkan calls that
//! "compatible", and a compatible render pass is all `vkCreateGraphicsPipelines`
//! or a framebuffer ever needs - so `getRenderPass` caches one object per
//! `(format, load op, layouts)` and every framebuffer of a format is happy to
//! begin with any of them.
//!
//! **One image per frame.** A surface's image is acquired by the first pass
//! into it after a `present` and kept, whatever number of submits draw into
//! it, until the next `present` shows it. Every submit has waited for the GPU
//! before it returns, so `present` needs no semaphore of its own.

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
const max_present_modes = 8;

pub const SurfaceRes = struct {
    surface: vk.gen.types.SurfaceKHR,
    swapchain: vk.gen.types.SwapchainKHR = .none,
    format: vk.gen.types.Format = .undefined,
    width: u32 = 0,
    height: u32 = 0,
    views: []vk.gen.types.ImageView = &.{},
    framebuffers: []vk.gen.types.Framebuffer = &.{},
    /// Signalled by the acquire; the submit that acquired waits on it.
    acquired_signal: vk.gen.types.Semaphore = .none,
    /// The image a pass acquired this frame, which `present` shows; null
    /// once it has.
    acquired: ?u32 = null,
    /// Whether each image has been drawn into since the swapchain was made:
    /// until it has, its layout is undefined rather than presentable.
    drawn: [max_swapchain_images]bool = @splat(false),
    /// Whether the swapchain was made presenting with vsync, and what the
    /// last `present` asked for: the next acquire makes it again when they
    /// differ.
    vsync: bool = true,
    want_vsync: bool = true,
    /// Told by an acquire or a present that it no longer fits the window:
    /// made again at the next acquire.
    stale: bool = false,
};

pub fn createSurface(impl: backend.Impl, desc: types.SurfaceDesc) types.Error!backend.Native {
    const self = vulkan.cast(impl);
    // The caller's own, or one the window makes from this device's instance.
    const made = if (desc.vulkan_surface != 0) desc.vulkan_surface else if (desc.window) |window|
        window.make_vulkan_surface(window.context, @intFromPtr(self.runtime.instance), @ptrCast(self.runtime.loader.getInstanceProcAddr)) orelse return error.Unsupported
    else
        return error.InvalidArgument;
    const surface: vk.gen.types.SurfaceKHR = @enumFromInt(made);
    errdefer if (self.runtime.vki.destroySurfaceKHR) |destroy| destroy(self.runtime.instance, surface, null);

    const res = try self.gpa.create(SurfaceRes);
    errdefer self.gpa.destroy(res);
    res.* = .{ .surface = surface, .vsync = desc.vsync, .want_vsync = desc.vsync };
    _ = self.runtime.vkd.createSemaphore(self.runtime.device, &.{}, null, &res.acquired_signal).check() catch return error.Failed;
    errdefer self.runtime.vkd.destroySemaphore(self.runtime.device, res.acquired_signal, null);
    errdefer destroySwapchainObjects(self, res);

    try buildSwapchain(self, res, desc.width, desc.height, .none);
    return res;
}

pub fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    _ = self.runtime.vkd.deviceWaitIdle(self.runtime.device);
    destroySwapchainObjects(self, res);
    self.runtime.vkd.destroySemaphore(self.runtime.device, res.acquired_signal, null);
    if (self.runtime.vki.destroySurfaceKHR) |destroy| destroy(self.runtime.instance, res.surface, null);
    self.gpa.destroy(res);
}

pub fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) types.Error!void {
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    try rebuild(self, res, width, height);
}

pub fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

/// Make the swapchain again, for a new size or present mode. An image
/// acquired and not yet shown is given up.
fn rebuild(self: *Vk, res: *SurfaceRes, width: u32, height: u32) types.Error!void {
    _ = self.runtime.vkd.deviceWaitIdle(self.runtime.device);
    destroySwapchainViews(self, res);
    res.acquired = null;
    try buildSwapchain(self, res, width, height, res.swapchain);
}

/// The image this frame draws into: acquired by the first pass into the
/// surface since the last `present`, and the same one after. True when this
/// call acquired it, which the submit it is in then waits for.
pub fn acquire(self: *Vk, res: *SurfaceRes) types.Error!bool {
    if (res.acquired != null) return false;
    if (res.stale or res.vsync != res.want_vsync) try rebuild(self, res, res.width, res.height);
    const acquireFn = self.runtime.vkd.acquireNextImageKHR orelse return error.Unsupported;
    var tries: u32 = 0;
    while (true) : (tries += 1) {
        var index: u32 = 0;
        const result = acquireFn(self.runtime.device, res.swapchain, forever, res.acquired_signal, .none, &index).check() catch |err| switch (err) {
            error.OutOfDate => if (tries == 0) {
                try rebuild(self, res, res.width, res.height);
                continue;
            } else return error.Failed,
            error.DeviceLost => return error.DeviceLost,
            else => return error.Failed,
        };
        if (result == .suboptimal_khr) res.stale = true;
        res.acquired = index;
        return true;
    }
}

/// Shows the image this frame drew into. A frame that drew nothing into the
/// surface has nothing to show, and shows nothing. `vsync` takes effect at
/// the next frame's acquire.
pub fn present(impl: backend.Impl, native: backend.Native, vsync: bool) types.Error!void {
    const self = vulkan.cast(impl);
    const res = as(SurfaceRes, native);
    res.want_vsync = vsync;
    const index = res.acquired orelse return;
    res.acquired = null;
    const presentFn = self.runtime.vkd.queuePresentKHR orelse return error.Unsupported;

    const swapchains = [_]vk.gen.types.SwapchainKHR{res.swapchain};
    const indices = [_]u32{index};
    // Every submit that drew into the image waited for its fence before it
    // returned: the drawing is done, and no semaphore is needed to say so.
    const result = presentFn(self.runtime.graphics_queue, &.{
        .wait_semaphore_count = 0,
        .wait_semaphores = null,
        .swapchain_count = swapchains.len,
        .swapchains = &swapchains,
        .image_indices = &indices,
    }).check() catch |err| switch (err) {
        error.DeviceLost => return error.DeviceLost,
        error.OutOfDate => {
            res.stale = true;
            return;
        },
        else => return error.Failed,
    };
    if (result == .suboptimal_khr) res.stale = true;

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
/// image views and framebuffers. `res.surface`, `res.want_vsync` and
/// `res.swapchain` (as `old`) are the fields read; the rest are written.
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

    // RGBA first, the format pipelines are made for when they say nothing;
    // BGRA when that is all there is, which a pipeline then meets with a
    // variant of its own.
    var chosen: ?vk.gen.types.SurfaceFormatKHR = null;
    for ([_]vk.gen.types.Format{ .r8g8b8a8_unorm, .b8g8r8a8_unorm }) |wanted| {
        if (chosen != null) break;
        for (formats) |f| if (f.format == wanted and f.color_space == .srgb_nonlinear) {
            chosen = f;
            break;
        };
    }
    const chosen_format = (chosen orelse return error.Unsupported).format;
    const chosen_space = chosen.?.color_space;

    // FIFO is required by the spec on every implementation, and is what
    // `vsync = true` means everywhere else in this library. Without vsync,
    // mailbox - never tearing - and then immediate, when there are.
    var present_mode: vk.gen.types.PresentModeKHR = .fifo;
    if (!res.want_vsync) {
        if (vki.getPhysicalDeviceSurfacePresentModesKHR) |getModes| {
            var mode_count: u32 = max_present_modes;
            var mode_buf: [max_present_modes]vk.gen.types.PresentModeKHR = undefined;
            if (getModes(phys, res.surface, &mode_count, &mode_buf).check()) |_| {
                const modes = mode_buf[0..mode_count];
                if (std.mem.indexOfScalar(vk.gen.types.PresentModeKHR, modes, .mailbox) != null) {
                    present_mode = .mailbox;
                } else if (std.mem.indexOfScalar(vk.gen.types.PresentModeKHR, modes, .immediate) != null) {
                    present_mode = .immediate;
                }
            } else |_| {}
        }
    }

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
        res.swapchain = .none;
    }

    var image_count_actual: u32 = 0;
    const getImages = vkd.getSwapchainImagesKHR orelse return error.Unsupported;
    _ = getImages(self.runtime.device, swapchain, &image_count_actual, null).check() catch return error.Failed;
    if (image_count_actual == 0 or image_count_actual > max_swapchain_images) return error.Unsupported;
    var images_buf: [max_swapchain_images]vk.gen.types.Image = undefined;
    _ = getImages(self.runtime.device, swapchain, &image_count_actual, &images_buf).check() catch return error.Failed;
    const images = images_buf[0..image_count_actual];

    const render_pass = try getRenderPass(self, chosen_format, .clear, .undefined, .present_src_khr);

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
    res.acquired = null;
    res.drawn = @splat(false);
    res.vsync = res.want_vsync;
    res.stale = false;
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
        res.swapchain = .none;
    }
}

// -------------------------------------------------------------------------
// The render pass cache
// -------------------------------------------------------------------------

pub const RenderPassEntry = struct {
    format: vk.gen.types.Format,
    load: vk.gen.types.AttachmentLoadOp,
    initial: vk.gen.types.ImageLayout,
    final: vk.gen.types.ImageLayout,
    pass: vk.gen.types.RenderPass,
};

/// The render pass for one `(format, load op, initial layout, final layout)`,
/// making it the first time it is asked for. `vulkan.zig`'s
/// `Vk.render_passes` is the cache; this is the only function that reads or
/// writes it.
///
/// A pass that ends in `shader_read_only_optimal` draws into a texture that
/// is sampled after: its dependencies make the draws of this command buffer,
/// and of the submits before it, that read the texture finish before it is
/// written, and the writes land before anything after reads them. One that
/// ends in `present_src_khr` draws into a swapchain image.
pub fn getRenderPass(
    self: *Vk,
    format: vk.gen.types.Format,
    load: vk.gen.types.AttachmentLoadOp,
    initial: vk.gen.types.ImageLayout,
    final: vk.gen.types.ImageLayout,
) types.Error!vk.gen.types.RenderPass {
    for (self.render_passes) |maybe| {
        if (maybe) |entry| {
            if (entry.format == format and entry.load == load and entry.initial == initial and entry.final == final) return entry.pass;
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
        .initial_layout = initial,
        .final_layout = final,
    }};
    const color_ref = [_]vk.gen.types.AttachmentReference{.{ .attachment = 0, .layout = .color_attachment_optimal }};
    const subpass = [_]vk.gen.types.SubpassDescription{.{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = color_ref.len,
        .color_attachments = &color_ref,
    }};

    // The same two for every render pass, because two passes whose
    // dependencies differ are not compatible - and a pipeline is made with
    // one and drawn with any. Into the pass: what earlier draws wrote and read
    // is done first - including, for a swapchain image, the acquire, whose
    // semaphore gates COLOR_ATTACHMENT_OUTPUT, where the attachment's layout
    // transition is then tied rather than at TOP_OF_PIPE. Out of it: what it
    // wrote lands before a later pass loads it or a shader samples it.
    const dependencies = [_]vk.gen.types.SubpassDependency{
        .{
            .src_subpass = vk.gen.types.subpass_external,
            .dst_subpass = 0,
            .src_stage_mask = .{ .fragment_shader = true, .color_attachment_output = true },
            .dst_stage_mask = .{ .color_attachment_output = true },
            .src_access_mask = .{ .color_attachment_write = true },
            .dst_access_mask = .{ .color_attachment_read = true, .color_attachment_write = true },
        },
        .{
            .src_subpass = 0,
            .dst_subpass = vk.gen.types.subpass_external,
            .src_stage_mask = .{ .color_attachment_output = true },
            .dst_stage_mask = .{ .fragment_shader = true, .color_attachment_output = true },
            .src_access_mask = .{ .color_attachment_write = true },
            .dst_access_mask = .{ .shader_read = true, .color_attachment_read = true, .color_attachment_write = true },
        },
    };

    var render_pass: vk.gen.types.RenderPass = .none;
    _ = self.runtime.vkd.createRenderPass(self.runtime.device, &.{
        .attachment_count = attachment.len,
        .attachments = &attachment,
        .subpass_count = subpass.len,
        .subpasses = &subpass,
        .dependency_count = dependencies.len,
        .dependencies = &dependencies,
    }, null, &render_pass).check() catch return error.Failed;

    self.render_passes[index] = .{ .format = format, .load = load, .initial = initial, .final = final, .pass = render_pass };
    return render_pass;
}
