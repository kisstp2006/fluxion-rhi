// SPDX-License-Identifier: BSD-2-Clause

//! The Vulkan backend: `Runtime` (loader, instance, physical device, logical
//! device, graphics queue) plus a real `Vtable` over it.
//!
//! Three files split the work `backend.Vtable` asks for:
//!
//!   `vulkan.zig`            this file - `Runtime`, the backend's own state
//!                            (`Vk`), opening, capabilities, and `submit`,
//!                            which is the one place a `commands.Command`
//!                            list becomes real `vkCmd*` calls.
//!   `vulkan_resources.zig`  buffers, textures, samplers, shaders, pipelines.
//!   `vulkan_swapchain.zig`  the surface: swapchain, image views,
//!                            framebuffers, the render-pass cache, acquire
//!                            and present.
//!
//! **Fully synchronous.** One command buffer, one fence, no frames in
//! flight: `submit` records, submits, and blocks on the fence before
//! returning, and so does every upload and readback. Slower than
//! double-buffering, and unambiguously correct.
//!
//! **Up is up, as on Direct3D.** Every viewport is given to Vulkan with a
//! negative height (`VK_KHR_maintenance1`), so clip-space Y points up the
//! picture and a pass draws the top of it into the first row of its
//! target - the Direct3D convention, `Backend.clip` `.d3d`. A shader written
//! for either runs here the same way up, and a texture drawn into is sampled
//! with the coordinates of one that was uploaded.
//!
//! **One shared pipeline layout.** Every pipeline this backend makes uses
//! the same two descriptor sets - `set 0` four uniform-buffer slots, `set 1`
//! four combined-image-sampler slots - so `setUniformBuffer`/`setTexture`
//! and a draw never have to know which pipeline is bound. Descriptor pools
//! reset once per `submit`, and a pair of sets allocated fresh whenever a
//! binding actually changes (`flushBindings`), is what makes that safe: a
//! descriptor set's *content* is whatever the last `vkUpdateDescriptorSets`
//! on it wrote, checked only when the GPU executes the draw that reads it -
//! reusing one set across draws with different textures in the same list
//! would silently make every earlier draw use the last texture, not the one
//! it was recorded with. A pool that runs out is followed by another.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const vk = @import("fluxion_vulkan");
const types = @import("../types.zig");
const commands = @import("../commands.zig");
const backend = @import("../backend.zig");
const Device = @import("../Device.zig");

const resources = @import("vulkan_resources.zig");
const swapchain = @import("vulkan_swapchain.zig");

pub const BufferRes = resources.BufferRes;
pub const TextureRes = resources.TextureRes;
pub const SamplerRes = resources.SamplerRes;
pub const ShaderRes = resources.ShaderRes;
pub const PipelineRes = resources.PipelineRes;
pub const SurfaceRes = swapchain.SurfaceRes;

/// `set 0`'s width: four uniform-buffer bindings.
pub const uniform_slots = 4;
/// `set 1`'s width: four combined-image-sampler bindings.
pub const texture_slots = 4;

/// How many `(uniform, texture)` descriptor set pairs one pool holds. A
/// submit that changes its bindings more often than that takes another pool;
/// every pool is reset at the next submit.
const binding_states_per_pool = 256;

/// How many distinct `(format, load op, layouts)` render passes
/// `getRenderPass` caches: three formats, three load ops, and a swapchain's
/// and a texture's layouts come to well under this.
const max_render_passes = 32;

/// The most surfaces one submit can draw into: each acquires its image, and
/// the submit waits for every one it acquired.
const max_surfaces_per_submit = 4;

// -------------------------------------------------------------------------
// Format mapping
// -------------------------------------------------------------------------

/// The colour formats this backend makes textures and targets of -
/// everything else is `null`, so `Device`'s caps-driven pre-validation (from
/// `caps` below never marking another format `sampled`) is what actually
/// keeps requests for the rest from ever reaching this backend.
pub fn toVkFormat(format: types.Format) ?vk.gen.types.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        .r8_unorm => .r8_unorm,
        else => null,
    };
}

pub fn fromVkFormat(format: vk.gen.types.Format) ?types.Format {
    return switch (format) {
        .r8g8b8a8_unorm => .rgba8_unorm,
        .b8g8r8a8_unorm => .bgra8_unorm,
        .r8_unorm => .r8_unorm,
        else => null,
    };
}

// -------------------------------------------------------------------------
// Runtime: loader, instance, physical device, logical device, queue
// -------------------------------------------------------------------------

/// The long-lived Vulkan objects a rendering backend is built on. They are
/// deliberately owned in destruction order: device, instance, then loader.
pub const Runtime = struct {
    loader: vk.Loader,
    instance: vk.Instance,
    instance_commands: vk.InstanceCommands,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    device_commands: vk.DeviceCommands,
    graphics_queue: vk.Queue,
    graphics_family: u32,
    /// Every command a resource, a swapchain or a command buffer needs -
    /// `instance_commands`/`device_commands` only have what the loader
    /// itself uses to get this far.
    vki: vk.gen.commands.Instance,
    vkd: vk.gen.commands.Device,
    memory_properties: vk.gen.types.PhysicalDeviceMemoryProperties,

    /// Opens a Vulkan execution context that can present: the instance asks
    /// for `VK_KHR_surface` and every `VK_KHR_*_surface` extension the
    /// loader actually has (so this needs no platform `#ifdef` of its own -
    /// whichever the platform is, its extension is either there or is not
    /// asked for), and the device requires `VK_KHR_swapchain` and
    /// `VK_KHR_maintenance1`, whose negative viewport height turns the
    /// picture the Direct3D way up. With `debug`, the Khronos validation
    /// layer, when it is installed: what it finds goes to the standard
    /// output.
    pub fn open(gpa: Allocator, debug: bool) types.Error!Runtime {
        var loader = vk.Loader.init() catch return error.NoDevice;
        errdefer loader.deinit();

        const available = loader.extensions(gpa, null) catch return error.NoDevice;
        defer gpa.free(available);
        if (!vk.has(available, "VK_KHR_surface")) return error.Unsupported;

        var enabled_buf: [8][*:0]const u8 = undefined;
        var enabled_count: usize = 0;
        enabled_buf[0] = "VK_KHR_surface";
        enabled_count = 1;
        for (available) |*ext| {
            if (enabled_count >= enabled_buf.len) break;
            const name = ext.name();
            if (std.mem.eql(u8, name, "VK_KHR_surface")) continue;
            if (std.mem.startsWith(u8, name, "VK_KHR_") and std.mem.endsWith(u8, name, "_surface")) {
                // `name` borrows `ext`'s own zero-padded storage - the byte
                // right after it is the terminator `cstr` found, so this
                // pointer is good as a `[*:0]const u8` for as long as
                // `available` (freed right after `createInstance`, below).
                enabled_buf[enabled_count] = @ptrCast(name.ptr);
                enabled_count += 1;
            }
        }
        const enabled = enabled_buf[0..enabled_count];

        const validation = "VK_LAYER_KHRONOS_validation";
        const with_validation = debug and blk: {
            const layers = loader.layers(gpa) catch break :blk false;
            defer gpa.free(layers);
            for (layers) |*layer| {
                if (std.mem.eql(u8, layer.name(), validation)) break :blk true;
            }
            break :blk false;
        };

        const app: vk.ApplicationInfo = .{ .application_name = "fluxion-rhi", .api_version = vk.v1_0.toInt() };
        var create_info: vk.InstanceCreateInfo = .{ .application_info = &app };
        create_info.setExtensions(enabled);
        const layer_names = [_][*:0]const u8{validation};
        if (with_validation) create_info.setLayers(&layer_names);
        const instance = loader.createInstance(&create_info, null) catch return error.NoDevice;
        const instance_commands = loader.instanceCommands(instance) catch return error.NoDevice;
        errdefer instance_commands.destroyInstance(instance, null);

        const vki = vk.load(vk.gen.commands.Instance, loader.instanceResolver(instance)) catch return error.NoDevice;

        const physical_devices = vk.enumerate.physicalDevices(gpa, vki, instance) catch return error.NoDevice;
        defer gpa.free(physical_devices);
        if (physical_devices.len == 0) return error.NoDevice;

        // Prefer a discrete GPU, while still accepting the driver's first
        // integrated, virtual, or software device as a valid fallback.
        var physical_device = physical_devices[0];
        for (physical_devices) |candidate| {
            var properties: vk.PhysicalDeviceProperties = undefined;
            vki.getPhysicalDeviceProperties(candidate, &properties);
            if (properties.device_type == .discrete_gpu) {
                physical_device = candidate;
                break;
            }
        }

        const device_extensions = vk.enumerate.deviceExtensions(gpa, vki, physical_device, null) catch return error.NoDevice;
        defer gpa.free(device_extensions);
        if (!vk.has(device_extensions, "VK_KHR_swapchain")) return error.Unsupported;
        if (!vk.has(device_extensions, "VK_KHR_maintenance1")) return error.Unsupported;

        const families = vk.enumerate.queueFamilies(gpa, vki, physical_device) catch return error.NoDevice;
        defer gpa.free(families);
        const graphics_family = vk.queueFamily(families, .{ .graphics = true }) orelse return error.NoDevice;

        const priorities = [_]f32{1.0};
        const queues = [_]vk.DeviceQueueCreateInfo{.queues(graphics_family, &priorities)};
        const device_ext = [_][*:0]const u8{ "VK_KHR_swapchain", "VK_KHR_maintenance1" };
        var device_info: vk.DeviceCreateInfo = .{};
        device_info.setQueues(&queues);
        device_info.setExtensions(&device_ext);

        var device: vk.Device = undefined;
        _ = vki.createDevice(physical_device, &device_info, null, &device).check() catch return error.NoDevice;
        const device_commands = instance_commands.deviceCommands(device) catch return error.NoDevice;
        errdefer device_commands.destroyDevice(device, null);

        const vkd = vk.load(vk.gen.commands.Device, instance_commands.deviceResolver(device)) catch return error.NoDevice;

        var graphics_queue: vk.Queue = undefined;
        vkd.getDeviceQueue(device, graphics_family, 0, &graphics_queue);

        var memory_properties: vk.gen.types.PhysicalDeviceMemoryProperties = undefined;
        vki.getPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

        return .{
            .loader = loader,
            .instance = instance,
            .instance_commands = instance_commands,
            .physical_device = physical_device,
            .device = device,
            .device_commands = device_commands,
            .graphics_queue = graphics_queue,
            .graphics_family = graphics_family,
            .vki = vki,
            .vkd = vkd,
            .memory_properties = memory_properties,
        };
    }

    pub fn deinit(self: *Runtime) void {
        _ = self.device_commands.deviceWaitIdle(self.device).check() catch {};
        self.device_commands.destroyDevice(self.device, null);
        self.instance_commands.destroyInstance(self.instance, null);
        self.loader.deinit();
        self.* = undefined;
    }
};

/// True when the platform Vulkan loader can be opened. A loader alone is not a
/// usable device; physical-device and surface selection belongs to `open`.
pub fn loaderAvailable() bool {
    var loader = vk.Loader.init() catch return false;
    defer loader.deinit();
    return true;
}

// -------------------------------------------------------------------------
// The backend's own state
// -------------------------------------------------------------------------

const TexBinding = struct { texture: *TextureRes, sampler: *SamplerRes };

pub const Vk = struct {
    gpa: Allocator,
    runtime: Runtime,

    /// The one command buffer everything - resource uploads and every
    /// frame's draws alike - is recorded into, one at a time. See the
    /// module doc: this backend has no frames in flight.
    command_pool: vk.gen.types.CommandPool,
    command_buffer: vk.gen.types.CommandBuffer,
    fence: vk.gen.types.Fence,

    pipeline_layout: vk.gen.types.PipelineLayout,
    set_layout_uniforms: vk.gen.types.DescriptorSetLayout,
    set_layout_textures: vk.gen.types.DescriptorSetLayout,
    /// The pools descriptor sets come from; `pool_index` is the one this
    /// submit takes from now.
    descriptor_pools: std.ArrayListUnmanaged(vk.gen.types.DescriptorPool) = .empty,
    pool_index: usize = 0,

    /// What an unbound uniform or texture slot reads from - see
    /// `vulkan_resources.zig`'s "Dummy resources" section.
    dummy_buffer: *BufferRes,
    dummy_texture: *TextureRes,
    dummy_sampler: *SamplerRes,

    render_passes: [max_render_passes]?swapchain.RenderPassEntry = @splat(null),

    renderer: [256]u8 = undefined,
    renderer_len: usize = 0,

    // --- state that only means something while a command buffer is being
    // recorded, reset by `record` and by every `beginPass` ---
    recording: bool = false,
    in_pass: bool = false,
    /// The surfaces whose images this submit acquired: it waits for each.
    acquired: [max_surfaces_per_submit]vk.gen.types.Semaphore = undefined,
    acquired_count: usize = 0,
    pass_format: vk.gen.types.Format = .undefined,
    pass_extent: [2]u32 = .{ 0, 0 },
    current_pipeline: ?*PipelineRes = null,
    pipeline_dirty: bool = false,
    current_ubo: [uniform_slots]?*BufferRes = @splat(null),
    current_tex: [texture_slots]?TexBinding = @splat(null),
    bindings_dirty: bool = true,

    pub fn cast(impl: backend.Impl) *Vk {
        return @ptrCast(@alignCast(impl));
    }
};

pub fn cast(impl: backend.Impl) *Vk {
    return Vk.cast(impl);
}

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

/// Opens `Runtime`, the shared pipeline layout, and the dummy resources
/// unbound slots read from, then returns the real `Vtable` below. What
/// `Device.opener(.vulkan)` calls. `desc.software` has no meaning here.
pub fn open(gpa: Allocator, desc: types.DeviceDesc) types.Error!backend.Opened {
    var runtime = try Runtime.open(gpa, desc.debug);
    errdefer runtime.deinit();
    const vkd = runtime.vkd;

    var command_pool: vk.gen.types.CommandPool = .none;
    _ = vkd.createCommandPool(runtime.device, &.{
        .flags = .{ .reset_command_buffer = true },
        .queue_family_index = runtime.graphics_family,
    }, null, &command_pool).check() catch return error.NoDevice;
    errdefer vkd.destroyCommandPool(runtime.device, command_pool, null);

    var command_buffer: vk.gen.types.CommandBuffer = undefined;
    _ = vkd.allocateCommandBuffers(runtime.device, &.{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&command_buffer)).check() catch return error.NoDevice;

    var fence: vk.gen.types.Fence = .none;
    _ = vkd.createFence(runtime.device, &.{}, null, &fence).check() catch return error.NoDevice;
    errdefer vkd.destroyFence(runtime.device, fence, null);

    var ubo_bindings: [uniform_slots]vk.gen.types.DescriptorSetLayoutBinding = undefined;
    for (&ubo_bindings, 0..) |*b, i| b.* = .{
        .binding = @intCast(i),
        .descriptor_type = .uniform_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex = true, .fragment = true },
    };
    var set_layout_uniforms: vk.gen.types.DescriptorSetLayout = .none;
    _ = vkd.createDescriptorSetLayout(runtime.device, &.{
        .binding_count = ubo_bindings.len,
        .bindings = &ubo_bindings,
    }, null, &set_layout_uniforms).check() catch return error.NoDevice;
    errdefer vkd.destroyDescriptorSetLayout(runtime.device, set_layout_uniforms, null);

    // Every stage, as the Direct3D backends bind a texture to both.
    var tex_bindings: [texture_slots]vk.gen.types.DescriptorSetLayoutBinding = undefined;
    for (&tex_bindings, 0..) |*b, i| b.* = .{
        .binding = @intCast(i),
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex = true, .fragment = true },
    };
    var set_layout_textures: vk.gen.types.DescriptorSetLayout = .none;
    _ = vkd.createDescriptorSetLayout(runtime.device, &.{
        .binding_count = tex_bindings.len,
        .bindings = &tex_bindings,
    }, null, &set_layout_textures).check() catch return error.NoDevice;
    errdefer vkd.destroyDescriptorSetLayout(runtime.device, set_layout_textures, null);

    const set_layouts = [_]vk.gen.types.DescriptorSetLayout{ set_layout_uniforms, set_layout_textures };
    var pipeline_layout: vk.gen.types.PipelineLayout = .none;
    _ = vkd.createPipelineLayout(runtime.device, &.{
        .set_layout_count = set_layouts.len,
        .set_layouts = &set_layouts,
    }, null, &pipeline_layout).check() catch return error.NoDevice;
    errdefer vkd.destroyPipelineLayout(runtime.device, pipeline_layout, null);

    const self = try gpa.create(Vk);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .runtime = runtime,
        .command_pool = command_pool,
        .command_buffer = command_buffer,
        .fence = fence,
        .pipeline_layout = pipeline_layout,
        .set_layout_uniforms = set_layout_uniforms,
        .set_layout_textures = set_layout_textures,
        .dummy_buffer = undefined,
        .dummy_texture = undefined,
        .dummy_sampler = undefined,
    };
    errdefer self.descriptor_pools.deinit(gpa);
    try addDescriptorPool(self);

    var props: vk.PhysicalDeviceProperties = undefined;
    self.runtime.vki.getPhysicalDeviceProperties(self.runtime.physical_device, &props);
    const name = props.name();
    const n = @min(name.len, self.renderer.len);
    @memcpy(self.renderer[0..n], name[0..n]);
    self.renderer_len = n;

    self.dummy_buffer = resources.createDummyBuffer(self) catch return error.NoDevice;
    self.dummy_texture = resources.createDummyTexture(self) catch return error.NoDevice;
    self.dummy_sampler = resources.createDummySampler(self) catch return error.NoDevice;

    return .{ self, &vtable };
}

/// The live `VkInstance` and its `vkGetInstanceProcAddr`, for
/// `Device.vulkanInstanceHandles` - `fluxion-platform`'s
/// `Window.createVulkanSurface` takes exactly these two.
pub fn instanceHandles(impl: backend.Impl) ?backend.VulkanInstanceHandles {
    const self = cast(impl);
    return .{
        .instance = @intFromPtr(self.runtime.instance),
        .get_instance_proc_addr = @ptrCast(self.runtime.loader.getInstanceProcAddr),
    };
}

const vtable: backend.Vtable = .{
    .deinit = deinit,
    .info = info,
    .caps = caps,
    .createBuffer = resources.createBuffer,
    .destroyBuffer = resources.destroyBuffer,
    .updateBuffer = resources.updateBuffer,
    .createTexture = resources.createTexture,
    .destroyTexture = resources.destroyTexture,
    .writeTexture = resources.writeTexture,
    .readTexture = resources.readTexture,
    .createSampler = resources.createSampler,
    .destroySampler = resources.destroySampler,
    .createShader = resources.createShader,
    .destroyShader = resources.destroyShader,
    .createPipeline = resources.createPipeline,
    .destroyPipeline = resources.destroyPipeline,
    .createSurface = swapchain.createSurface,
    .destroySurface = swapchain.destroySurface,
    .resizeSurface = swapchain.resizeSurface,
    .surfaceSize = swapchain.surfaceSize,
    .present = swapchain.present,
    .submit = submit,
};

fn deinit(impl: backend.Impl) void {
    const self = cast(impl);
    const vkd = self.runtime.vkd;
    _ = vkd.deviceWaitIdle(self.runtime.device).check() catch {};

    resources.destroyDummyResources(self);
    for (self.render_passes) |maybe| if (maybe) |entry| vkd.destroyRenderPass(self.runtime.device, entry.pass, null);

    for (self.descriptor_pools.items) |pool| vkd.destroyDescriptorPool(self.runtime.device, pool, null);
    self.descriptor_pools.deinit(self.gpa);
    vkd.destroyPipelineLayout(self.runtime.device, self.pipeline_layout, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_textures, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_uniforms, null);
    vkd.destroyFence(self.runtime.device, self.fence, null);
    // Frees `self.command_buffer` too.
    vkd.destroyCommandPool(self.runtime.device, self.command_pool, null);

    self.runtime.deinit();
    self.gpa.destroy(self);
}

fn info(impl: backend.Impl) types.Info {
    const self = cast(impl);
    return .{ .backend = .vulkan, .renderer = self.renderer[0..self.renderer_len] };
}

// -------------------------------------------------------------------------
// Capabilities
// -------------------------------------------------------------------------

/// `.d2`, one mip, one sample, `rgba8_unorm`, `bgra8_unorm` and `r8_unorm`,
/// each as the device says it can be sampled, filtered, drawn into and
/// blended. Every format not listed here defaults to `FormatSupport{}` -
/// unsupported - so `Device`'s caps-driven pre-validation refuses the rest
/// before this backend is ever asked.
fn caps(impl: backend.Impl) types.Caps {
    const self = cast(impl);
    const vki = self.runtime.vki;

    var props: vk.PhysicalDeviceProperties = undefined;
    vki.getPhysicalDeviceProperties(self.runtime.physical_device, &props);
    const limits = props.limits;

    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = limits.max_image_dimension_2d,
            .max_texture_3d = limits.max_image_dimension_3d,
            .max_texture_cube = limits.max_image_dimension_cube,
            .max_texture_layers = limits.max_image_array_layers,
            // Anisotropic filtering is not here yet, so one is the most
            // `createSampler` will ever be asked to give.
            .max_anisotropy = 1,
            // No `extra_colors`, so one colour attachment is all a pass has.
            .max_color_attachments = 1,
        },
        .features = .{ .sampler_border = true, .sampler_lod_bias = true },
    };

    for ([_]types.Format{ .rgba8_unorm, .bgra8_unorm, .r8_unorm }) |format| {
        const vk_format = toVkFormat(format).?;
        var fp: vk.gen.types.FormatProperties = undefined;
        vki.getPhysicalDeviceFormatProperties(self.runtime.physical_device, vk_format, &fp);
        const feats = fp.optimal_tiling_features;
        answer.formats.set(format, .{
            .sampled = feats.sampled_image,
            .filterable = feats.sampled_image_filter_linear,
            .render_target = feats.color_attachment,
            .blendable = feats.color_attachment_blend,
            .generate_mips = false,
            .sample_counts = 0b1,
            .dimensions = std.EnumSet(types.Dimension).initOne(.d2),
        });
    }
    return answer;
}

// -------------------------------------------------------------------------
// Recording: the command buffer every submit, upload and readback records
// into, one after another
// -------------------------------------------------------------------------

const forever: u64 = ~@as(u64, 0);

/// Start recording into the one command buffer.
pub fn record(self: *Vk) types.Error!void {
    const vkd = self.runtime.vkd;
    _ = vkd.resetCommandBuffer(self.command_buffer, .{}).check() catch return error.Failed;
    _ = vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit = true } }).check() catch return error.Failed;
    self.recording = true;
    self.in_pass = false;
    self.acquired_count = 0;
}

/// Stop recording, submit - waiting for every image this recording
/// acquired - and block until the GPU has done it all. The fence is reset
/// only here, right before the submit that signals it, so a recording given
/// up on never leaves it waiting for a signal that will not come.
pub fn finish(self: *Vk) types.Error!void {
    const vkd = self.runtime.vkd;
    self.recording = false;
    _ = vkd.endCommandBuffer(self.command_buffer).check() catch return error.Failed;

    var wait_stages: [max_surfaces_per_submit]vk.gen.types.PipelineStageFlags = @splat(.{ .color_attachment_output = true });
    const waits = self.acquired_count;
    self.acquired_count = 0;
    const cmd_buffers = [_]vk.gen.types.CommandBuffer{self.command_buffer};
    const submit_info = [_]vk.gen.types.SubmitInfo{.{
        .wait_semaphore_count = @intCast(waits),
        .wait_semaphores = if (waits > 0) &self.acquired else null,
        .wait_dst_stage_mask = if (waits > 0) &wait_stages else null,
        .command_buffer_count = cmd_buffers.len,
        .command_buffers = &cmd_buffers,
    }};
    _ = vkd.resetFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}).check() catch return error.Failed;
    _ = vkd.queueSubmit(self.runtime.graphics_queue, 1, &submit_info, self.fence).check() catch |err| switch (err) {
        error.DeviceLost => return error.DeviceLost,
        else => return error.Failed,
    };
    _ = vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
}

/// What a submit that failed part-way leaves: its pass ended and what was
/// recorded submitted, so that every image it acquired is waited for, every
/// texture is back in the layout it is sampled in, and the next recording
/// starts clean.
fn abandon(self: *Vk) void {
    if (!self.recording) return;
    if (self.in_pass) self.runtime.vkd.cmdEndRenderPass(self.command_buffer);
    self.in_pass = false;
    finish(self) catch {};
}

// -------------------------------------------------------------------------
// Submitting a frame
// -------------------------------------------------------------------------

/// Walks the `Command` union once, recording into `self.command_buffer` -
/// then submits it and blocks on `self.fence` before returning, per the
/// module doc's "fully synchronous" rule.
fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) types.Error!void {
    const self = cast(impl);
    const vkd = self.runtime.vkd;
    const cmd = self.command_buffer;

    try record(self);
    errdefer abandon(self);
    try resetDescriptorPools(self);
    self.current_pipeline = null;
    self.pipeline_dirty = false;
    self.current_ubo = @splat(null);
    self.current_tex = @splat(null);
    self.bindings_dirty = true;

    for (list) |command| {
        switch (command) {
            .begin_pass => |pass| try beginPass(self, device, pass),
            .end_pass => {
                vkd.cmdEndRenderPass(cmd);
                self.in_pass = false;
            },
            .set_pipeline => |h| {
                self.current_pipeline = as(PipelineRes, device.pipelines.get(h).?.native);
                self.pipeline_dirty = true;
            },
            .set_viewport => |v| setViewport(self, v),
            .set_scissor => |maybe| setScissor(self, maybe),
            .set_vertex_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                const buffers = [_]vk.gen.types.Buffer{res.buffer};
                const offsets = [_]vk.gen.types.DeviceSize{b.offset};
                vkd.cmdBindVertexBuffers(cmd, b.slot, 1, &buffers, &offsets);
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                vkd.cmdBindIndexBuffer(cmd, res.buffer, 0, if (b.format == .u16) .uint16 else .uint32);
            },
            .set_uniform_buffer => |b| {
                if (b.slot >= uniform_slots) return error.Unsupported;
                self.current_ubo[b.slot] = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.bindings_dirty = true;
            },
            .set_texture => |b| {
                if (b.slot >= texture_slots) return error.Unsupported;
                self.current_tex[b.slot] = .{
                    .texture = as(TextureRes, device.textures.get(b.texture).?.native),
                    .sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native),
                };
                self.bindings_dirty = true;
            },
            .draw => |d| {
                try prepareDraw(self);
                vkd.cmdDraw(cmd, d.vertex_count, d.instance_count, d.first_vertex, 0);
            },
            .draw_indexed => |d| {
                try prepareDraw(self);
                vkd.cmdDrawIndexed(cmd, d.index_count, d.instance_count, d.first_index, d.base_vertex, 0);
            },
            // `caps.formats[*].generate_mips` is always false, so `Device`
            // refuses this before it reaches here.
            .generate_mips => return error.Unsupported,
        }
    }

    try finish(self);
}

fn beginPass(self: *Vk, device: *Device, pass: types.RenderPassDesc) types.Error!void {
    if (pass.depth != null or pass.extra_colors.len > 0) return error.Unsupported;
    const color = pass.color orelse return error.Unsupported;
    const load = swapchain.mapLoadOp(color.load);

    var framebuffer: vk.gen.types.Framebuffer = .none;
    var render_pass: vk.gen.types.RenderPass = .none;
    var format: vk.gen.types.Format = .undefined;
    var extent: [2]u32 = undefined;
    switch (color.target) {
        .surface => |h| {
            const surface = as(SurfaceRes, device.surfaces.get(h).?.native);
            if (try swapchain.acquire(self, surface)) {
                if (self.acquired_count >= max_surfaces_per_submit) return error.Unsupported;
                self.acquired[self.acquired_count] = surface.acquired_signal;
                self.acquired_count += 1;
            }
            const index = surface.acquired.?;
            // An image drawn into before is presentable; one that never was
            // has nothing in it to keep.
            const initial: vk.gen.types.ImageLayout = if (load == .load and surface.drawn[index]) .present_src_khr else .undefined;
            surface.drawn[index] = true;
            render_pass = try swapchain.getRenderPass(self, surface.format, load, initial, .present_src_khr);
            framebuffer = surface.framebuffers[index];
            format = surface.format;
            extent = .{ surface.width, surface.height };
        },
        .texture => |h| {
            const texture = as(TextureRes, device.textures.get(h).?.native);
            if (texture.framebuffer == .none) return error.Unsupported;
            const initial: vk.gen.types.ImageLayout = if (load == .load) .shader_read_only_optimal else .undefined;
            render_pass = try swapchain.getRenderPass(self, texture.vk_format, load, initial, .shader_read_only_optimal);
            framebuffer = texture.framebuffer;
            format = texture.vk_format;
            extent = .{ texture.width, texture.height };
        },
    }

    const clears = [_]vk.gen.types.ClearValue{.{ .color = .{ .float32 = color.clear_color } }};
    self.runtime.vkd.cmdBeginRenderPass(self.command_buffer, &.{
        .render_pass = render_pass,
        .framebuffer = framebuffer,
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = extent[0], .height = extent[1] } },
        .clear_value_count = clears.len,
        .clear_values = &clears,
    }, .@"inline");
    self.in_pass = true;
    self.pass_format = format;
    self.pass_extent = extent;

    // A pass begins covering the whole attachment, like every other
    // backend - `set_viewport`/`set_scissor` override this afterwards.
    setViewport(self, .{ .width = @floatFromInt(extent[0]), .height = @floatFromInt(extent[1]) });
    setScissor(self, null);

    self.current_pipeline = null;
    self.pipeline_dirty = false;
    self.current_ubo = @splat(null);
    self.current_tex = @splat(null);
    self.bindings_dirty = true;
}

/// A viewport given to Vulkan from its bottom edge, with a negative height:
/// clip-space Y then points up the picture, as it does on Direct3D. See the
/// module doc.
fn setViewport(self: *Vk, v: types.Viewport) void {
    const viewport = [_]vk.gen.types.Viewport{.{
        .x = v.x,
        .y = v.y + v.height,
        .width = v.width,
        .height = -v.height,
        .min_depth = v.min_depth,
        .max_depth = v.max_depth,
    }};
    self.runtime.vkd.cmdSetViewport(self.command_buffer, 0, 1, &viewport);
}

/// Null is the whole attachment. Vulkan takes no negative offset, so a
/// rectangle that starts outside the target is cut to what is inside it.
fn setScissor(self: *Vk, maybe: ?types.Rect) void {
    const r: vk.gen.types.Rect2D = if (maybe) |rect| blk: {
        const left = @max(rect.x, 0);
        const top = @max(rect.y, 0);
        const right = @max(rect.x + @as(i32, @intCast(rect.width)), left);
        const bottom = @max(rect.y + @as(i32, @intCast(rect.height)), top);
        break :blk .{
            .offset = .{ .x = left, .y = top },
            .extent = .{ .width = @intCast(right - left), .height = @intCast(bottom - top) },
        };
    } else .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = .{ .width = self.pass_extent[0], .height = self.pass_extent[1] },
    };
    const scissor = [_]vk.gen.types.Rect2D{r};
    self.runtime.vkd.cmdSetScissor(self.command_buffer, 0, 1, &scissor);
}

/// What a draw needs bound: the pipeline made for this pass's format, and
/// the descriptor sets for what is bound now.
fn prepareDraw(self: *Vk) types.Error!void {
    if (self.pipeline_dirty) {
        const res = self.current_pipeline orelse return error.InvalidArgument;
        const pipeline = try resources.pipelineFor(self, res, self.pass_format);
        self.runtime.vkd.cmdBindPipeline(self.command_buffer, .graphics, pipeline);
        self.pipeline_dirty = false;
    }
    try flushBindings(self);
}

fn addDescriptorPool(self: *Vk) types.Error!void {
    const pool_sizes = [_]vk.gen.types.DescriptorPoolSize{
        .{ .type = .uniform_buffer, .descriptor_count = uniform_slots * binding_states_per_pool },
        .{ .type = .combined_image_sampler, .descriptor_count = texture_slots * binding_states_per_pool },
    };
    var pool: vk.gen.types.DescriptorPool = .none;
    _ = self.runtime.vkd.createDescriptorPool(self.runtime.device, &.{
        .max_sets = binding_states_per_pool * 2,
        .pool_size_count = pool_sizes.len,
        .pool_sizes = &pool_sizes,
    }, null, &pool).check() catch return error.OutOfMemory;
    self.descriptor_pools.append(self.gpa, pool) catch {
        self.runtime.vkd.destroyDescriptorPool(self.runtime.device, pool, null);
        return error.OutOfMemory;
    };
}

fn resetDescriptorPools(self: *Vk) types.Error!void {
    for (self.descriptor_pools.items) |pool| {
        _ = self.runtime.vkd.resetDescriptorPool(self.runtime.device, pool, 0).check() catch return error.Failed;
    }
    self.pool_index = 0;
}

/// Allocates a fresh `(set 0, set 1)` pair, writes it from
/// `self.current_ubo`/`current_tex` - an unbound slot reads the dummy
/// resource, see `vulkan_resources.zig` - and binds it, but only when a
/// binding actually changed since the last draw. A pool that is full is
/// followed by the next, made when there is none.
fn flushBindings(self: *Vk) types.Error!void {
    if (!self.bindings_dirty) return;
    self.bindings_dirty = false;
    const vkd = self.runtime.vkd;

    var uniform_infos: [uniform_slots]vk.gen.types.DescriptorBufferInfo = undefined;
    for (&uniform_infos, 0..) |*info_, i| {
        const res = self.current_ubo[i] orelse self.dummy_buffer;
        info_.* = .{ .buffer = res.buffer, .offset = 0, .range = vk.gen.types.whole_size };
    }
    var texture_infos: [texture_slots]vk.gen.types.DescriptorImageInfo = undefined;
    for (&texture_infos, 0..) |*info_, i| {
        const binding = self.current_tex[i];
        const tex = if (binding) |b| b.texture else self.dummy_texture;
        const samp = if (binding) |b| b.sampler else self.dummy_sampler;
        info_.* = .{ .sampler = samp.sampler, .image_view = tex.view, .image_layout = .shader_read_only_optimal };
    }

    const set_layouts = [_]vk.gen.types.DescriptorSetLayout{ self.set_layout_uniforms, self.set_layout_textures };
    var sets: [2]vk.gen.types.DescriptorSet = undefined;
    while (true) {
        if (self.pool_index >= self.descriptor_pools.items.len) try addDescriptorPool(self);
        _ = vkd.allocateDescriptorSets(self.runtime.device, &.{
            .descriptor_pool = self.descriptor_pools.items[self.pool_index],
            .descriptor_set_count = set_layouts.len,
            .set_layouts = &set_layouts,
        }, &sets).check() catch |err| switch (err) {
            error.OutOfPoolMemory, error.FragmentedPool => {
                self.pool_index += 1;
                continue;
            },
            else => return error.OutOfMemory,
        };
        break;
    }

    // Binding zero of each set, with `descriptorCount` wider than that one
    // binding's array: Vulkan overflows the write into bindings one, two and
    // three of the same set, which is the standard way to fill several
    // single-descriptor bindings in one `VkWriteDescriptorSet`.
    const writes = [_]vk.gen.types.WriteDescriptorSet{
        .{ .dst_set = sets[0], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = uniform_slots, .descriptor_type = .uniform_buffer, .buffer_info = &uniform_infos },
        .{ .dst_set = sets[1], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = texture_slots, .descriptor_type = .combined_image_sampler, .image_info = &texture_infos },
    };
    vkd.updateDescriptorSets(self.runtime.device, writes.len, &writes, 0, null);

    vkd.cmdBindDescriptorSets(self.command_buffer, .graphics, self.pipeline_layout, 0, sets.len, &sets, 0, null);
}

// -------------------------------------------------------------------------
// Tests. Skipped on a machine with no Vulkan device. What a pass drew is read
// back from the texture it drew into; the sprites example's tests draw a
// whole textured, blended, instanced frame here and compare it with
// Direct3D's.
// -------------------------------------------------------------------------

const testing = std.testing;

test "a Vulkan runtime opens and closes when the machine has a graphics device" {
    var runtime = Runtime.open(std.testing.allocator, false) catch |err| switch (err) {
        error.NoDevice, error.Unsupported => return error.SkipZigTest,
        else => return err,
    };
    defer runtime.deinit();
    try testing.expect(runtime.graphics_family < std.math.maxInt(u32));
}

fn openTestDevice() !Device {
    return Device.init(testing.allocator, .{ .backend = .vulkan }) catch |err| switch (err) {
        error.NoDevice, error.Unsupported => return error.SkipZigTest,
        else => return err,
    };
}

test "the backend opens, reports itself, and hands back a live instance" {
    var device = try openTestDevice();
    defer device.deinit();

    try testing.expectEqual(types.Backend.vulkan, device.info().backend);
    try testing.expect(device.info().renderer.len > 0);

    const handles = device.vulkanInstanceHandles() orelse return error.TestExpectedEqual;
    try testing.expect(handles.instance != 0);
}

test "resources create: buffer, texture, target, sampler, shader, pipeline" {
    var device = try openTestDevice();
    defer device.deinit();

    const vertices = try device.createBuffer(.{ .kind = .vertex, .size = 64 });
    try device.updateBuffer(vertices, 0, "abcd");
    device.destroyBuffer(vertices);

    var pixel = [4]u8{ 10, 20, 30, 255 };
    const texture = try device.createTexture(.{ .width = 1, .height = 1, .data = &pixel });
    const target = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    const sampler = try device.createSampler(.linear);
    const border = try device.createSampler(.{ .wrap_u = .border, .border = .opaque_white });
    _ = .{ texture, target, sampler, border };

    const shader = try device.createShader(.{ .spirv = .{ .vertex = triangle_vertex, .fragment = triangle_fragment } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
    });
    _ = pipeline;
}

test "a texture reads back as it was made, and a box written into it changes that box" {
    var device = try openTestDevice();
    defer device.deinit();

    var pixels: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| pixels[i * 4 ..][0..4].* = .{ @intCast(i * 10), 20, 30, 255 };
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .data = &pixels });
    {
        const back = try device.readTexture(texture, testing.allocator);
        defer testing.allocator.free(back);
        try testing.expectEqualSlices(u8, &pixels, back);
    }

    const green = [_]u8{ 0, 255, 0, 255 } ** 4;
    try device.writeTexture(texture, .{ .x = 1, .y = 1, .width = 2, .height = 2 }, &green, 8, 0);
    const back = try device.readTexture(texture, testing.allocator);
    defer testing.allocator.free(back);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[(1 * 4 + 1) * 4 ..][0..4].*);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[(2 * 4 + 2) * 4 ..][0..4].*);
    try testing.expectEqual([4]u8{ 0, 20, 30, 255 }, back[0..4].*);
}

test "bgra8 and r8 read back as RGBA" {
    var device = try openTestDevice();
    defer device.deinit();

    const bgra = try device.createTexture(.{ .width = 1, .height = 1, .format = .bgra8_unorm, .data = &.{ 10, 20, 30, 40 } });
    const from_bgra = try device.readTexture(bgra, testing.allocator);
    defer testing.allocator.free(from_bgra);
    try testing.expectEqualSlices(u8, &.{ 30, 20, 10, 40 }, from_bgra);

    const r8 = try device.createTexture(.{ .width = 3, .height = 2, .format = .r8_unorm, .data = &.{ 1, 2, 3, 4, 5, 6 } });
    try device.writeTexture(r8, .{ .y = 1, .width = 3, .height = 1 }, &.{ 7, 8, 9 }, 3, 0);
    const from_r8 = try device.readTexture(r8, testing.allocator);
    defer testing.allocator.free(from_r8);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 255 }, from_r8[0..4]);
    try testing.expectEqualSlices(u8, &.{ 9, 9, 9, 255 }, from_r8[5 * 4 ..][0..4]);
}

test "passes clear textures, submit after submit, and each reads back its colour" {
    var device = try openTestDevice();
    defer device.deinit();

    const first = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const second = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    for (0..3) |round| {
        const red: f32 = if (round % 2 == 0) 1 else 0;
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = first }, .clear_color = .{ red, 0, 0, 1 } } });
        try cmd.endPass();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = second }, .clear_color = .{ 0, 0, 1, 1 } } });
        try cmd.endPass();
        try device.submit();
    }
    const a = try device.readTexture(first, testing.allocator);
    defer testing.allocator.free(a);
    const b = try device.readTexture(second, testing.allocator);
    defer testing.allocator.free(b);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, a[0..4].*);
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, b[(7 * 8 + 7) * 4 ..][0..4].*);
}

test "what is not here yet is refused as Unsupported" {
    var device = try openTestDevice();
    defer device.deinit();

    // A shape `caps` never lists a dimension for.
    try testing.expectError(error.Unsupported, device.createTexture(.{
        .dimension = .cube,
        .width = 4,
        .height = 4,
    }));
    // Multisampling.
    try testing.expectError(error.Unsupported, device.createTexture(.{
        .width = 4,
        .height = 4,
        .samples = 4,
        .usage = .{ .sampled = false, .render_target = true },
    }));
    // A depth pipeline: not a `caps` rejection (`Device.createPipeline` has
    // no caps check for it), but the backend's own - see
    // `vulkan_resources.createPipeline`.
    const shader = try device.createShader(.{ .spirv = .{ .vertex = triangle_vertex, .fragment = triangle_fragment } });
    try testing.expectError(error.Unsupported, device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
        .depth_format = .depth32_float,
    }));
}

// -------------------------------------------------------------------------
// Test fixture: the smallest valid SPIR-V pair
//
// `createShader`/`createPipeline` need real, validator-accepted SPIR-V - a
// garbage `[]u32` fails `vkCreateShaderModule` outright, which would make
// every test above it a test of nothing. Assembled by hand, the same way
// `fluxion-vulkan`'s own examples do (`examples/spirv.zig`, not reachable
// from this module): a header, then a stream of `(word_count << 16) |
// opcode` instructions. Neither shader draws anything anyone looks at -
// the textured draws are the sprites example's - so both are the plainest
// thing that is still a real module: the vertex stage writes a fixed `gl_Position`, the
// fragment stage writes a fixed colour.
// -------------------------------------------------------------------------

fn spirvOp(comptime opcode: u16, comptime operands: []const u32) []const u32 {
    comptime {
        const count: u32 = @intCast(operands.len + 1);
        return [_]u32{(count << 16) | opcode} ++ operands;
    }
}

fn f32Bits(comptime value: f32) u32 {
    return @bitCast(value);
}

const triangle_vertex: []const u32 = blk: {
    // Ids: 1 void, 2 fn(void), 3 float, 4 vec4, 5 gl_PerVertex{vec4}, 6
    // ptr(Output,5), 7 gl_PerVertex var, 8 float 0, 9 float 1, 10 vec4(0,0,0,1),
    // 11 int, 12 int 0, 13 ptr(Output,vec4), 14 main, 15 entry label, 16
    // access-chain result. Bound 17.
    const header = [_]u32{ 0x07230203, 0x0001_0000, 0, 17, 0 };
    const cap = spirvOp(17, &.{1}); // OpCapability Shader
    const mem = spirvOp(14, &.{ 0, 1 }); // OpMemoryModel Logical GLSL450
    const entry = spirvOp(15, &.{ 0, 14, 0x6E69616D, 0, 7 }); // OpEntryPoint Vertex %14 "main" %7
    const dec1 = spirvOp(72, &.{ 5, 0, 11, 0 }); // OpMemberDecorate %5 0 BuiltIn Position
    const dec2 = spirvOp(71, &.{ 5, 2 }); // OpDecorate %5 Block
    const g1 = spirvOp(19, &.{1}); // OpTypeVoid %1
    const g2 = spirvOp(33, &.{ 2, 1 }); // OpTypeFunction %2 %1
    const g3 = spirvOp(22, &.{ 3, 32 }); // OpTypeFloat %3 32
    const g4 = spirvOp(23, &.{ 4, 3, 4 }); // OpTypeVector %4 %3 4
    const g5 = spirvOp(30, &.{ 5, 4 }); // OpTypeStruct %5 %4
    const g6 = spirvOp(32, &.{ 6, 3, 5 }); // OpTypePointer %6 Output %5
    const g7 = spirvOp(59, &.{ 6, 7, 3 }); // OpVariable %6 %7 Output
    const g8 = spirvOp(43, &.{ 3, 8, f32Bits(0.0) }); // OpConstant %3 %8 0.0
    const g9 = spirvOp(43, &.{ 3, 9, f32Bits(1.0) }); // OpConstant %3 %9 1.0
    const g10 = spirvOp(44, &.{ 4, 10, 8, 8, 8, 9 }); // OpConstantComposite %4 %10 %8 %8 %8 %9
    const g11 = spirvOp(21, &.{ 11, 32, 1 }); // OpTypeInt %11 32 1
    const g12 = spirvOp(43, &.{ 11, 12, 0 }); // OpConstant %11 %12 0
    const g13 = spirvOp(32, &.{ 13, 3, 4 }); // OpTypePointer %13 Output %4
    const f1 = spirvOp(54, &.{ 1, 14, 0, 2 }); // OpFunction %1 %14 None %2
    const f2 = spirvOp(248, &.{15}); // OpLabel %15
    const f3 = spirvOp(65, &.{ 13, 16, 7, 12 }); // OpAccessChain %13 %16 %7 %12
    const f4 = spirvOp(62, &.{ 16, 10 }); // OpStore %16 %10
    const f5 = spirvOp(253, &.{}); // OpReturn
    const f6 = spirvOp(56, &.{}); // OpFunctionEnd

    break :blk header ++ cap ++ mem ++ entry ++ dec1 ++ dec2 ++
        g1 ++ g2 ++ g3 ++ g4 ++ g5 ++ g6 ++ g7 ++ g8 ++ g9 ++ g10 ++ g11 ++ g12 ++ g13 ++
        f1 ++ f2 ++ f3 ++ f4 ++ f5 ++ f6;
};

const triangle_fragment: []const u32 = blk: {
    // Ids: 1 void, 2 fn(void), 3 float, 4 vec4, 5 ptr(Output,vec4), 6 out
    // colour var, 7 float 1, 8 vec4(1,1,1,1), 9 main, 10 entry label. Bound 11.
    const header = [_]u32{ 0x07230203, 0x0001_0000, 0, 11, 0 };
    const cap = spirvOp(17, &.{1});
    const mem = spirvOp(14, &.{ 0, 1 });
    const entry = spirvOp(15, &.{ 4, 9, 0x6E69616D, 0, 6 }); // ExecutionModel Fragment
    const exec = spirvOp(16, &.{ 9, 7 }); // OpExecutionMode %9 OriginUpperLeft
    const dec = spirvOp(71, &.{ 6, 30, 0 }); // OpDecorate %6 Location 0
    const g1 = spirvOp(19, &.{1});
    const g2 = spirvOp(33, &.{ 2, 1 });
    const g3 = spirvOp(22, &.{ 3, 32 });
    const g4 = spirvOp(23, &.{ 4, 3, 4 });
    const g5 = spirvOp(32, &.{ 5, 3, 4 }); // ptr Output vec4
    const g6 = spirvOp(59, &.{ 5, 6, 3 }); // variable Output
    const g7 = spirvOp(43, &.{ 3, 7, f32Bits(1.0) });
    const g8 = spirvOp(44, &.{ 4, 8, 7, 7, 7, 7 });
    const f1 = spirvOp(54, &.{ 1, 9, 0, 2 });
    const f2 = spirvOp(248, &.{10});
    const f3 = spirvOp(62, &.{ 6, 8 });
    const f4 = spirvOp(253, &.{});
    const f5 = spirvOp(56, &.{});

    break :blk header ++ cap ++ mem ++ entry ++ exec ++ dec ++
        g1 ++ g2 ++ g3 ++ g4 ++ g5 ++ g6 ++ g7 ++ g8 ++
        f1 ++ f2 ++ f3 ++ f4 ++ f5;
};
