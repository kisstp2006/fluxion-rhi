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
//! returning: `present` waits on the render-finished semaphore (already
//! signalled by the time it is checked) and then blocks on the queue too.
//! Slower than double-buffering, and unambiguously correct - which is what
//! the plan's Key Decision 2 asks a v1 backend for.
//!
//! **One shared pipeline layout.** Every pipeline this backend makes uses
//! the same two descriptor sets - `set 0` four uniform-buffer slots, `set 1`
//! four combined-image-sampler slots - so `setUniformBuffer`/`setTexture`
//! and a draw never have to know which pipeline is bound. A descriptor pool
//! reset once per `submit` and a pair of sets allocated fresh whenever a
//! binding actually changes (`flushBindings`) is what makes that safe: a
//! descriptor set's *content* is whatever the last `vkUpdateDescriptorSets`
//! on it wrote, checked only when the GPU executes the draw that reads it -
//! reusing one set across draws with different textures in the same list
//! would silently make every earlier draw use the last texture, not the one
//! it was recorded with.

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

/// `set 0`'s width: four uniform-buffer bindings. See the plan's binding
/// model (Key Decision 3).
pub const uniform_slots = 4;
/// `set 1`'s width: four combined-image-sampler bindings.
pub const texture_slots = 4;

/// How many `(uniform, texture)` descriptor set pairs one `submit` can
/// allocate - reset every `submit`, so this bounds "distinct binding states
/// in one frame", not draws: consecutive draws that do not change what is
/// bound share a pair.
const max_binding_states = 256;

/// How many distinct `(format, load op)` render passes `getRenderPass`
/// caches - two colour formats times three load ops is six; eight leaves room.
const max_render_passes = 8;

// -------------------------------------------------------------------------
// Format mapping
// -------------------------------------------------------------------------

/// The only two colour formats in the MVP scope - everything else is
/// `null`, so `Device`'s caps-driven pre-validation (from `caps` below never
/// marking another format `sampled`) is what actually keeps requests for the
/// rest from ever reaching this backend.
pub fn toVkFormat(format: types.Format) ?vk.gen.types.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        else => null,
    };
}

pub fn fromVkFormat(format: vk.gen.types.Format) ?types.Format {
    return switch (format) {
        .r8g8b8a8_unorm => .rgba8_unorm,
        .b8g8r8a8_unorm => .bgra8_unorm,
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
    /// asked for), and the device requires `VK_KHR_swapchain`.
    pub fn open(gpa: Allocator) types.Error!Runtime {
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

        const app: vk.ApplicationInfo = .{ .application_name = "fluxion-rhi", .api_version = vk.v1_0.toInt() };
        var create_info: vk.InstanceCreateInfo = .{ .application_info = &app };
        create_info.setExtensions(enabled);
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

        const families = vk.enumerate.queueFamilies(gpa, vki, physical_device) catch return error.NoDevice;
        defer gpa.free(families);
        const graphics_family = vk.queueFamily(families, .{ .graphics = true }) orelse return error.NoDevice;

        const priorities = [_]f32{1.0};
        const queues = [_]vk.DeviceQueueCreateInfo{.queues(graphics_family, &priorities)};
        const swapchain_ext = [_][*:0]const u8{"VK_KHR_swapchain"};
        var device_info: vk.DeviceCreateInfo = .{};
        device_info.setQueues(&queues);
        device_info.setExtensions(&swapchain_ext);

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
/// usable device; physical-device and surface selection belongs to `open` when
/// this becomes a full `backend.Opener`.
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
    sem_image_available: vk.gen.types.Semaphore,
    sem_render_finished: vk.gen.types.Semaphore,

    pipeline_layout: vk.gen.types.PipelineLayout,
    set_layout_uniforms: vk.gen.types.DescriptorSetLayout,
    set_layout_textures: vk.gen.types.DescriptorSetLayout,
    descriptor_pool: vk.gen.types.DescriptorPool,

    /// What an unbound uniform or texture slot reads from - see
    /// `vulkan_resources.zig`'s "Dummy resources" section.
    dummy_buffer: *BufferRes,
    dummy_texture: *TextureRes,
    dummy_sampler: *SamplerRes,

    render_passes: [max_render_passes]?swapchain.RenderPassEntry = @splat(null),

    renderer: [256]u8 = undefined,
    renderer_len: usize = 0,

    // --- state that only means something between `submit`'s first command
    // and its last, reset at the top of every `submit` and every `beginPass` ---
    current_surface: ?*SurfaceRes = null,
    current_ubo: [uniform_slots]?*BufferRes = @splat(null),
    current_tex: [texture_slots]?TexBinding = @splat(null),
    bindings_dirty: bool = true,
    pass_extent: [2]u32 = .{ 0, 0 },

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

/// Opens `Runtime`, the shared pipeline layout and descriptor pool, and the
/// dummy resources unbound slots read from, then returns the real `Vtable`
/// below. What `Device.opener(.vulkan)` calls.
pub fn open(gpa: Allocator, desc: types.DeviceDesc) types.Error!backend.Opened {
    _ = desc; // MVP: no debug layer toggle and no software-renderer preference yet.

    var runtime = try Runtime.open(gpa);
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
    _ = vkd.createFence(runtime.device, &.{ .flags = .{ .signaled = true } }, null, &fence).check() catch return error.NoDevice;
    errdefer vkd.destroyFence(runtime.device, fence, null);

    var sem_image_available: vk.gen.types.Semaphore = .none;
    _ = vkd.createSemaphore(runtime.device, &.{}, null, &sem_image_available).check() catch return error.NoDevice;
    errdefer vkd.destroySemaphore(runtime.device, sem_image_available, null);

    var sem_render_finished: vk.gen.types.Semaphore = .none;
    _ = vkd.createSemaphore(runtime.device, &.{}, null, &sem_render_finished).check() catch return error.NoDevice;
    errdefer vkd.destroySemaphore(runtime.device, sem_render_finished, null);

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

    var tex_bindings: [texture_slots]vk.gen.types.DescriptorSetLayoutBinding = undefined;
    for (&tex_bindings, 0..) |*b, i| b.* = .{
        .binding = @intCast(i),
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment = true },
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

    const pool_sizes = [_]vk.gen.types.DescriptorPoolSize{
        .{ .type = .uniform_buffer, .descriptor_count = uniform_slots * max_binding_states },
        .{ .type = .combined_image_sampler, .descriptor_count = texture_slots * max_binding_states },
    };
    var descriptor_pool: vk.gen.types.DescriptorPool = .none;
    _ = vkd.createDescriptorPool(runtime.device, &.{
        .max_sets = max_binding_states * 2,
        .pool_size_count = pool_sizes.len,
        .pool_sizes = &pool_sizes,
    }, null, &descriptor_pool).check() catch return error.NoDevice;
    errdefer vkd.destroyDescriptorPool(runtime.device, descriptor_pool, null);

    const self = try gpa.create(Vk);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .runtime = runtime,
        .command_pool = command_pool,
        .command_buffer = command_buffer,
        .fence = fence,
        .sem_image_available = sem_image_available,
        .sem_render_finished = sem_render_finished,
        .pipeline_layout = pipeline_layout,
        .set_layout_uniforms = set_layout_uniforms,
        .set_layout_textures = set_layout_textures,
        .descriptor_pool = descriptor_pool,
        .dummy_buffer = undefined,
        .dummy_texture = undefined,
        .dummy_sampler = undefined,
    };

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

    vkd.destroyDescriptorPool(self.runtime.device, self.descriptor_pool, null);
    vkd.destroyPipelineLayout(self.runtime.device, self.pipeline_layout, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_textures, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_uniforms, null);
    vkd.destroySemaphore(self.runtime.device, self.sem_render_finished, null);
    vkd.destroySemaphore(self.runtime.device, self.sem_image_available, null);
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

/// Reports narrowly, per the plan's MVP scope table: only `.d2`, one mip,
/// one sample, `rgba8_unorm`/`bgra8_unorm`, never a render target. Every
/// format not listed here defaults to `FormatSupport{}` - unsupported - so
/// `Device`'s caps-driven pre-validation refuses the rest before this
/// backend is ever asked.
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
            // Anisotropic filtering is out of the MVP scope, so one is the
            // most `createSampler` will ever be asked to give.
            .max_anisotropy = 1,
            // No `extra_colors`, so one colour attachment is all a pass has.
            .max_color_attachments = 1,
        },
        .features = .{},
    };

    for ([_]types.Format{ .rgba8_unorm, .bgra8_unorm }) |format| {
        const vk_format = toVkFormat(format).?;
        var fp: vk.gen.types.FormatProperties = undefined;
        vki.getPhysicalDeviceFormatProperties(self.runtime.physical_device, vk_format, &fp);
        const feats = fp.optimal_tiling_features;
        answer.formats.set(format, .{
            .sampled = feats.sampled_image,
            .filterable = feats.sampled_image_filter_linear,
            .render_target = false,
            .blendable = false,
            .generate_mips = false,
            .sample_counts = 0,
            .dimensions = std.EnumSet(types.Dimension).initOne(.d2),
        });
    }
    return answer;
}

// -------------------------------------------------------------------------
// Recording and submitting a frame
// -------------------------------------------------------------------------

/// Walks the `Command` union once, recording into `self.command_buffer` -
/// then submits it and blocks on `self.fence` before returning, per the
/// module doc's "fully synchronous" rule.
fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) types.Error!void {
    const self = cast(impl);
    const vkd = self.runtime.vkd;
    const cmd = self.command_buffer;

    // The previous `submit` already waited on this same fence before it
    // returned, so this is a formality on every call but the first - where
    // it is what makes the pre-signalled fence from `open` harmless.
    _ = vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
    _ = vkd.resetFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}).check() catch return error.Failed;
    _ = vkd.resetDescriptorPool(self.runtime.device, self.descriptor_pool, 0).check() catch return error.Failed;
    _ = vkd.resetCommandBuffer(cmd, .{}).check() catch return error.Failed;
    _ = vkd.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit = true } }).check() catch return error.Failed;

    self.current_surface = null;
    self.current_ubo = @splat(null);
    self.current_tex = @splat(null);
    self.bindings_dirty = true;

    for (list) |command| {
        switch (command) {
            .begin_pass => |pass| try beginPass(self, device, pass),
            .end_pass => vkd.cmdEndRenderPass(cmd),
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                vkd.cmdBindPipeline(cmd, .graphics, res.pipeline);
            },
            .set_viewport => |v| {
                const viewport = [_]vk.gen.types.Viewport{.{
                    .x = v.x,
                    .y = v.y,
                    .width = v.width,
                    .height = v.height,
                    .min_depth = v.min_depth,
                    .max_depth = v.max_depth,
                }};
                vkd.cmdSetViewport(cmd, 0, 1, &viewport);
            },
            .set_scissor => |maybe| {
                const r: vk.gen.types.Rect2D = if (maybe) |rect| .{
                    .offset = .{ .x = rect.x, .y = rect.y },
                    .extent = .{ .width = rect.width, .height = rect.height },
                } else .{
                    .offset = .{ .x = 0, .y = 0 },
                    .extent = .{ .width = self.pass_extent[0], .height = self.pass_extent[1] },
                };
                const scissor = [_]vk.gen.types.Rect2D{r};
                vkd.cmdSetScissor(cmd, 0, 1, &scissor);
            },
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
                try flushBindings(self);
                vkd.cmdDraw(cmd, d.vertex_count, d.instance_count, d.first_vertex, 0);
            },
            .draw_indexed => |d| {
                try flushBindings(self);
                vkd.cmdDrawIndexed(cmd, d.index_count, d.instance_count, d.first_index, d.base_vertex, 0);
            },
            // Out of the MVP scope, and `caps.formats[*].generate_mips` is
            // always false, so `Device` refuses this before it reaches here.
            .generate_mips => return error.Unsupported,
        }
    }

    _ = vkd.endCommandBuffer(cmd).check() catch return error.Failed;

    var wait_semaphores: [1]vk.gen.types.Semaphore = undefined;
    var wait_stages: [1]vk.gen.types.PipelineStageFlags = undefined;
    const wait_count: u32 = if (self.current_surface != null) blk: {
        wait_semaphores[0] = self.sem_image_available;
        wait_stages[0] = .{ .color_attachment_output = true };
        break :blk 1;
    } else 0;

    const cmd_buffers = [_]vk.gen.types.CommandBuffer{cmd};
    const signal_semaphores = [_]vk.gen.types.Semaphore{self.sem_render_finished};
    const submit_info = [_]vk.gen.types.SubmitInfo{.{
        .wait_semaphore_count = wait_count,
        .wait_semaphores = if (wait_count > 0) &wait_semaphores else null,
        .wait_dst_stage_mask = if (wait_count > 0) &wait_stages else null,
        .command_buffer_count = cmd_buffers.len,
        .command_buffers = &cmd_buffers,
        .signal_semaphore_count = signal_semaphores.len,
        .signal_semaphores = &signal_semaphores,
    }};
    _ = vkd.queueSubmit(self.runtime.graphics_queue, 1, &submit_info, self.fence).check() catch return error.Failed;

    // Fully synchronous: block right here, so the descriptor pool, the
    // command buffer and every resource this list touched are safe to
    // reuse - or to destroy - the moment this call returns.
    _ = vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{self.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
}

const forever: u64 = ~@as(u64, 0);

fn beginPass(self: *Vk, device: *Device, pass: types.RenderPassDesc) types.Error!void {
    // The MVP scope is a pass into the surface and nothing else - no depth,
    // no extra colours, and a colour texture is never reached because
    // `caps` never lets one be made with `usage.render_target`.
    if (pass.depth != null or pass.extra_colors.len > 0) return error.Unsupported;
    const color = pass.color orelse return error.Unsupported;
    const surface_h = switch (color.target) {
        .surface => |h| h,
        .texture => return error.Unsupported,
    };
    const surface = as(SurfaceRes, device.surfaces.get(surface_h).?.native);

    if (self.current_surface != surface) {
        try swapchain.acquire(self, surface);
        self.current_surface = surface;
    }

    const load = swapchain.mapLoadOp(color.load);
    const render_pass = try swapchain.getRenderPass(self, surface.format, load);
    const framebuffer = surface.framebuffers[surface.image_index];

    const clears = [_]vk.gen.types.ClearValue{.{ .color = .{ .float32 = color.clear_color } }};
    self.runtime.vkd.cmdBeginRenderPass(self.command_buffer, &.{
        .render_pass = render_pass,
        .framebuffer = framebuffer,
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = surface.width, .height = surface.height } },
        .clear_value_count = clears.len,
        .clear_values = &clears,
    }, .@"inline");

    self.pass_extent = .{ surface.width, surface.height };

    // A pass begins covering the whole attachment, like every other
    // backend - `set_viewport`/`set_scissor` override this afterwards.
    const viewport = [_]vk.gen.types.Viewport{.{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(surface.width),
        .height = @floatFromInt(surface.height),
        .min_depth = 0,
        .max_depth = 1,
    }};
    self.runtime.vkd.cmdSetViewport(self.command_buffer, 0, 1, &viewport);
    const scissor = [_]vk.gen.types.Rect2D{.{
        .offset = .{ .x = 0, .y = 0 },
        .extent = .{ .width = surface.width, .height = surface.height },
    }};
    self.runtime.vkd.cmdSetScissor(self.command_buffer, 0, 1, &scissor);

    self.current_ubo = @splat(null);
    self.current_tex = @splat(null);
    self.bindings_dirty = true;
}

/// Allocates a fresh `(set 0, set 1)` pair from the pool `submit` resets
/// every frame, writes it from `self.current_ubo`/`current_tex` - an unbound
/// slot reads the dummy resource, see `vulkan_resources.zig` - and binds it,
/// but only when a binding actually changed since the last draw.
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
    _ = vkd.allocateDescriptorSets(self.runtime.device, &.{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = set_layouts.len,
        .set_layouts = &set_layouts,
    }, &sets).check() catch return error.OutOfMemory;

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
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "a Vulkan runtime opens and closes when the machine has a graphics device" {
    var runtime = Runtime.open(std.testing.allocator) catch |err| switch (err) {
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

test "the caps-scoped resources create: buffer, texture, sampler, shader, pipeline" {
    var device = try openTestDevice();
    defer device.deinit();

    const vertices = try device.createBuffer(.{ .kind = .vertex, .size = 64 });
    try device.updateBuffer(vertices, 0, "abcd");
    device.destroyBuffer(vertices);

    var pixel = [4]u8{ 10, 20, 30, 255 };
    const texture = try device.createTexture(.{ .width = 1, .height = 1, .data = &pixel });
    const sampler = try device.createSampler(.linear);
    _ = texture;
    _ = sampler;

    const shader = try device.createShader(.{ .spirv = .{ .vertex = triangle_vertex, .fragment = triangle_fragment } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{},
        .buffers = &.{},
    });
    _ = pipeline;
}

test "out-of-scope requests are refused as Unsupported, which proves caps is scoped" {
    var device = try openTestDevice();
    defer device.deinit();

    // A shape `caps` never lists a dimension for.
    try testing.expectError(error.Unsupported, device.createTexture(.{
        .dimension = .cube,
        .width = 4,
        .height = 4,
    }));

    // A render target - `caps` never marks any format `render_target`, and
    // multisampling needs one.
    try testing.expectError(error.Unsupported, device.createTexture(.{
        .width = 4,
        .height = 4,
        .samples = 4,
        .usage = .{ .render_target = true },
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
// opcode` instructions. Neither shader is ever executed here - there is no
// window in this file's tests - so both are the plainest thing that is
// still a real module: the vertex stage writes a fixed `gl_Position`, the
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
