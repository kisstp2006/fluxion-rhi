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
//! **Recordings in flight.** A submit, an upload or a transition is
//! recorded into the next of `ring_size` slots - a command buffer, a fence,
//! descriptor pools and acquire semaphores each - and handed to the queue
//! without waiting for it. A slot is waited for only when it comes round
//! again, so the CPU records while the GPU draws. Every submission has a
//! serial one more than the last, and `completed` is the last the GPU is
//! known to have finished. What that asks of the rest:
//!
//!   - A buffer the GPU may still read is not written: a write to one
//!     moves it to another copy of its memory - a version - which starts
//!     as the buffer's bytes kept on the CPU, the way a Direct3D 11 driver
//!     renames a buffer written with discard. A version is used again once
//!     the GPU is past the last submission that used it.
//!   - What is destroyed, and a staging buffer, is buried with the serial of
//!     the last submission, and freed once `completed` reaches it.
//!   - A readback waits for its own submission.
//!   - A present waits for a semaphore an empty submission signals, after
//!     everything submitted before it.
//!
//! The queue runs submissions in order, so an upload's barriers and a
//! pass's order it against every draw before and after it, as before.
//!
//! **One recording a frame.** A `submit`, an upload or a transition goes on
//! at the end of the recording that is open, rather than into one of its
//! own, and the recording is handed to the queue when something needs it
//! done: a present, a readback, a first draw into a surface after other
//! work (so that work need not wait for the image), or `max_lists` lists. A
//! frame of many passes costs one queue submission rather than one each.
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
//! and a draw never have to know which pipeline is bound. A slot's
//! descriptor pools, reset when the slot is recorded into again, and a pair
//! of sets allocated fresh whenever a binding actually changes
//! (`flushBindings`), is what makes that safe: a
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

/// `set 0`'s width: eight uniform-buffer bindings - within the twelve a
/// stage is promised.
pub const uniform_slots = 8;
/// `set 1`'s width: eight combined-image-sampler bindings.
pub const texture_slots = types.max_texture_slots;

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

/// How many recordings can be on their way to the GPU at once: one waits
/// for the one this many before it, and no other.
const ring_size = 4;

/// The most `submit`s one recording takes before it is handed over: a
/// program that never presents still has its work done.
const max_lists = 64;

// -------------------------------------------------------------------------
// Format mapping
// -------------------------------------------------------------------------

/// The formats this backend makes textures and targets of: the uncompressed
/// ones. Everything else is `null`, so `Device`'s caps-driven pre-validation
/// (from `caps` below never marking another format `sampled`) is what
/// actually keeps requests for the rest from ever reaching this backend.
pub fn toVkFormat(format: types.Format) ?vk.gen.types.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        .r8_unorm => .r8_unorm,
        .rg8_unorm => .r8g8_unorm,
        .r16_float => .r16_sfloat,
        .rg16_float => .r16g16_sfloat,
        .rgba16_float => .r16g16b16a16_sfloat,
        .r32_float => .r32_sfloat,
        .rg32_float => .r32g32_sfloat,
        .rgba32_float => .r32g32b32a32_sfloat,
        .rgb10a2_unorm => .a2b10g10r10_unorm_pack32,
        .rg11b10_float => .b10g11r11_ufloat_pack32,
        .depth16_unorm => .d16_unorm,
        .depth24_stencil8 => .d24_unorm_s8_uint,
        .depth32_float => .d32_sfloat,
        .depth32_float_stencil8 => .d32_sfloat_s8_uint,
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

/// One recording on its way to the GPU: its command buffer, the fence it
/// signals when it is done, the pools its draws' descriptor sets came from
/// and the semaphores its acquires signalled. Recorded into again only once
/// the GPU is done with it.
pub const Slot = struct {
    command_buffer: vk.gen.types.CommandBuffer,
    fence: vk.gen.types.Fence = .none,
    /// The serial of the submission recorded here last; nought for none.
    serial: u64 = 0,
    descriptor_pools: std.ArrayListUnmanaged(vk.gen.types.DescriptorPool) = .empty,
    /// The pool the recording takes sets from now.
    pool_index: usize = 0,
    acquires: [max_surfaces_per_submit]vk.gen.types.Semaphore = @splat(.none),
};

/// Something freed once the GPU is past `serial`.
pub const Grave = struct {
    serial: u64,
    what: union(enum) {
        buffer: resources.Version,
        texture: *TextureRes,
        sampler: *SamplerRes,
        pipeline: *PipelineRes,
        /// A pass's own, made for a colour target and a depth one.
        framebuffer: vk.gen.types.Framebuffer,
    },
};

pub const Vk = struct {
    gpa: Allocator,
    runtime: Runtime,

    command_pool: vk.gen.types.CommandPool,
    /// What is being recorded into now: the slot's at `at`. See the module
    /// doc for the ring.
    command_buffer: vk.gen.types.CommandBuffer = undefined,
    slots: [ring_size]Slot,
    at: usize = 0,
    /// Serials: the last submission handed to the queue, and the last the
    /// GPU is known to have done.
    submitted: u64 = 0,
    completed: u64 = 0,
    graveyard: std.ArrayListUnmanaged(Grave) = .empty,

    pipeline_layout: vk.gen.types.PipelineLayout,
    set_layout_uniforms: vk.gen.types.DescriptorSetLayout,
    set_layout_textures: vk.gen.types.DescriptorSetLayout,

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
    /// How many of the slot's acquire semaphores this recording signalled:
    /// it waits for each.
    acquired_count: usize = 0,
    /// How many lists the open recording holds, and whether it holds
    /// anything at all.
    lists: u32 = 0,
    has_work: bool = false,
    pass_format: vk.gen.types.Format = .undefined,
    /// The open pass's depth attachment's format, `undefined` for none.
    pass_depth: vk.gen.types.Format = .undefined,
    /// The open pass's samples a pixel.
    pass_samples: u8 = 1,
    pass_extent: [2]u32 = .{ 0, 0 },
    current_pipeline: ?*PipelineRes = null,
    pipeline_dirty: bool = false,
    current_ubo: [uniform_slots]?*BufferRes = @splat(null),
    /// Where in its buffer each slot's block is, and how much of it: nought
    /// for the whole buffer.
    current_ubo_range: [uniform_slots]commands.Command.UniformBinding = undefined,
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

    var command_buffers: [ring_size]vk.gen.types.CommandBuffer = undefined;
    _ = vkd.allocateCommandBuffers(runtime.device, &.{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = ring_size,
    }, &command_buffers).check() catch return error.NoDevice;
    var slots: [ring_size]Slot = undefined;
    for (&slots, command_buffers) |*one, buffer| one.* = .{ .command_buffer = buffer };
    errdefer for (&slots) |*one| destroySlot(vkd, runtime.device, one);
    for (&slots) |*one| {
        _ = vkd.createFence(runtime.device, &.{}, null, &one.fence).check() catch return error.NoDevice;
        for (&one.acquires) |*semaphore| {
            _ = vkd.createSemaphore(runtime.device, &.{}, null, semaphore).check() catch return error.NoDevice;
        }
    }

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
        .slots = slots,
        .pipeline_layout = pipeline_layout,
        .set_layout_uniforms = set_layout_uniforms,
        .set_layout_textures = set_layout_textures,
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
    flush(self) catch {};
    waitIdle(self);

    resources.destroyDummyResources(self);
    collect(self);
    self.graveyard.deinit(self.gpa);
    for (self.render_passes) |maybe| if (maybe) |entry| vkd.destroyRenderPass(self.runtime.device, entry.pass, null);

    for (&self.slots) |*one| {
        for (one.descriptor_pools.items) |pool| vkd.destroyDescriptorPool(self.runtime.device, pool, null);
        one.descriptor_pools.deinit(self.gpa);
        destroySlot(vkd, self.runtime.device, one);
    }
    vkd.destroyPipelineLayout(self.runtime.device, self.pipeline_layout, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_textures, null);
    vkd.destroyDescriptorSetLayout(self.runtime.device, self.set_layout_uniforms, null);
    // Frees the slots' command buffers too.
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

/// `.d2` and one mip; the uncompressed formats, each as the device says it
/// can be sampled, filtered, drawn into and blended, and as many samples a
/// pixel as its framebuffers take. Depth is drawn into, not sampled. Every
/// format not listed here defaults to `FormatSupport{}` - unsupported - so
/// `Device`'s caps-driven pre-validation refuses the rest before this backend
/// is ever asked.
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
            .uniform_offset_alignment = @intCast(@max(limits.min_uniform_buffer_offset_alignment, 1)),
        },
        .features = .{ .sampler_border = true, .sampler_lod_bias = true },
    };

    // Bit n is 2^n samples, as `FormatSupport.sample_counts` counts them.
    const color_counts: u8 = @truncate(@as(u32, @bitCast(limits.framebuffer_color_sample_counts)) & 0xF);
    const depth_counts: u8 = @truncate(@as(u32, @bitCast(limits.framebuffer_depth_sample_counts)) & 0xF);
    for ([_]types.Format{ .rgba8_unorm, .bgra8_unorm, .r8_unorm, .rg8_unorm, .r16_float, .rg16_float, .rgba16_float, .r32_float, .rg32_float, .rgba32_float, .rgb10a2_unorm, .rg11b10_float }) |format| {
        const vk_format = toVkFormat(format).?;
        var fp: vk.gen.types.FormatProperties = undefined;
        vki.getPhysicalDeviceFormatProperties(self.runtime.physical_device, vk_format, &fp);
        const feats = fp.optimal_tiling_features;
        if (!feats.sampled_image) continue;
        answer.formats.set(format, .{
            .sampled = feats.sampled_image,
            .filterable = feats.sampled_image_filter_linear,
            .render_target = feats.color_attachment,
            .blendable = feats.color_attachment_blend,
            .generate_mips = false,
            .sample_counts = if (feats.color_attachment) color_counts | 1 else 0b1,
            .dimensions = std.EnumSet(types.Dimension).initOne(.d2),
        });
    }
    // A depth format the device draws depth into, for 3D, and reads where it
    // can: a shadow map. For a depth format, filtered means a sampler that
    // compares filters it.
    for ([_]types.Format{ .depth16_unorm, .depth24_stencil8, .depth32_float, .depth32_float_stencil8 }) |format| {
        var fp: vk.gen.types.FormatProperties = undefined;
        vki.getPhysicalDeviceFormatProperties(self.runtime.physical_device, toVkFormat(format).?, &fp);
        if (!fp.optimal_tiling_features.depth_stencil_attachment) continue;
        answer.formats.set(format, .{
            .sampled = fp.optimal_tiling_features.sampled_image,
            .filterable = fp.optimal_tiling_features.sampled_image_filter_linear,
            .render_target = true,
            .blendable = false,
            .generate_mips = false,
            .sample_counts = depth_counts | 1,
            .dimensions = std.EnumSet(types.Dimension).initOne(.d2),
        });
    }
    return answer;
}

// -------------------------------------------------------------------------
// Recording: every submit, upload and readback takes the next slot of the
// ring, and hands it to the queue without waiting for it
// -------------------------------------------------------------------------

const forever: u64 = ~@as(u64, 0);

/// A slot's fence and semaphores; its command buffer goes with the pool.
fn destroySlot(vkd: vk.gen.commands.Device, device: vk.Device, which: *Slot) void {
    if (which.fence != .none) vkd.destroyFence(device, which.fence, null);
    which.fence = .none;
    for (&which.acquires) |*semaphore| {
        if (semaphore.* != .none) vkd.destroySemaphore(device, semaphore.*, null);
        semaphore.* = .none;
    }
}

/// The serial the recording under way gets when it is submitted.
pub fn recordingSerial(self: *const Vk) u64 {
    return self.submitted + 1;
}

/// Learn what the GPU has finished since last asked, without waiting.
pub fn poll(self: *Vk) void {
    const vkd = self.runtime.vkd;
    for (&self.slots) |*one| {
        if (one.serial <= self.completed) continue;
        const status = vkd.getFenceStatus(self.runtime.device, one.fence);
        if (status == .success) self.completed = @max(self.completed, one.serial);
    }
}

/// Wait until the GPU has done the submission `serial` and all before it.
pub fn waitFor(self: *Vk, serial: u64) types.Error!void {
    if (serial <= self.completed) return;
    for (&self.slots) |*one| {
        if (one.serial != serial) continue;
        _ = self.runtime.vkd.waitForFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{one.fence}, vk.gen.types.vk_true, forever).check() catch return error.DeviceLost;
        break;
    }
    // A serial no slot holds any more was waited for when its slot came
    // round again; the queue does them in order.
    self.completed = @max(self.completed, serial);
}

/// Wait for everything submitted, and free what waited for it.
pub fn waitIdle(self: *Vk) void {
    _ = self.runtime.vkd.deviceWaitIdle(self.runtime.device).check() catch {};
    self.completed = self.submitted;
    collect(self);
}

/// Free `what` once the GPU is past everything recorded so far - the open
/// recording too - or now when it already is.
pub fn bury(self: *Vk, what: @FieldType(Grave, "what")) void {
    if (!self.recording and self.completed >= self.submitted) return free(self, what);
    const serial = if (self.recording) recordingSerial(self) else self.submitted;
    self.graveyard.append(self.gpa, .{ .serial = serial, .what = what }) catch {
        flush(self) catch {};
        // No room to wait: wait for it instead.
        waitIdle(self);
        free(self, what);
    };
}

/// Free what the GPU is done with.
pub fn collect(self: *Vk) void {
    var i: usize = 0;
    while (i < self.graveyard.items.len) {
        const grave = self.graveyard.items[i];
        if (grave.serial > self.completed) {
            i += 1;
            continue;
        }
        _ = self.graveyard.swapRemove(i);
        free(self, grave.what);
    }
}

fn free(self: *Vk, what: @FieldType(Grave, "what")) void {
    switch (what) {
        .buffer => |version| resources.freeVersion(self, version),
        .texture => |res| resources.freeTexture(self, res),
        .sampler => |res| resources.freeSampler(self, res),
        .pipeline => |res| resources.freePipeline(self, res),
        .framebuffer => |fb| self.runtime.vkd.destroyFramebuffer(self.runtime.device, fb, null),
    }
}

/// The slot being recorded into.
pub fn recordingSlot(self: *Vk) *Slot {
    return &self.slots[self.at];
}

/// Start recording into the next slot, once the GPU is done with it.
pub fn record(self: *Vk) types.Error!void {
    const vkd = self.runtime.vkd;
    self.at = (self.at + 1) % ring_size;
    const next = &self.slots[self.at];
    try waitFor(self, next.serial);
    for (next.descriptor_pools.items) |pool| {
        _ = vkd.resetDescriptorPool(self.runtime.device, pool, 0).check() catch return error.Failed;
    }
    next.pool_index = 0;
    poll(self);
    collect(self);
    self.command_buffer = next.command_buffer;
    _ = vkd.resetCommandBuffer(self.command_buffer, .{}).check() catch return error.Failed;
    _ = vkd.beginCommandBuffer(self.command_buffer, &.{ .flags = .{ .one_time_submit = true } }).check() catch return error.Failed;
    self.recording = true;
    self.in_pass = false;
    self.acquired_count = 0;
    self.lists = 0;
    self.has_work = false;
}

/// The recording that is open, or a new one.
pub fn ensureRecording(self: *Vk) types.Error!void {
    if (!self.recording) try record(self);
}

/// Hand the open recording to the queue, if there is one.
pub fn flush(self: *Vk) types.Error!void {
    if (self.recording) try finish(self);
}

/// Stop recording and hand it to the queue - waiting there for every image
/// this recording acquired - without waiting for the GPU. The fence is
/// reset only here, right before the submit that signals it, so a
/// recording given up on never leaves it waiting for a signal that will not
/// come.
pub fn finish(self: *Vk) types.Error!void {
    const vkd = self.runtime.vkd;
    const now = recordingSlot(self);
    self.recording = false;
    _ = vkd.endCommandBuffer(self.command_buffer).check() catch return error.Failed;

    var wait_stages: [max_surfaces_per_submit]vk.gen.types.PipelineStageFlags = @splat(.{ .color_attachment_output = true });
    const waits = self.acquired_count;
    self.acquired_count = 0;
    const cmd_buffers = [_]vk.gen.types.CommandBuffer{self.command_buffer};
    const submit_info = [_]vk.gen.types.SubmitInfo{.{
        .wait_semaphore_count = @intCast(waits),
        .wait_semaphores = if (waits > 0) &now.acquires else null,
        .wait_dst_stage_mask = if (waits > 0) &wait_stages else null,
        .command_buffer_count = cmd_buffers.len,
        .command_buffers = &cmd_buffers,
    }};
    _ = vkd.resetFences(self.runtime.device, 1, &[_]vk.gen.types.Fence{now.fence}).check() catch return error.Failed;
    _ = vkd.queueSubmit(self.runtime.graphics_queue, 1, &submit_info, now.fence).check() catch |err| switch (err) {
        error.DeviceLost => return error.DeviceLost,
        else => return error.Failed,
    };
    self.submitted += 1;
    now.serial = self.submitted;
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

/// Walks the `Command` union once, recording onto the end of the open
/// recording: see the module doc.
fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) types.Error!void {
    const self = cast(impl);
    const vkd = self.runtime.vkd;

    try ensureRecording(self);
    errdefer abandon(self);
    self.current_pipeline = null;
    self.pipeline_dirty = false;
    self.current_ubo = @splat(null);
    self.current_tex = @splat(null);
    self.bindings_dirty = true;

    for (list) |command| {
        // Read at each command: a draw into a surface may hand what came
        // before it over and start another recording.
        const cmd = self.command_buffer;
        const serial = recordingSerial(self);
        defer self.has_work = true;
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
                res.current.busy = serial;
                const buffers = [_]vk.gen.types.Buffer{res.current.buffer};
                const offsets = [_]vk.gen.types.DeviceSize{b.offset};
                vkd.cmdBindVertexBuffers(cmd, b.slot, 1, &buffers, &offsets);
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                res.current.busy = serial;
                vkd.cmdBindIndexBuffer(cmd, res.current.buffer, 0, if (b.format == .u16) .uint16 else .uint32);
            },
            .set_uniform_buffer => |b| {
                if (b.slot >= uniform_slots) return error.Unsupported;
                self.current_ubo[b.slot] = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.current_ubo_range[b.slot] = b;
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

    self.lists += 1;
    if (self.lists >= max_lists) try finish(self);
}

/// Where a pass's samples are averaged into when it ends: a texture's view, or
/// a swapchain image's, and the layout it is left in.
const ResolveInto = struct {
    view: vk.gen.types.ImageView,
    final: vk.gen.types.ImageLayout,
};

/// The image of `surface` this recording draws into, acquired the first time
/// it is asked for. Its index.
fn surfaceImage(self: *Vk, surface: *SurfaceRes) types.Error!u32 {
    if (surface.acquired == null) {
        // What came before goes now: it has no need to wait for the image
        // the acquire waits for.
        if (self.has_work) {
            try finish(self);
            try record(self);
        }
        if (self.acquired_count >= max_surfaces_per_submit) return error.Unsupported;
        if (try swapchain.acquire(self, surface, recordingSlot(self).acquires[self.acquired_count])) self.acquired_count += 1;
    }
    return surface.acquired.?;
}

fn resolveInto(self: *Vk, device: *Device, target: types.RenderTarget) types.Error!ResolveInto {
    return switch (target) {
        .surface => |h| blk: {
            const surface = as(SurfaceRes, device.surfaces.get(h).?.native);
            const index = try surfaceImage(self, surface);
            surface.drawn[index] = true;
            break :blk .{ .view = surface.views[index], .final = .present_src_khr };
        },
        .texture => |h| .{ .view = as(TextureRes, device.textures.get(h).?.native).view, .final = .shader_read_only_optimal },
    };
}

fn beginPass(self: *Vk, device: *Device, pass: types.RenderPassDesc) types.Error!void {
    if (pass.extra_colors.len > 0) return error.Unsupported;
    const depth_texture: ?*TextureRes = if (pass.depth) |depth| as(TextureRes, device.textures.get(depth.texture).?.native) else null;
    const depth: swapchain.Depth = if (pass.depth) |held| .{ .format = depth_texture.?.vk_format, .load = swapchain.mapLoadOp(held.load), .sampled = depth_texture.?.sampled_depth } else .none;

    var framebuffer: vk.gen.types.Framebuffer = .none;
    var render_pass: vk.gen.types.RenderPass = .none;
    var format: vk.gen.types.Format = .undefined;
    var extent: [2]u32 = undefined;
    var color_view: vk.gen.types.ImageView = .none;
    var samples: u8 = if (depth_texture) |held| held.samples else 1;
    var resolve: ?ResolveInto = null;
    if (pass.color) |color| switch (color.target) {
        .surface => |h| {
            const surface = as(SurfaceRes, device.surfaces.get(h).?.native);
            const index = try surfaceImage(self, surface);
            const load = swapchain.mapLoadOp(color.load);
            // An image drawn into before is presentable; one that never was
            // has nothing in it to keep.
            const initial: vk.gen.types.ImageLayout = if (load == .load and surface.drawn[index]) .present_src_khr else .undefined;
            surface.drawn[index] = true;
            render_pass = try swapchain.getRenderPass(self, surface.format, load, initial, .present_src_khr, depth, 1, null);
            framebuffer = surface.framebuffers[index];
            color_view = surface.views[index];
            format = surface.format;
            extent = .{ surface.width, surface.height };
        },
        .texture => |h| {
            const texture = as(TextureRes, device.textures.get(h).?.native);
            const load = swapchain.mapLoadOp(color.load);
            samples = texture.samples;
            if (texture.samples > 1) {
                // Drawn into and resolved, never sampled: it stays where
                // colour is written, and its samples are averaged into the
                // resolve target when the pass ends.
                const into = color.resolve orelse return error.Unsupported;
                resolve = try resolveInto(self, device, into);
                const initial: vk.gen.types.ImageLayout = if (load == .load) .color_attachment_optimal else .undefined;
                render_pass = try swapchain.getRenderPass(self, texture.vk_format, load, initial, .color_attachment_optimal, depth, texture.samples, resolve.?.final);
            } else {
                if (texture.framebuffer == .none or color.resolve != null) return error.Unsupported;
                const initial: vk.gen.types.ImageLayout = if (load == .load) .shader_read_only_optimal else .undefined;
                render_pass = try swapchain.getRenderPass(self, texture.vk_format, load, initial, .shader_read_only_optimal, depth, 1, null);
                framebuffer = texture.framebuffer;
            }
            color_view = texture.view;
            format = texture.vk_format;
            extent = .{ texture.width, texture.height };
        },
    } else {
        // Depth alone: a shadow map, a depth prepass.
        render_pass = try swapchain.getRenderPass(self, .undefined, .dont_care, .undefined, .undefined, depth, samples, null);
        extent = .{ depth_texture.?.width, depth_texture.?.height };
    }

    // A pass with depth, or one that resolves, has a framebuffer of its own:
    // its colour view, its depth view and the view it resolves into, freed
    // once the GPU is past this recording.
    if (depth_texture != null or resolve != null) {
        var views: [3]vk.gen.types.ImageView = undefined;
        var count: u32 = 0;
        if (color_view != .none) {
            views[count] = color_view;
            count += 1;
        }
        if (depth_texture) |held| {
            views[count] = held.view;
            count += 1;
        }
        if (resolve) |into| {
            views[count] = into.view;
            count += 1;
        }
        framebuffer = .none;
        _ = self.runtime.vkd.createFramebuffer(self.runtime.device, &.{
            .render_pass = render_pass,
            .attachment_count = count,
            .attachments = &views,
            .width = extent[0],
            .height = extent[1],
            .layers = 1,
        }, null, &framebuffer).check() catch return error.Failed;
        bury(self, .{ .framebuffer = framebuffer });
    }

    var clears: [2]vk.gen.types.ClearValue = undefined;
    var clear_count: u32 = 0;
    if (pass.color) |color| {
        clears[clear_count] = .{ .color = .{ .float32 = color.clear_color } };
        clear_count += 1;
    }
    if (pass.depth) |held| {
        clears[clear_count] = .{ .depth_stencil = .{ .depth = held.clear_depth, .stencil = held.clear_stencil } };
        clear_count += 1;
    }
    self.runtime.vkd.cmdBeginRenderPass(self.command_buffer, &.{
        .render_pass = render_pass,
        .framebuffer = framebuffer,
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = extent[0], .height = extent[1] } },
        .clear_value_count = clear_count,
        .clear_values = &clears,
    }, .@"inline");
    self.in_pass = true;
    self.pass_format = format;
    self.pass_depth = depth.format;
    self.pass_samples = samples;
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
        if (res.samples != self.pass_samples) return error.InvalidArgument;
        const pipeline = try resources.pipelineFor(self, res, self.pass_format, self.pass_depth);
        self.runtime.vkd.cmdBindPipeline(self.command_buffer, .graphics, pipeline);
        self.pipeline_dirty = false;
    }
    try flushBindings(self);
}

fn addDescriptorPool(self: *Vk) types.Error!void {
    const pools = &recordingSlot(self).descriptor_pools;
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
    pools.append(self.gpa, pool) catch {
        self.runtime.vkd.destroyDescriptorPool(self.runtime.device, pool, null);
        return error.OutOfMemory;
    };
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
        res.current.busy = recordingSerial(self);
        info_.* = .{ .buffer = res.current.buffer, .offset = 0, .range = vk.gen.types.whole_size };
        if (self.current_ubo[i] != null) {
            const range = self.current_ubo_range[i];
            if (!range.whole()) info_.* = .{
                .buffer = res.current.buffer,
                .offset = range.offset,
                .range = if (range.size == 0) res.size - range.offset else range.size,
            };
        }
    }
    var texture_infos: [texture_slots]vk.gen.types.DescriptorImageInfo = undefined;
    for (&texture_infos, 0..) |*info_, i| {
        const binding = self.current_tex[i];
        const tex = if (binding) |b| b.texture else self.dummy_texture;
        const samp = if (binding) |b| b.sampler else self.dummy_sampler;
        info_.* = .{ .sampler = samp.sampler, .image_view = tex.sampledView(), .image_layout = .shader_read_only_optimal };
    }

    const set_layouts = [_]vk.gen.types.DescriptorSetLayout{ self.set_layout_uniforms, self.set_layout_textures };
    var sets: [2]vk.gen.types.DescriptorSet = undefined;
    const now = recordingSlot(self);
    while (true) {
        if (now.pool_index >= now.descriptor_pools.items.len) try addDescriptorPool(self);
        _ = vkd.allocateDescriptorSets(self.runtime.device, &.{
            .descriptor_pool = now.descriptor_pools.items[now.pool_index],
            .descriptor_set_count = set_layouts.len,
            .set_layouts = &set_layouts,
        }, &sets).check() catch |err| switch (err) {
            error.OutOfPoolMemory, error.FragmentedPool => {
                now.pool_index += 1;
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

test "a buffer the GPU may still read is written into another copy, which keeps the rest of its bytes" {
    var device = try openTestDevice();
    defer device.deinit();
    const self = cast(device.impl);

    const handle = try device.createBuffer(.{ .kind = .vertex, .size = 8, .data = "abcdefgh" });
    const res = as(BufferRes, device.buffers.get(handle).?.native);
    const first = res.current;
    // As if a submission the GPU has not done yet drew with it.
    res.current.busy = self.submitted + 1;
    try device.updateBuffer(handle, 2, "XY");
    try testing.expect(res.current.buffer != first.buffer);
    try testing.expectEqualSlices(u8, "abcdefgh", first.mapped[0..8]);
    try testing.expectEqualSlices(u8, "abXYefgh", res.current.mapped[0..8]);

    // One no submission uses is written where it is.
    const second = res.current.buffer;
    try device.updateBuffer(handle, 0, "Z");
    try testing.expectEqual(second, res.current.buffer);
    try testing.expectEqualSlices(u8, "ZbXYefgh", res.current.mapped[0..8]);
}

test "submits go into one recording, handed over when needed, and what they used is freed once the GPU is done" {
    var device = try openTestDevice();
    defer device.deinit();
    const self = cast(device.impl);

    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    try flush(self);
    const before = self.submitted;
    for (0..12) |_| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 1, 0, 1 } } });
        try cmd.endPass();
        try device.submit();
    }
    // Twelve lists, and nothing handed to the queue yet.
    try testing.expectEqual(before, self.submitted);
    try testing.expect(self.recording);
    // A texture destroyed while the open recording may use it waits in the
    // ground until that recording is done.
    const doomed = try device.createTexture(.{ .width = 2, .height = 2 });
    const graves = self.graveyard.items.len;
    device.destroyTexture(doomed);
    try testing.expectEqual(graves + 1, self.graveyard.items.len);
    try flush(self);
    try testing.expectEqual(before + 1, self.submitted);
    waitIdle(self);
    try testing.expectEqual(self.submitted, self.completed);
    try testing.expectEqual(@as(usize, 0), self.graveyard.items.len);

    const back = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(back);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[0..4].*);
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
    // A chain of levels.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 2 }));
}

test "a depth texture keeps what is nearer, whichever is drawn last, and a pass can write depth alone" {
    var device = try openTestDevice();
    defer device.deinit();

    const shader = try device.createShader(.{ .spirv = .{ .vertex = shaded_vertex, .fragment = shaded_fragment } });
    defer device.destroyShader(shader);
    const attributes = [_]types.VertexAttribute{ .{ .location = 0, .format = .float3, .offset = 0 }, .{ .location = 1, .format = .float4, .offset = 12 } };
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &attributes,
        .buffers = &.{.{ .stride = 28 }},
        .depth = .standard,
        .depth_format = .depth32_float,
    });
    defer device.destroyPipeline(pipeline);
    const only_depth = try device.createPipeline(.{
        .shader = shader,
        .attributes = &attributes,
        .buffers = &.{.{ .stride = 28 }},
        .depth = .standard,
        .color_format = null,
        .depth_format = .depth32_float,
    });
    defer device.destroyPipeline(only_depth);

    // A triangle over the whole target, near and red; then one far and blue.
    const corners = [_]f32{
        -1, -1, 0.2, 1, 0, 0, 1, 3, -1, 0.2, 1, 0, 0, 1, -1, 3, 0.2, 1, 0, 0, 1,
        -1, -1, 0.8, 0, 0, 1, 1, 3, -1, 0.8, 0, 0, 1, 1, -1, 3, 0.8, 0, 0, 1, 1,
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });
    defer device.destroyBuffer(buffer);
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    defer device.destroyTexture(target);
    const depth = device.createTexture(.{ .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } }) catch |err| switch (err) {
        // A device with no 32-bit float depth: every desktop one has it.
        error.Unsupported => return error.SkipZigTest,
        else => return err,
    };
    defer device.destroyTexture(depth);

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } }, .depth = .{ .texture = depth } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.draw(.{ .vertex_count = 6 });
    try cmd.endPass();
    try device.submit();
    {
        const pixels = try device.readTexture(target, testing.allocator);
        defer testing.allocator.free(pixels);
        try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, pixels[(4 * 8 + 4) * 4 ..][0..4]);
    }

    // The near triangle into the depth alone, kept; then the far one with
    // colour, loading that depth: nothing passes, and the clear shows.
    const pass = device.begin();
    try pass.beginPass(.{ .depth = .{ .texture = depth } });
    try pass.setPipeline(only_depth);
    try pass.setVertexBuffer(0, buffer, 0);
    try pass.draw(.{ .vertex_count = 3 });
    try pass.endPass();
    try pass.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 1, 0, 1 } }, .depth = .{ .texture = depth, .load = .load } });
    try pass.setPipeline(pipeline);
    try pass.setVertexBuffer(0, buffer, 0);
    try pass.draw(.{ .vertex_count = 3, .first_vertex = 3 });
    try pass.endPass();
    try device.submit();
    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqualSlices(u8, &.{ 0, 255, 0, 255 }, pixels[(4 * 8 + 4) * 4 ..][0..4]);
}

test "a depth texture drawn into is read by a sampler that compares, and a bias pushes what is drawn back" {
    var device = try openTestDevice();
    defer device.deinit();
    // A device that cannot read 32-bit float depth: every desktop one can.
    if (!device.caps().formatSupport(.depth32_float).sampled) return error.SkipZigTest;

    // A plane at depth one half, drawn into two shadow maps: once as it is,
    // and once pushed back by 2^18 of a 32-bit float's steps at one half,
    // 2^-24 each - a sixty-fourth.
    const flat = try device.createShader(.{ .spirv = .{ .vertex = shaded_vertex, .fragment = shaded_fragment } });
    defer device.destroyShader(flat);
    const plane = [_]f32{ -1, -1, 0.5, 1, 1, 1, 1, 3, -1, 0.5, 1, 1, 1, 1, -1, 3, 0.5, 1, 1, 1, 1 };
    const plane_buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(plane)), .data = std.mem.asBytes(&plane) });
    defer device.destroyBuffer(plane_buffer);
    const attributes = [_]types.VertexAttribute{ .{ .location = 0, .format = .float3, .offset = 0 }, .{ .location = 1, .format = .float4, .offset = 12 } };
    var maps: [2]types.Texture = undefined;
    var pipelines: [2]types.Pipeline = undefined;
    for (&maps, &pipelines, [_]i32{ 0, 1 << 18 }) |*map, *pipeline, bias| {
        pipeline.* = try device.createPipeline(.{
            .shader = flat,
            .attributes = &attributes,
            .buffers = &.{.{ .stride = 28 }},
            .color_format = null,
            .depth_format = .depth32_float,
            .depth = .{ .test_enabled = true, .write = true, .compare = .less, .bias = bias },
        });
        map.* = try device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = map.* } });
        try cmd.setPipeline(pipeline.*);
        try cmd.setVertexBuffer(0, plane_buffer, 0);
        try cmd.draw(.{ .vertex_count = 3 });
        try cmd.endPass();
        try device.submit();
    }
    defer for (maps, pipelines) |map, pipeline| {
        device.destroyTexture(map);
        device.destroyPipeline(pipeline);
    };

    // Read through a sampler that compares: lit where the depth asked about
    // is at or before what is stored.
    const shader = try device.createShader(.{ .spirv = .{ .vertex = corner_vertex, .fragment = compare_fragment } });
    defer device.destroyShader(shader);
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
    });
    defer device.destroyPipeline(pipeline);
    const corners = [_]f32{ -1, -1, 1, -1, -1, 1, 1, 1 };
    const corner_buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });
    defer device.destroyBuffer(corner_buffer);
    const compare = try device.createSampler(.{ .compare = .less_equal });
    defer device.destroySampler(compare);
    const params = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    defer device.destroyBuffer(params);
    const out = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    defer device.destroyTexture(out);

    const cases = [_]struct { map: usize, depth: f32, lit: bool }{
        .{ .map = 0, .depth = 0.3, .lit = true },
        .{ .map = 0, .depth = 0.7, .lit = false },
        .{ .map = 0, .depth = 0.505, .lit = false },
        .{ .map = 1, .depth = 0.505, .lit = true },
        .{ .map = 1, .depth = 0.53, .lit = false },
    };
    for (cases) |case| {
        try device.updateBuffer(params, 0, std.mem.asBytes(&[4]f32{ case.depth, 0, 0, 0 }));
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = out } } });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, corner_buffer, 0);
        try cmd.setUniformBuffer(0, params);
        try cmd.setTexture(0, maps[case.map], compare);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        const pixels = try device.readTexture(out, testing.allocator);
        defer testing.allocator.free(pixels);
        const want: [4]u8 = if (case.lit) .{ 255, 255, 255, 255 } else .{ 0, 0, 0, 255 };
        try testing.expectEqualSlices(u8, &want, pixels[(2 * 4 + 1) * 4 ..][0..4]);
    }
}

test "the sixteenth texture slot is bound" {
    var device = try openTestDevice();
    defer device.deinit();
    if (!device.caps().formatSupport(.depth32_float).sampled) return error.SkipZigTest;

    // The shadow map reader, reading binding 15 rather than 0.
    const last_fragment = comptime blk: {
        var words: [compare_fragment.len]u32 = compare_fragment[0..compare_fragment.len].*;
        for (0..words.len - 3) |i| {
            // OpDecorate %12 Binding 0
            if (words[i] == 0x00040047 and words[i + 1] == 0x0000000c and words[i + 2] == 0x00000021) {
                words[i + 3] = 15;
                break;
            }
        } else @compileError("no binding to move");
        const final = words;
        break :blk final;
    };
    const shader = try device.createShader(.{ .spirv = .{ .vertex = corner_vertex, .fragment = &last_fragment } });
    defer device.destroyShader(shader);
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
    });
    defer device.destroyPipeline(pipeline);
    const corners = [_]f32{ -1, -1, 1, -1, -1, 1, 1, 1 };
    const corner_buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });
    defer device.destroyBuffer(corner_buffer);
    const map = try device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
    defer device.destroyTexture(map);
    const compare = try device.createSampler(.{ .compare = .less_equal });
    defer device.destroySampler(compare);
    const params = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    defer device.destroyBuffer(params);
    const out = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    defer device.destroyTexture(out);

    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = map, .clear_depth = 0.5 } });
        try cmd.endPass();
        try device.submit();
    }
    for ([_]f32{ 0.25, 0.75 }) |depth| {
        try device.updateBuffer(params, 0, std.mem.asBytes(&[4]f32{ depth, 0, 0, 0 }));
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = out } } });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, corner_buffer, 0);
        try cmd.setUniformBuffer(0, params);
        try cmd.setTexture(15, map, compare);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        const pixels = try device.readTexture(out, testing.allocator);
        defer testing.allocator.free(pixels);
        const want: [4]u8 = if (depth < 0.5) .{ 255, 255, 255, 255 } else .{ 0, 0, 0, 255 };
        try testing.expectEqualSlices(u8, &want, pixels[(2 * 4 + 1) * 4 ..][0..4]);
    }
}

test "a float target keeps light over one, read back clamped" {
    var device = try openTestDevice();
    defer device.deinit();
    if (!device.caps().formatSupport(.rgba16_float).render_target) return error.SkipZigTest;

    const bright = try device.createTexture(.{ .width = 4, .height = 4, .format = .rgba16_float, .usage = .{ .sampled = true, .render_target = true } });
    defer device.destroyTexture(bright);
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = bright }, .clear_color = .{ 2, 0.25, 0, 1 } } });
    try cmd.endPass();
    try device.submit();
    const pixels = try device.readTexture(bright, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqualSlices(u8, &.{ 255, 64, 0, 255 }, pixels[(1 * 4 + 1) * 4 ..][0..4]);
}

test "two draws read two parts of one uniform buffer" {
    var device = try openTestDevice();
    defer device.deinit();
    const shader = try device.createShader(.{ .spirv = .{ .vertex = shaded_vertex, .fragment = look_fragment } });
    defer device.destroyShader(shader);
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{ .{ .location = 0, .format = .float3, .offset = 0 }, .{ .location = 1, .format = .float4, .offset = 12 } },
        .buffers = &.{.{ .stride = 28 }},
    });
    defer device.destroyPipeline(pipeline);
    // The left half, then the right half; the colour is the block's.
    const w = [4]f32{ 1, 1, 1, 1 };
    const halves = [_][7]f32{
        .{ -1, -1, 0 } ++ w, .{ -1, 1, 0 } ++ w, .{ 0, -1, 0 } ++ w, .{ 0, -1, 0 } ++ w, .{ -1, 1, 0 } ++ w, .{ 0, 1, 0 } ++ w,
        .{ 0, -1, 0 } ++ w,  .{ 0, 1, 0 } ++ w,  .{ 1, -1, 0 } ++ w, .{ 1, -1, 0 } ++ w, .{ 0, 1, 0 } ++ w,  .{ 1, 1, 0 } ++ w,
    };
    const corners = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(halves)), .data = std.mem.asBytes(&halves) });
    defer device.destroyBuffer(corners);
    const step = device.caps().limits.uniform_offset_alignment;
    const colours = try device.createBuffer(.{ .kind = .uniform, .size = step + 16 });
    defer device.destroyBuffer(colours);
    try device.updateBuffer(colours, 0, std.mem.asBytes(&[4]f32{ 1, 0, 0, 1 }));
    try device.updateBuffer(colours, step, std.mem.asBytes(&[4]f32{ 0, 0, 1, 1 }));
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .sampled = true, .render_target = true } });
    defer device.destroyTexture(target);
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, corners, 0);
    try cmd.setUniformBufferRange(0, colours, 0, 16);
    try cmd.draw(.{ .vertex_count = 6 });
    try cmd.setUniformBufferRange(0, colours, step, 0);
    try cmd.draw(.{ .vertex_count = 6, .first_vertex = 6 });
    try cmd.endPass();
    try device.submit();
    const drawn = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(drawn);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, drawn[(4 * 8 + 1) * 4 ..][0..4].*);
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, drawn[(4 * 8 + 6) * 4 ..][0..4].*);
}

test "a multisampled target, with multisampled depth, is resolved into a texture at the end of the pass" {
    var device = try openTestDevice();
    defer device.deinit();
    const c = device.caps();
    if (!c.formatSupport(.rgba16_float).supportsSamples(4) or !c.formatSupport(.depth32_float).supportsSamples(4)) return error.SkipZigTest;

    const shader = try device.createShader(.{ .spirv = .{ .vertex = shaded_vertex, .fragment = shaded_fragment } });
    defer device.destroyShader(shader);
    const attributes = [_]types.VertexAttribute{ .{ .location = 0, .format = .float3, .offset = 0 }, .{ .location = 1, .format = .float4, .offset = 12 } };
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &attributes,
        .buffers = &.{.{ .stride = 28 }},
        .depth = .standard,
        .color_format = .rgba16_float,
        .depth_format = .depth32_float,
        .samples = 4,
    });
    defer device.destroyPipeline(pipeline);

    // A red triangle over blue, with slanted sides, so that some pixels are
    // only partly covered.
    const corners = [_]f32{ -0.8, -0.8, 0.5, 1, 0, 0, 1, 0.8, -0.8, 0.5, 1, 0, 0, 1, 0, 0.8, 0.5, 1, 0, 0, 1 };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });
    defer device.destroyBuffer(buffer);
    const msaa = try device.createTexture(.{ .width = 32, .height = 32, .format = .rgba16_float, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    defer device.destroyTexture(msaa);
    const depth = try device.createTexture(.{ .width = 32, .height = 32, .format = .depth32_float, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    defer device.destroyTexture(depth);
    const resolved = try device.createTexture(.{ .width = 32, .height = 32, .format = .rgba16_float, .usage = .{ .sampled = true, .render_target = true } });
    defer device.destroyTexture(resolved);

    // Twice: a second pass loads what the first left, and resolves again.
    for (0..2) |round| {
        const cmd = device.begin();
        try cmd.beginPass(.{
            .color = .{ .target = .{ .texture = msaa }, .clear_color = .{ 0, 0, 1, 1 }, .load = if (round == 0) .clear else .load, .resolve = .{ .texture = resolved } },
            .depth = .{ .texture = depth },
        });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, buffer, 0);
        try cmd.draw(.{ .vertex_count = 3 });
        try cmd.endPass();
        try device.submit();
    }

    const pixels = try device.readTexture(resolved, testing.allocator);
    defer testing.allocator.free(pixels);
    // Inside is red, outside blue, and the edge neither: its samples averaged.
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, pixels[(16 * 32 + 16) * 4 ..][0..4]);
    try testing.expectEqualSlices(u8, &.{ 0, 0, 255, 255 }, pixels[(1 * 32 + 1) * 4 ..][0..4]);
    var blended: usize = 0;
    for (0..32 * 32) |i| {
        const pixel = pixels[i * 4 ..][0..4];
        if (pixel[0] > 0 and pixel[0] < 255 and pixel[2] > 0 and pixel[2] < 255) blended += 1;
    }
    try testing.expect(blended > 10);
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

// A pair that draws something to look at, compiled from
//
//     layout(location = 0) in vec3 position;
//     layout(location = 1) in vec4 color;
//     layout(location = 0) out vec4 shade;
//     void main() { gl_Position = vec4(position, 1); shade = color; }
//
// and a fragment stage that writes `shade`: the depth test's.
const shaded_vertex: []const u32 = &.{
    0x07230203, 0x00010000, 0x000d000b, 0x0000001f, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001b,
    0x0000001d, 0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000, 0x0000000b,
    0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b,
    0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004,
    0x00040047, 0x00000012, 0x0000001e, 0x00000000, 0x00040047, 0x0000001b, 0x0000001e, 0x00000000,
    0x00040047, 0x0000001d, 0x0000001e, 0x00000001, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001,
    0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006,
    0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
    0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e,
    0x0000000f, 0x00000000, 0x00040017, 0x00000010, 0x00000006, 0x00000003, 0x00040020, 0x00000011,
    0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006,
    0x00000014, 0x3f800000, 0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x0004003b, 0x00000019,
    0x0000001b, 0x00000003, 0x00040020, 0x0000001c, 0x00000001, 0x00000007, 0x0004003b, 0x0000001c,
    0x0000001d, 0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003d, 0x00000010, 0x00000013, 0x00000012, 0x00050051, 0x00000006, 0x00000015,
    0x00000013, 0x00000000, 0x00050051, 0x00000006, 0x00000016, 0x00000013, 0x00000001, 0x00050051,
    0x00000006, 0x00000017, 0x00000013, 0x00000002, 0x00070050, 0x00000007, 0x00000018, 0x00000015,
    0x00000016, 0x00000017, 0x00000014, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x0000000f,
    0x0003003e, 0x0000001a, 0x00000018, 0x0004003d, 0x00000007, 0x0000001e, 0x0000001d, 0x0003003e,
    0x0000001b, 0x0000001e, 0x000100fd, 0x00010038,
};

/// A quad from its corners, and the uv across it, top row first.
const corner_vertex: []const u32 = &.{
    0x07230203, 0x00010000, 0x0008000b, 0x00000024, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00000019,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
    0x00000009, 0x00007675, 0x00040005, 0x0000000b, 0x6e726f63, 0x00007265, 0x00060005, 0x00000017,
    0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000017, 0x00000000, 0x505f6c67,
    0x7469736f, 0x006e6f69, 0x00070006, 0x00000017, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
    0x00000000, 0x00070006, 0x00000017, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e,
    0x00070006, 0x00000017, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005,
    0x00000019, 0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00030047, 0x00000017, 0x00000002, 0x00050048, 0x00000017, 0x00000000,
    0x0000000b, 0x00000000, 0x00050048, 0x00000017, 0x00000001, 0x0000000b, 0x00000001, 0x00050048,
    0x00000017, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x00000017, 0x00000003, 0x0000000b,
    0x00000004, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000d,
    0x3f000000, 0x0004002b, 0x00000006, 0x0000000e, 0xbf000000, 0x0005002c, 0x00000007, 0x0000000f,
    0x0000000d, 0x0000000e, 0x00040017, 0x00000013, 0x00000006, 0x00000004, 0x00040015, 0x00000014,
    0x00000020, 0x00000000, 0x0004002b, 0x00000014, 0x00000015, 0x00000001, 0x0004001c, 0x00000016,
    0x00000006, 0x00000015, 0x0006001e, 0x00000017, 0x00000013, 0x00000006, 0x00000016, 0x00000016,
    0x00040020, 0x00000018, 0x00000003, 0x00000017, 0x0004003b, 0x00000018, 0x00000019, 0x00000003,
    0x00040015, 0x0000001a, 0x00000020, 0x00000001, 0x0004002b, 0x0000001a, 0x0000001b, 0x00000000,
    0x0004002b, 0x00000006, 0x0000001d, 0x00000000, 0x0004002b, 0x00000006, 0x0000001e, 0x3f800000,
    0x00040020, 0x00000022, 0x00000003, 0x00000013, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x0000000c, 0x0000000b, 0x00050085,
    0x00000007, 0x00000010, 0x0000000c, 0x0000000f, 0x00050050, 0x00000007, 0x00000011, 0x0000000d,
    0x0000000d, 0x00050081, 0x00000007, 0x00000012, 0x00000010, 0x00000011, 0x0003003e, 0x00000009,
    0x00000012, 0x0004003d, 0x00000007, 0x0000001c, 0x0000000b, 0x00050051, 0x00000006, 0x0000001f,
    0x0000001c, 0x00000000, 0x00050051, 0x00000006, 0x00000020, 0x0000001c, 0x00000001, 0x00070050,
    0x00000013, 0x00000021, 0x0000001f, 0x00000020, 0x0000001d, 0x0000001e, 0x00050041, 0x00000022,
    0x00000023, 0x00000019, 0x0000001b, 0x0003003e, 0x00000023, 0x00000021, 0x000100fd, 0x00010038,
};

/// `layout(set = 1, binding = 0) uniform sampler2DShadow map` compared at
/// `uv` with `param.x` of the block at set 0, binding 0, written as grey.
const compare_fragment: []const u32 = &.{
    0x07230203, 0x00010000, 0x0008000b, 0x00000028, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000010, 0x00000024, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000008, 0x00000076, 0x00030005, 0x0000000c, 0x0070616d, 0x00030005,
    0x00000010, 0x00007675, 0x00040005, 0x00000013, 0x61726150, 0x0000736d, 0x00050006, 0x00000013,
    0x00000000, 0x61726170, 0x0000006d, 0x00030005, 0x00000015, 0x00000000, 0x00040005, 0x00000024,
    0x67726174, 0x00007465, 0x00040047, 0x0000000c, 0x00000021, 0x00000000, 0x00040047, 0x0000000c,
    0x00000022, 0x00000001, 0x00040047, 0x00000010, 0x0000001e, 0x00000000, 0x00030047, 0x00000013,
    0x00000002, 0x00050048, 0x00000013, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000015,
    0x00000021, 0x00000000, 0x00040047, 0x00000015, 0x00000022, 0x00000000, 0x00040047, 0x00000024,
    0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00090019, 0x00000009,
    0x00000006, 0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b,
    0x0000000a, 0x00000009, 0x00040020, 0x0000000b, 0x00000000, 0x0000000a, 0x0004003b, 0x0000000b,
    0x0000000c, 0x00000000, 0x00040017, 0x0000000e, 0x00000006, 0x00000002, 0x00040020, 0x0000000f,
    0x00000001, 0x0000000e, 0x0004003b, 0x0000000f, 0x00000010, 0x00000001, 0x00040017, 0x00000012,
    0x00000006, 0x00000004, 0x0003001e, 0x00000013, 0x00000012, 0x00040020, 0x00000014, 0x00000002,
    0x00000013, 0x0004003b, 0x00000014, 0x00000015, 0x00000002, 0x00040015, 0x00000016, 0x00000020,
    0x00000001, 0x0004002b, 0x00000016, 0x00000017, 0x00000000, 0x00040015, 0x00000018, 0x00000020,
    0x00000000, 0x0004002b, 0x00000018, 0x00000019, 0x00000000, 0x00040020, 0x0000001a, 0x00000002,
    0x00000006, 0x00040017, 0x0000001d, 0x00000006, 0x00000003, 0x00040020, 0x00000023, 0x00000003,
    0x00000012, 0x0004003b, 0x00000023, 0x00000024, 0x00000003, 0x0004002b, 0x00000006, 0x00000026,
    0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c,
    0x0004003d, 0x0000000e, 0x00000011, 0x00000010, 0x00060041, 0x0000001a, 0x0000001b, 0x00000015,
    0x00000017, 0x00000019, 0x0004003d, 0x00000006, 0x0000001c, 0x0000001b, 0x00050051, 0x00000006,
    0x0000001e, 0x00000011, 0x00000000, 0x00050051, 0x00000006, 0x0000001f, 0x00000011, 0x00000001,
    0x00060050, 0x0000001d, 0x00000020, 0x0000001e, 0x0000001f, 0x0000001c, 0x00050051, 0x00000006,
    0x00000021, 0x00000020, 0x00000002, 0x00060059, 0x00000006, 0x00000022, 0x0000000d, 0x00000020,
    0x00000021, 0x0003003e, 0x00000008, 0x00000022, 0x0004003d, 0x00000006, 0x00000025, 0x00000008,
    0x00070050, 0x00000012, 0x00000027, 0x00000025, 0x00000025, 0x00000025, 0x00000026, 0x0003003e,
    0x00000024, 0x00000027, 0x000100fd, 0x00010038,
};

/// `layout(set = 0, binding = 0) uniform Look { vec4 colour; }`, written
/// to the target as it is: glslangValidator's output for it.
const look_fragment: []const u32 = &.{
    0x07230203, 0x00010000, 0x0008000b, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00040005, 0x00000009, 0x67726174, 0x00007465, 0x00040005, 0x0000000a, 0x6b6f6f4c, 0x00000000,
    0x00050006, 0x0000000a, 0x00000000, 0x6f6c6f63, 0x00007275, 0x00030005, 0x0000000c, 0x00000000,
    0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00050048,
    0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c, 0x00000021, 0x00000000,
    0x00040047, 0x0000000c, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
    0x0003001e, 0x0000000a, 0x00000007, 0x00040020, 0x0000000b, 0x00000002, 0x0000000a, 0x0004003b,
    0x0000000b, 0x0000000c, 0x00000002, 0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b,
    0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000002, 0x00000007, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000f,
    0x00000010, 0x0000000c, 0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e,
    0x00000009, 0x00000011, 0x000100fd, 0x00010038,
};

const shaded_fragment: []const u32 = &.{
    0x07230203, 0x00010000, 0x000d000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
    0x00000004, 0x00000007, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
    0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020, 0x0000000a,
    0x00000001, 0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x0000000c,
    0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
};
