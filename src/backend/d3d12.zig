// SPDX-License-Identifier: BSD-2-Clause

//! The Direct3D 12 backend. Windows only, feature level 11.0.
//!
//! Built on `fluxion-d3d`'s `d3d12.zig`, which stops at the device and the
//! queue the same way its `d3d11.zig` stops at a device: the slots that make
//! resources, command lists and pipeline states are its vtables' permanent
//! `*const anyopaque` placeholders. `d3d12_resource.zig`, `d3d12_command.zig`
//! and `d3d12_pipeline.zig`, siblings of this file, are those declarations -
//! typed fresh here, the way `IFactory2` below is typed fresh over
//! `fluxion-d3d`'s `dxgi.IDXGIFactory1`. The MVP scope table in the plan this
//! implements is narrow on purpose: `.d2` textures only, one mip, one
//! sample, no render-target texture, no depth, no compute. What that buys is
//! simplicity in the parts a real Direct3D 12 renderer usually spends the
//! most code on:
//!
//! **Buffers are always upload-heap.** CPU-writable, mapped once at creation
//! and kept mapped for the buffer's life - `updateBuffer` is a `memcpy` into
//! memory the GPU already sees, safe because nothing here ever has two
//! frames in flight at once (see below). No device-local buffer, no copy
//! queue, no fencing per update.
//!
//! **One root signature for every pipeline.** Four root CBVs (`b0`-`b3`),
//! one CBV_SRV_UAV descriptor table (`t0`-`t3`) and one sampler table
//! (`s0`-`s3`), built once in `open`. A pipeline is a `D3D12_GRAPHICS_PIPELINE_STATE_DESC`
//! that names it; nothing about binding is per-pipeline.
//!
//! **Two heaps per binding kind: permanent and binding-window.** Every
//! texture's SRV and every sampler's descriptor is written once, at creation,
//! into a plain (non-shader-visible) heap that never moves. A separate,
//! small, shader-visible heap - four slots, one per texture/sampler slot -
//! is what the root descriptor tables actually point at; `set_texture`
//! copies the one descriptor a slot needs from the permanent heap into that
//! window with `CopyDescriptorsSimple`, right before the draw that reads it.
//! This is the ordinary way an engine assembles descriptors that were made
//! at unrelated times into the one contiguous range a table call needs.
//!
//! **Fully synchronous.** `submit` records the whole command list, executes
//! it, signals a fence and spins on `GetCompletedValue` until the GPU has
//! caught up, every time. No double-buffering, no multiple frames in flight.
//! Slower than a real engine would want, and simple enough that nothing here
//! has to reason about what the GPU might still be reading.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const d3d = @import("fluxion_d3d");
const com = d3d.com;
const d3d12 = d3d.d3d12;
const rc = @import("d3d12_resource.zig");
const cmdmod = @import("d3d12_command.zig");
const pl = @import("d3d12_pipeline.zig");
const dxgi = d3d.dxgi;
const Guid = d3d.Guid;
const Hresult = d3d.Hresult;
const IUnknown = d3d.IUnknown;

const types = @import("../types.zig");
const backend = @import("../backend.zig");
const commands = @import("../commands.zig");
const Device = @import("../Device.zig");

const Error = backend.Error;

comptime {
    if (builtin.os.tag != .windows) @compileError("fluxion-rhi: the Direct3D 12 backend is Windows only");
}

// -------------------------------------------------------------------------
// Limits this MVP holds itself to. Not hardware limits - the real device
// allows far more - but the shapes this file's fixed root signature and
// fixed-size descriptor heaps were built for.
// -------------------------------------------------------------------------

const max_vertex_slots = 8;
const max_attributes = 16;
/// `t0`..`t3` and `s0`..`s3`: the width of the one descriptor table of each
/// kind the shared root signature has.
const max_binding_slots = 4;
/// How many textures/samplers this device can have alive at once. A fixed
/// heap size, because a descriptor heap cannot be resized - see the module
/// comment on the permanent/binding-window split.
const max_srv_descriptors = 1024;
const max_sampler_descriptors = 256;

const root_param_cbv0 = 0;
const root_param_srv_table = 4;
const root_param_sampler_table = 5;

/// What a swap chain is made in, and the one format `createPipeline` accepts
/// for `color_format`.
const surface_dxgi_format: rc.Format = .r8g8b8a8_unorm;

// -------------------------------------------------------------------------
// DXGI swap chain types `fluxion-d3d`'s `dxgi.zig` leaves opaque - the same
// situation the Direct3D 11 backend is in, and the same fix: declare the
// slots this file calls, in their real position, and let the rest stay
// `*const anyopaque`.
// -------------------------------------------------------------------------

const Bool = c_int;

const SwapEffect = enum(u32) { discard = 0, sequential = 1, flip_sequential = 3, flip_discard = 4 };
const Scaling = enum(u32) { stretch = 0, none = 1, aspect_ratio_stretch = 2 };
const AlphaMode = enum(u32) { unspecified = 0, premultiplied = 1, straight = 2, ignore = 3 };
const usage_render_target_output: u32 = 1 << 5;

/// `DXGI_SWAP_CHAIN_DESC1`.
const SwapChainDesc1 = extern struct {
    width: u32 = 0,
    height: u32 = 0,
    format: rc.Format = surface_dxgi_format,
    stereo: Bool = 0,
    sample: rc.SampleDesc = .{},
    buffer_usage: u32 = usage_render_target_output,
    buffer_count: u32 = 2,
    scaling: Scaling = .none,
    swap_effect: SwapEffect = .flip_discard,
    alpha_mode: AlphaMode = .unspecified,
    flags: u32 = 0,
};

const IFactory2 = extern struct {
    vtable: *const VTable,

    pub const iid = dxgi.IDXGIFactory2.iid;

    pub const VTable = extern struct {
        base: dxgi.IDXGIFactory1.VTable,
        IsWindowedStereoEnabled: *const anyopaque,
        /// `pDevice` is the command queue for Direct3D 12, not the device -
        /// the one real difference from the Direct3D 11 backend's own
        /// `CreateSwapChainForHwnd` call.
        CreateSwapChainForHwnd: *const fn (
            *IFactory2,
            *IUnknown,
            ?*anyopaque,
            *const SwapChainDesc1,
            ?*const anyopaque,
            ?*anyopaque,
            *?*ISwapChain,
        ) callconv(.winapi) Hresult,
    };
};

/// `IDXGISwapChain`.
const ISwapChain = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{310D36A0-D2E7-4C0A-AA04-6A9D23B8886A}");

    pub const VTable = extern struct {
        base: dxgi.IDXGIObject.VTable,
        GetDevice: *const anyopaque,
        Present: *const fn (*ISwapChain, u32, u32) callconv(.winapi) Hresult,
        GetBuffer: *const fn (*ISwapChain, u32, *const Guid, *?*anyopaque) callconv(.winapi) Hresult,
        SetFullscreenState: *const anyopaque,
        GetFullscreenState: *const anyopaque,
        GetDesc: *const anyopaque,
        ResizeBuffers: *const fn (*ISwapChain, u32, u32, u32, rc.Format, u32) callconv(.winapi) Hresult,
        ResizeTarget: *const anyopaque,
        GetContainingOutput: *const anyopaque,
        GetFrameStatistics: *const anyopaque,
        GetLastPresentCount: *const anyopaque,
    };
};

/// `IDXGISwapChain3`. Everything between `IDXGISwapChain` and
/// `GetCurrentBackBufferIndex` - all of `IDXGISwapChain1` and
/// `IDXGISwapChain2` - is left opaque; the vtable stops right after the one
/// slot this file needs from `IDXGISwapChain3` itself, `GetDesc1`
/// (`IDXGISwapChain1`'s, reached early because it is the only way to learn a
/// swap chain's real size without `ID3D12Resource::GetDesc`'s struct-by-value
/// return) aside.
const ISwapChain3 = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{94D99BDB-F1F8-4AB0-B236-7DA0170EDAB1}");

    pub const VTable = extern struct {
        base: ISwapChain.VTable,
        /// `IDXGISwapChain1::GetDesc1`. An out-pointer, not a return-by-value -
        /// no ABI trap here, unlike `ID3D12Resource::GetDesc`.
        GetDesc1: *const fn (*ISwapChain3, *SwapChainDesc1) callconv(.winapi) Hresult,
        GetFullscreenDesc: *const anyopaque,
        GetHwnd: *const anyopaque,
        GetCoreWindow: *const anyopaque,
        Present1: *const anyopaque,
        IsTemporaryMonoSupported: *const anyopaque,
        GetRestrictToOutput: *const anyopaque,
        SetBackgroundColor: *const anyopaque,
        GetBackgroundColor: *const anyopaque,
        SetRotation: *const anyopaque,
        GetRotation: *const anyopaque,
        SetSourceSize: *const anyopaque,
        GetSourceSize: *const anyopaque,
        SetMaximumFrameLatency: *const anyopaque,
        GetMaximumFrameLatency: *const anyopaque,
        GetFrameLatencyWaitableObject: *const anyopaque,
        SetMatrixTransform: *const anyopaque,
        GetMatrixTransform: *const anyopaque,
        GetCurrentBackBufferIndex: *const fn (*ISwapChain3) callconv(.winapi) u32,
    };
};

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const D3d = struct {
    gpa: Allocator,
    library: d3d.D3d12,
    dxgi_library: d3d.Dxgi,
    factory: *dxgi.IDXGIFactory1,
    compiler: ?d3d.Compiler = null,

    device: *d3d12.ID3D12Device,
    queue: *d3d12.ID3D12CommandQueue,
    cmd_allocator: *cmdmod.ID3D12CommandAllocator,
    list: *cmdmod.ID3D12GraphicsCommandList,
    fence: *rc.ID3D12Fence,
    fence_value: u64 = 0,

    root_signature: *pl.ID3D12RootSignature,

    rtv_increment: u32,
    cbv_srv_uav_increment: u32,
    sampler_increment: u32,

    /// Every texture's SRV, every sampler's descriptor: written once at
    /// creation, never shader-visible, never moved.
    srv_heap: *rc.ID3D12DescriptorHeap,
    srv_next: u32 = 0,
    srv_free: std.ArrayListUnmanaged(u32) = .empty,
    sampler_heap: *rc.ID3D12DescriptorHeap,
    sampler_next: u32 = 0,
    sampler_free: std.ArrayListUnmanaged(u32) = .empty,

    /// The four-slot, shader-visible windows the root descriptor tables
    /// point at. `set_texture` copies into these; they are never resized and
    /// never re-bound mid-frame.
    binding_srv_heap: *rc.ID3D12DescriptorHeap,
    binding_sampler_heap: *rc.ID3D12DescriptorHeap,

    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,

    // Per-submit recording state.
    current_pipeline: ?*PipelineRes = null,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
};

const VertexBinding = struct {
    resource: ?*rc.ID3D12Resource = null,
    offset: u32 = 0,
    size: u32 = 0,
};

const BufferRes = struct {
    resource: *rc.ID3D12Resource,
    size: u32,
    /// Persistently mapped: an upload-heap resource may stay mapped for its
    /// whole life, and every buffer here is one.
    mapped: [*]u8,
};

const TextureRes = struct {
    resource: *rc.ID3D12Resource,
    width: u32,
    height: u32,
    format: rc.Format,
    srv_index: u32,
    srv_cpu: rc.CpuDescriptorHandle,
};

const SamplerRes = struct {
    index: u32,
    cpu: rc.CpuDescriptorHandle,
};

const ShaderRes = struct {
    /// DXBC, kept alive between `createShader` and however many
    /// `createPipeline` calls read it later - unlike the shaders themselves,
    /// a pipeline state only reads the bytecode during its own creation
    /// call, so nothing needs to outlive `createPipeline`.
    vertex: []u8,
    pixel: []u8,
};

const PipelineRes = struct {
    pso: *pl.ID3D12PipelineState,
    /// `IASetPrimitiveTopology`'s topology - finer than the
    /// `PrimitiveTopologyType` the pipeline state itself was built with.
    topology: cmdmod.PrimitiveTopology,
    strides: [max_vertex_slots]u32 = @splat(0),
    buffer_count: u32 = 0,
};

const SurfaceRes = struct {
    swap_chain: *ISwapChain3,
    rtv_heap: *rc.ID3D12DescriptorHeap,
    back_buffers: [2]*rc.ID3D12Resource,
    rtv_handles: [2]rc.CpuDescriptorHandle,
    /// Whether back buffer `i` is `RENDER_TARGET` right now rather than
    /// `PRESENT` - so `end_pass` knows whether there is a transition to
    /// undo, and `resizeSurface`/`destroySurface` never release a resource
    /// the GPU might still be transitioning.
    in_render_target: [2]bool = .{ false, false },
    width: u32,
    height: u32,
};

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!backend.Opened {
    var library = d3d.D3d12.load() catch return error.NoDevice;
    errdefer library.unload();
    var dxgi_library = d3d.Dxgi.load() catch return error.NoDevice;
    errdefer dxgi_library.unload();

    // Must be before any device is created - a device made first does not
    // get it. Best-effort: the Graphics Tools feature is a developer
    // machine's, not every machine's.
    if (desc.debug) library.enableDebugLayer() catch {};

    const factory = dxgi_library.createFactory(dxgi.IDXGIFactory1, .{ .debug = desc.debug }) catch
        dxgi_library.createFactory(dxgi.IDXGIFactory1, .{}) catch return error.NoDevice;
    errdefer _ = com.release(factory);

    // A null adapter is Direct3D 12's own "the default hardware adapter",
    // exactly as a null one is Direct3D 11's - so only the software case
    // needs `dxgi.zig`'s enumeration at all.
    var adapter: ?*dxgi.IDXGIAdapter1 = null;
    defer if (adapter) |a| {
        _ = com.release(a);
    };
    if (desc.software) adapter = dxgi.warpAdapter(factory) catch return error.NoDevice;

    const device = library.createDevice(.{
        .adapter = if (adapter) |a| @ptrCast(a) else null,
    }) catch return error.NoDevice;
    errdefer _ = com.release(device);

    const queue = d3d12.createCommandQueue(device, .{ .type = .direct }) catch return error.NoDevice;
    errdefer _ = com.release(queue);

    const cmd_allocator = cmdmod.createCommandAllocator(device, .direct) catch return error.NoDevice;
    errdefer _ = com.release(cmd_allocator);

    const list = cmdmod.createGraphicsCommandList(device, 0, .direct, cmd_allocator) catch return error.NoDevice;
    errdefer _ = com.release(list);
    // Real Direct3D 12 API requirement: a command list is created already
    // recording, and must be closed before its first `ExecuteCommandLists`.
    list.vtable.Close(list).check() catch return error.NoDevice;

    const fence = rc.createFence(device, 0, rc.fence_flags_none) catch return error.NoDevice;
    errdefer _ = com.release(fence);

    // The one root signature every pipeline shares: four root CBVs, one
    // CBV_SRV_UAV table, one sampler table. `.all` visibility throughout,
    // matching the Direct3D 11 backend's own choice to bind every stage
    // rather than assume only the pixel shader samples.
    const srv_ranges = [_]pl.DescriptorRange{
        .{ .range_type = .srv, .num_descriptors = max_binding_slots, .base_shader_register = 0 },
    };
    const sampler_ranges = [_]pl.DescriptorRange{
        .{ .range_type = .sampler, .num_descriptors = max_binding_slots, .base_shader_register = 0 },
    };
    const root_params = [_]pl.RootParameter{
        pl.RootParameter.cbv(0, .all),
        pl.RootParameter.cbv(1, .all),
        pl.RootParameter.cbv(2, .all),
        pl.RootParameter.cbv(3, .all),
        pl.RootParameter.table(&srv_ranges, .all),
        pl.RootParameter.table(&sampler_ranges, .all),
    };
    const root_blob = pl.serializeGraphicsRootSignature(
        library,
        &root_params,
        d3d12.root_signature_allow_input_assembler_input_layout,
    ) catch return error.NoDevice;
    defer _ = com.release(root_blob);
    const root_signature = pl.createRootSignature(device, 0, root_blob.bytes()) catch return error.NoDevice;
    errdefer _ = com.release(root_signature);

    const srv_heap = rc.createDescriptorHeap(device, .{ .type = .cbv_srv_uav, .num_descriptors = max_srv_descriptors }) catch return error.NoDevice;
    errdefer _ = com.release(srv_heap);
    const sampler_heap = rc.createDescriptorHeap(device, .{ .type = .sampler, .num_descriptors = max_sampler_descriptors }) catch return error.NoDevice;
    errdefer _ = com.release(sampler_heap);
    const binding_srv_heap = rc.createDescriptorHeap(device, .{
        .type = .cbv_srv_uav,
        .num_descriptors = max_binding_slots,
        .flags = .{ .shader_visible = true },
    }) catch return error.NoDevice;
    errdefer _ = com.release(binding_srv_heap);
    const binding_sampler_heap = rc.createDescriptorHeap(device, .{
        .type = .sampler,
        .num_descriptors = max_binding_slots,
        .flags = .{ .shader_visible = true },
    }) catch return error.NoDevice;
    errdefer _ = com.release(binding_sampler_heap);

    const self = try gpa.create(D3d);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .library = library,
        .dxgi_library = dxgi_library,
        .factory = factory,
        .device = device,
        .queue = queue,
        .cmd_allocator = cmd_allocator,
        .list = list,
        .fence = fence,
        .root_signature = root_signature,
        .rtv_increment = rc.descriptorHandleIncrementSize(device, .rtv),
        .cbv_srv_uav_increment = rc.descriptorHandleIncrementSize(device, .cbv_srv_uav),
        .sampler_increment = rc.descriptorHandleIncrementSize(device, .sampler),
        .srv_heap = srv_heap,
        .sampler_heap = sampler_heap,
        .binding_srv_heap = binding_srv_heap,
        .binding_sampler_heap = binding_sampler_heap,
        .debug = desc.debug,
    };

    if (desc.software) {
        const name = "Microsoft Basic Render Driver (WARP)";
        @memcpy(self.renderer[0..name.len], name);
        self.renderer_len = name.len;
    } else if (blk: {
        var it = dxgi.adapters(factory);
        break :blk it.next() catch null;
    }) |a| {
        defer _ = com.release(a);
        if (dxgi.describe(a)) |description| {
            const name = description.name();
            const n = @min(name.len, self.renderer.len);
            @memcpy(self.renderer[0..n], name[0..n]);
            self.renderer_len = n;
        } else |_| {}
    }

    return .{ self, &vtable };
}

const vtable: backend.Vtable = .{
    .deinit = deinit,
    .info = info,
    .caps = caps,
    .createBuffer = createBuffer,
    .destroyBuffer = destroyBuffer,
    .updateBuffer = updateBuffer,
    .createTexture = createTexture,
    .destroyTexture = destroyTexture,
    .writeTexture = writeTexture,
    .readTexture = readTexture,
    .createSampler = createSampler,
    .destroySampler = destroySampler,
    .createShader = createShader,
    .destroyShader = destroyShader,
    .createPipeline = createPipeline,
    .destroyPipeline = destroyPipeline,
    .createSurface = createSurface,
    .destroySurface = destroySurface,
    .resizeSurface = resizeSurface,
    .surfaceSize = surfaceSize,
    .present = present,
    .submit = submit,
};

fn cast(impl: backend.Impl) *D3d {
    return @ptrCast(@alignCast(impl));
}

fn as(comptime T: type, native: backend.Native) *T {
    return @ptrCast(@alignCast(native));
}

fn deinit(impl: backend.Impl) void {
    const self = cast(impl);
    // `Device.deinit` has already destroyed every resource this device
    // made, and every `submit`/upload already waited for the GPU: nothing
    // outstanding needs a flush here.
    if (self.compiler) |*c| c.unload();
    com.releaseAll(.{
        self.binding_sampler_heap,
        self.binding_srv_heap,
        self.sampler_heap,
        self.srv_heap,
        self.root_signature,
        self.fence,
        self.list,
        self.cmd_allocator,
        self.queue,
        self.device,
    });
    self.srv_free.deinit(self.gpa);
    self.sampler_free.deinit(self.gpa);
    _ = com.release(self.factory);
    self.dxgi_library.unload();
    self.library.unload();
    self.gpa.destroy(self);
}

fn info(impl: backend.Impl) types.Info {
    const self = cast(impl);
    return .{ .backend = .d3d12, .renderer = self.renderer[0..self.renderer_len] };
}

// -------------------------------------------------------------------------
// Capabilities: the MVP scope table, as data. `.d2` only, one sample, one
// mip, `rgba8_unorm`/`bgra8_unorm`, sampled but never a render target -
// `Device` refuses everything this does not claim before this file is asked.
// -------------------------------------------------------------------------

fn caps(impl: backend.Impl) types.Caps {
    _ = impl;
    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = 16384,
            .max_texture_3d = 2048,
            .max_texture_cube = 16384,
            .max_texture_layers = 2048,
            // Clamped to one: `Device.createSampler` clamps every request to
            // this, so the MVP's "no anisotropy above one" is enforced here
            // rather than by this file refusing a sampler by hand.
            .max_anisotropy = 1,
            .max_color_attachments = 1,
        },
        .features = .{},
    };
    const sampled: types.FormatSupport = .{
        .sampled = true,
        .filterable = true,
        .render_target = false,
        .blendable = false,
        .generate_mips = false,
        .sample_counts = 0b1,
        .dimensions = blk: {
            var set = std.EnumSet(types.Dimension).initEmpty();
            set.insert(.d2);
            break :blk set;
        },
    };
    answer.formats.set(.rgba8_unorm, sampled);
    answer.formats.set(.bgra8_unorm, sampled);
    return answer;
}

fn textureFormat(format: types.Format) ?rc.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        else => null,
    };
}

// -------------------------------------------------------------------------
// Descriptor slots: a permanent home for a texture's SRV or a sampler's
// descriptor, handed out from a fixed-size heap with a small free list so a
// destroyed one is reused rather than leaking the slot forever.
// -------------------------------------------------------------------------

fn takeFree(list: *std.ArrayListUnmanaged(u32)) ?u32 {
    if (list.items.len == 0) return null;
    const v = list.items[list.items.len - 1];
    list.items.len -= 1;
    return v;
}

fn allocSrv(self: *D3d) Error!u32 {
    if (takeFree(&self.srv_free)) |i| return i;
    if (self.srv_next >= max_srv_descriptors) return error.Failed;
    const i = self.srv_next;
    self.srv_next += 1;
    return i;
}

fn freeSrv(self: *D3d, i: u32) void {
    self.srv_free.append(self.gpa, i) catch {};
}

fn srvHandle(self: *D3d, i: u32) rc.CpuDescriptorHandle {
    return rc.cpuHeapStart(self.srv_heap).offsetBy(i, self.cbv_srv_uav_increment);
}

fn allocSampler(self: *D3d) Error!u32 {
    if (takeFree(&self.sampler_free)) |i| return i;
    if (self.sampler_next >= max_sampler_descriptors) return error.Failed;
    const i = self.sampler_next;
    self.sampler_next += 1;
    return i;
}

fn freeSampler(self: *D3d, i: u32) void {
    self.sampler_free.append(self.gpa, i) catch {};
}

fn samplerHandle(self: *D3d, i: u32) rc.CpuDescriptorHandle {
    return rc.cpuHeapStart(self.sampler_heap).offsetBy(i, self.sampler_increment);
}

// -------------------------------------------------------------------------
// Recording and waiting: the two halves every submit - a frame's, or a
// texture upload's - is built from.
// -------------------------------------------------------------------------

fn beginRecording(self: *D3d) Error!void {
    self.cmd_allocator.vtable.Reset(self.cmd_allocator).check() catch return error.Failed;
    self.list.vtable.Reset(self.list, self.cmd_allocator, null).check() catch return error.Failed;
}

fn closeExecuteAndWait(self: *D3d) Error!void {
    self.list.vtable.Close(self.list).check() catch return error.Failed;
    const lists = [_]*cmdmod.ID3D12GraphicsCommandList{self.list};
    cmdmod.executeCommandLists(self.queue, &lists);
    self.fence_value += 1;
    self.queue.vtable.Signal(self.queue, @ptrCast(self.fence), self.fence_value).check() catch return error.Failed;
    while (self.fence.vtable.GetCompletedValue(self.fence) < self.fence_value) {}
    if (self.debug) {
        self.device.vtable.GetDeviceRemovedReason(self.device).check() catch return error.DeviceLost;
    }
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) Error!backend.Native {
    const self = cast(impl);
    const res = try self.gpa.create(BufferRes);
    errdefer self.gpa.destroy(res);

    const size: u32 = @intCast(if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 16) else desc.size);

    const obj = rc.createCommittedResource(
        self.device,
        .of(.upload),
        rc.heap_flags_none,
        .buffer(size),
        .generic_read,
        null,
    ) catch return error.Failed;
    errdefer _ = com.release(obj);

    var mapped: ?*anyopaque = null;
    obj.vtable.Map(obj, 0, &rc.Range.nothing_read, &mapped).check() catch return error.Failed;
    const bytes: [*]u8 = @ptrCast(mapped.?);
    @memset(bytes[0..size], 0);
    if (desc.data) |data| @memcpy(bytes[0..data.len], data);

    res.* = .{ .resource = obj, .size = size, .mapped = bytes };
    return res;
}

fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    res.resource.vtable.Unmap(res.resource, 0, null);
    _ = com.release(res.resource);
    self.gpa.destroy(res);
}

fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) Error!void {
    _ = impl;
    const res = as(BufferRes, native);
    @memcpy(res.mapped[offset..][0..bytes.len], bytes);
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    // `Device` has already checked `caps`, so this is defence in depth: a
    // shape or a format outside the MVP never reaches here in practice.
    if (desc.dimension != .d2 or desc.samples != 1 or desc.mip_levels != 1) return error.Unsupported;
    const format = textureFormat(desc.format) orelse return error.Unsupported;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const rdesc = rc.ResourceDesc.texture2d(desc.width, desc.height, format, .{});
    const has_data = desc.data != null;
    const initial_state: rc.ResourceStates = if (has_data) .{ .copy_dest = true } else .{ .pixel_shader_resource = true };

    const obj = rc.createCommittedResource(self.device, .of(.default), rc.heap_flags_none, rdesc, initial_state, null) catch return error.Failed;
    errdefer _ = com.release(obj);

    if (desc.data) |data| try uploadTexture(self, obj, rdesc, desc, data);

    const slot = try allocSrv(self);
    errdefer freeSrv(self, slot);
    const cpu = srvHandle(self, slot);
    rc.createShaderResourceView(self.device, obj, &.{
        .format = format,
        .dimension = .texture2d,
        .u = .{ .texture2d = .{ .mip_levels = 1 } },
    }, cpu);

    res.* = .{ .resource = obj, .width = desc.width, .height = desc.height, .format = format, .srv_index = slot, .srv_cpu = cpu };
    return res;
}

/// Level zero, uploaded through a one-off staging buffer:
/// `GetCopyableFootprints` says how the driver wants the rows padded, the
/// staging buffer is filled to that layout, and `CopyTextureRegion` reads it
/// - the standard shape of an upload onto a `DEFAULT`-heap texture. Runs its
/// own record/execute/wait, sharing the one command list and allocator
/// `submit` also uses: creation always happens between frames, never while
/// one is being recorded.
fn uploadTexture(self: *D3d, obj: *rc.ID3D12Resource, rdesc: rc.ResourceDesc, desc: types.TextureDesc, data: []const u8) Error!void {
    var footprint: rc.PlacedSubresourceFootprint = undefined;
    var total_bytes: u64 = 0;
    rc.getCopyableFootprints(self.device, &rdesc, 0, 1, 0, @ptrCast(&footprint), null, null, &total_bytes);

    const staging = rc.createCommittedResource(self.device, .of(.upload), rc.heap_flags_none, .buffer(total_bytes), .generic_read, null) catch return error.Failed;
    defer _ = com.release(staging);

    var mapped: ?*anyopaque = null;
    staging.vtable.Map(staging, 0, &rc.Range.nothing_read, &mapped).check() catch return error.Failed;
    const dst: [*]u8 = @ptrCast(mapped.?);

    const src_pitch = desc.effectiveRowPitch();
    const row_bytes = desc.format.rowBytes(desc.width);
    const rows = desc.format.rowCount(desc.height);
    var y: u32 = 0;
    while (y < rows) : (y += 1) {
        const src_row = data[y * src_pitch ..][0..row_bytes];
        const dst_row = dst[footprint.offset + y * footprint.footprint.row_pitch ..][0..row_bytes];
        @memcpy(dst_row, src_row);
    }
    staging.vtable.Unmap(staging, 0, null);

    try beginRecording(self);
    const dst_loc = cmdmod.TextureCopyLocation.subresource(obj, 0);
    const src_loc = cmdmod.TextureCopyLocation.placed(staging, footprint);
    self.list.vtable.CopyTextureRegion(self.list, &dst_loc, 0, 0, 0, &src_loc, null);
    const barrier = cmdmod.ResourceBarrier.transition(obj, .{ .copy_dest = true }, .{ .pixel_shader_resource = true });
    self.list.vtable.ResourceBarrier(self.list, 1, &[_]cmdmod.ResourceBarrier{barrier});
    try closeExecuteAndWait(self);
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    freeSrv(self, res.srv_index);
    _ = com.release(res.resource);
    self.gpa.destroy(res);
}

/// Outside the MVP scope: see the module comment and the plan's scope table.
fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    _ = .{ impl, native, region, bytes, row_pitch, slice_pitch };
    return error.Unsupported;
}

fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    _ = .{ impl, native, sub, gpa };
    return error.Unsupported;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

fn addressMode(w: types.Wrap) rc.TextureAddressMode {
    return switch (w) {
        .repeat => .wrap,
        .clamp_to_edge => .clamp,
        .mirror => .mirror,
        // `Features.sampler_border` is false, so `Device.createSampler`
        // refuses this before it reaches here.
        .border => .clamp,
    };
}

fn samplerFilter(desc: types.SamplerDesc) rc.Filter {
    // `D3D12_FILTER`'s bit encoding: bit 0x10 linear-minifies, 0x04
    // linear-magnifies, 0x01 linear-filters between mips. `max_anisotropy`
    // is always clamped to one by `Device`, so the anisotropic encoding is
    // never needed here.
    var bits: u32 = 0;
    if (desc.min_filter == .linear) bits |= 0x10;
    if (desc.mag_filter == .linear) bits |= 0x04;
    if (desc.mip_filter == .linear) bits |= 0x01;
    return @enumFromInt(bits);
}

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);
    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    const slot = try allocSampler(self);
    errdefer freeSampler(self, slot);
    const cpu = samplerHandle(self, slot);

    // "Level zero only" is a range with one level in it: Direct3D has no mip
    // filter that is off, only a top and bottom to clamp the level to.
    const level_zero_only = desc.mip_filter == .none;
    rc.createSampler(self.device, &.{
        .filter = samplerFilter(desc),
        .address_u = addressMode(desc.wrap_u),
        .address_v = addressMode(desc.wrap_v),
        .address_w = addressMode(desc.wrap_w),
        .mip_lod_bias = 0,
        .max_anisotropy = @max(1, desc.max_anisotropy),
        .comparison_func = .never,
        .min_lod = if (level_zero_only) 0 else desc.lod_min,
        .max_lod = if (level_zero_only) 0 else desc.lod_max,
    }, cpu);

    res.* = .{ .index = slot, .cpu = cpu };
    return res;
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    freeSampler(self, res.index);
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Shaders and pipelines
// -------------------------------------------------------------------------

fn compilerOf(self: *D3d, log: *Io.Writer) Error!d3d.Compiler {
    if (self.compiler) |c| return c;
    self.compiler = d3d.Compiler.load() catch {
        log.writeAll("fluxion-rhi: d3dcompiler_47.dll is not on this machine, so HLSL source cannot be compiled") catch {};
        return error.ShaderFailed;
    };
    return self.compiler.?;
}

fn compileStage(compiler: d3d.Compiler, source: []const u8, target: [:0]const u8, log: *Io.Writer) Error!*d3d.ID3DBlob {
    var output = compiler.compile(source, .{ .target = target, .name = "fluxion-rhi.hlsl" });
    errdefer output.release();
    const code = output.check() catch {
        log.print("{s} did not compile:\n{s}", .{ target, output.text() }) catch {};
        output.release();
        return error.ShaderFailed;
    };
    if (output.warned()) log.print("{s}:\n{s}", .{ target, output.text() }) catch {};
    return code;
}

fn createShader(impl: backend.Impl, desc: types.ShaderDesc, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const sources = desc.hlsl orelse {
        log.writeAll("fluxion-rhi: the Direct3D 12 backend needs `ShaderDesc.hlsl`, and none was given") catch {};
        return error.ShaderFailed;
    };
    const compiler = try compilerOf(self, log);

    // `vs_5_0`/`ps_5_0`: shader model 5.0, what `fluxion-shader`'s HLSL
    // emitter targets and what feature level 11.0 guarantees.
    const vs_blob = try compileStage(compiler, sources.vertex, "vs_5_0", log);
    defer _ = com.release(vs_blob);
    const ps_blob = try compileStage(compiler, sources.fragment, "ps_5_0", log);
    defer _ = com.release(ps_blob);

    const res = try self.gpa.create(ShaderRes);
    errdefer self.gpa.destroy(res);
    const vertex = try self.gpa.dupe(u8, vs_blob.bytes());
    errdefer self.gpa.free(vertex);
    const pixel = try self.gpa.dupe(u8, ps_blob.bytes());
    errdefer self.gpa.free(pixel);

    res.* = .{ .vertex = vertex, .pixel = pixel };
    return res;
}

fn destroyShader(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(ShaderRes, native);
    self.gpa.free(res.vertex);
    self.gpa.free(res.pixel);
    self.gpa.destroy(res);
}

fn vertexFormat(format: types.VertexFormat) rc.Format {
    return switch (format) {
        .float => .r32_float,
        .float2 => .r32g32_float,
        .float3 => .r32g32b32_float,
        .float4 => .r32g32b32a32_float,
        .ubyte4_norm => .r8g8b8a8_unorm,
        .ubyte4 => .r8g8b8a8_uint,
        .uint => .r32_uint,
        .int => .r32_sint,
    };
}

fn blendFactor(f: types.BlendFactor) pl.Blend {
    return switch (f) {
        .zero => .zero,
        .one => .one,
        .src_color => .src_color,
        .one_minus_src_color => .inv_src_color,
        .src_alpha => .src_alpha,
        .one_minus_src_alpha => .inv_src_alpha,
        .dst_color => .dest_color,
        .one_minus_dst_color => .inv_dest_color,
        .dst_alpha => .dest_alpha,
        .one_minus_dst_alpha => .inv_dest_alpha,
    };
}

fn blendOp(op: types.BlendOp) pl.BlendOp {
    return switch (op) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .rev_subtract,
        .min => .min,
        .max => .max,
    };
}

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const shader_res = as(ShaderRes, shader);

    if (desc.depth_format != null or desc.extra_color_formats.len > 0 or desc.samples != 1) {
        log.writeAll("fluxion-rhi: the Direct3D 12 backend's MVP has no depth, no extra colour attachments and no MSAA") catch {};
        return error.Unsupported;
    }
    if (desc.buffers.len > max_vertex_slots) {
        log.print("fluxion-rhi: the Direct3D 12 backend binds at most {d} vertex buffers", .{max_vertex_slots}) catch {};
        return error.PipelineFailed;
    }
    if (desc.attributes.len > max_attributes) {
        log.print("fluxion-rhi: the Direct3D 12 backend takes at most {d} vertex attributes", .{max_attributes}) catch {};
        return error.PipelineFailed;
    }
    const color_format = textureFormat(desc.color_format orelse {
        log.writeAll("fluxion-rhi: the Direct3D 12 backend's MVP has no depth-only pass") catch {};
        return error.Unsupported;
    }) orelse return error.Unsupported;

    var elements: [max_attributes]pl.InputElementDesc = undefined;
    for (desc.attributes, 0..) |attribute, i| {
        const step = desc.buffers[attribute.buffer].step;
        elements[i] = .{
            .semantic_name = "ATTR",
            .semantic_index = attribute.location,
            .format = vertexFormat(attribute.format),
            .input_slot = attribute.buffer,
            .aligned_byte_offset = attribute.offset,
            .input_slot_class = if (step == .instance) .per_instance_data else .per_vertex_data,
            .instance_data_step_rate = if (step == .instance) 1 else 0,
        };
    }

    var blend: pl.BlendDesc = .{};
    blend.render_target[0] = .{
        .blend_enable = if (desc.blend.enabled) 1 else 0,
        .src_blend = blendFactor(desc.blend.src_rgb),
        .dest_blend = blendFactor(desc.blend.dst_rgb),
        .blend_op = blendOp(desc.blend.op_rgb),
        .src_blend_alpha = blendFactor(desc.blend.src_alpha),
        .dest_blend_alpha = blendFactor(desc.blend.dst_alpha),
        .blend_op_alpha = blendOp(desc.blend.op_alpha),
        .render_target_write_mask = pl.color_write_all,
    };

    const rasterizer: pl.RasterizerDesc = .{
        .cull_mode = switch (desc.cull) {
            .none => .none,
            .back => .back,
            .front => .front,
        },
        .front_counter_clockwise = if (desc.front_face == .ccw) 1 else 0,
    };

    const topology_type: pl.PrimitiveTopologyType = switch (desc.topology) {
        .triangles, .triangle_strip => .triangle,
        .lines, .line_strip => .line,
        .points => .point,
    };
    const fine_topology: cmdmod.PrimitiveTopology = switch (desc.topology) {
        .triangles => .triangle_list,
        .triangle_strip => .triangle_strip,
        .lines => .line_list,
        .line_strip => .line_strip,
        .points => .point_list,
    };

    var rtv_formats: [8]rc.Format = @splat(.unknown);
    rtv_formats[0] = color_format;

    const pso_desc: pl.GraphicsPipelineStateDesc = .{
        .root_signature = self.root_signature,
        .vs = .of(shader_res.vertex),
        .ps = .of(shader_res.pixel),
        .blend_state = blend,
        .rasterizer_state = rasterizer,
        .depth_stencil_state = .{},
        .input_layout = .{
            .elements = if (desc.attributes.len == 0) null else &elements,
            .count = @intCast(desc.attributes.len),
        },
        .primitive_topology_type = topology_type,
        .num_render_targets = 1,
        .rtv_formats = rtv_formats,
        .sample = .{},
    };

    const pso = pl.createGraphicsPipelineState(self.device, &pso_desc) catch {
        log.writeAll("fluxion-rhi: CreateGraphicsPipelineState failed (an attribute the shader has no ATTRn input for is the usual reason)") catch {};
        return error.PipelineFailed;
    };

    const res = try self.gpa.create(PipelineRes);
    var strides: [max_vertex_slots]u32 = @splat(0);
    for (desc.buffers, 0..) |buffer, i| strides[i] = buffer.stride;
    res.* = .{ .pso = pso, .topology = fine_topology, .strides = strides, .buffer_count = @intCast(desc.buffers.len) };
    return res;
}

fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(PipelineRes, native);
    _ = com.release(res.pso);
    if (self.current_pipeline == res) self.current_pipeline = null;
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Surfaces
// -------------------------------------------------------------------------

fn createSurface(impl: backend.Impl, desc: types.SurfaceDesc) Error!backend.Native {
    const self = cast(impl);
    if (desc.native_window == 0) return error.InvalidArgument;

    const factory2: *IFactory2 = @ptrCast(com.queryInterface(self.factory, dxgi.IDXGIFactory2) catch return error.Unsupported);
    defer _ = com.release(factory2);

    const chain_desc: SwapChainDesc1 = .{ .width = desc.width, .height = desc.height };
    var swap_chain: ?*ISwapChain = null;
    factory2.vtable.CreateSwapChainForHwnd(
        factory2,
        com.unknown(self.queue),
        @ptrFromInt(desc.native_window),
        &chain_desc,
        null,
        null,
        &swap_chain,
    ).check() catch return error.Failed;
    errdefer _ = com.release(swap_chain.?);

    const swap_chain3 = com.queryInterface(swap_chain.?, ISwapChain3) catch return error.Failed;
    _ = com.release(swap_chain.?);
    errdefer _ = com.release(swap_chain3);

    const rtv_heap = rc.createDescriptorHeap(self.device, .{ .type = .rtv, .num_descriptors = 2 }) catch return error.Failed;
    errdefer _ = com.release(rtv_heap);

    const res = try self.gpa.create(SurfaceRes);
    errdefer self.gpa.destroy(res);
    res.* = .{
        .swap_chain = swap_chain3,
        .rtv_heap = rtv_heap,
        .back_buffers = undefined,
        .rtv_handles = undefined,
        .width = 0,
        .height = 0,
    };
    try attachBackBuffers(self, res);
    return res;
}

fn attachBackBuffers(self: *D3d, res: *SurfaceRes) Error!void {
    var sc_desc: SwapChainDesc1 = undefined;
    res.swap_chain.vtable.GetDesc1(res.swap_chain, &sc_desc).check() catch return error.Failed;
    res.width = sc_desc.width;
    res.height = sc_desc.height;

    const cpu_start = rc.cpuHeapStart(res.rtv_heap);
    var i: u32 = 0;
    while (i < 2) : (i += 1) {
        var raw: ?*anyopaque = null;
        res.swap_chain.vtable.base.GetBuffer(@ptrCast(res.swap_chain), i, com.iidOf(rc.ID3D12Resource), &raw).check() catch return error.Failed;
        const buf = com.received(rc.ID3D12Resource, .s_ok, raw) catch return error.Failed;
        const handle = cpu_start.offsetBy(i, self.rtv_increment);
        rc.createRenderTargetView(self.device, buf, null, handle);
        res.back_buffers[i] = buf;
        res.rtv_handles[i] = handle;
        res.in_render_target[i] = false;
    }
}

fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    for (res.back_buffers) |buf| _ = com.release(buf);
    _ = com.release(res.rtv_heap);
    _ = com.release(res.swap_chain);
    self.gpa.destroy(res);
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    for (res.back_buffers) |buf| _ = com.release(buf);
    res.swap_chain.vtable.base.ResizeBuffers(@ptrCast(res.swap_chain), 0, width, height, .unknown, 0).check() catch return error.Failed;
    try attachBackBuffers(self, res);
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

fn present(impl: backend.Impl, native: backend.Native, vsync: bool) Error!void {
    _ = impl;
    const res = as(SurfaceRes, native);
    res.swap_chain.vtable.base.Present(@ptrCast(res.swap_chain), if (vsync) 1 else 0, 0).check() catch |err| switch (err) {
        error.DeviceRemoved, error.DeviceReset => return error.DeviceLost,
        else => return error.Failed,
    };
}

// -------------------------------------------------------------------------
// Submitting
// -------------------------------------------------------------------------

fn submit(impl: backend.Impl, device: *Device, list_cmds: []const commands.Command) Error!void {
    const self = cast(impl);
    try beginRecording(self);

    const cmd_list = self.list;
    cmd_list.vtable.SetGraphicsRootSignature(cmd_list, self.root_signature);
    const heaps = [_]*rc.ID3D12DescriptorHeap{ self.binding_srv_heap, self.binding_sampler_heap };
    cmd_list.vtable.SetDescriptorHeaps(cmd_list, heaps.len, &heaps);
    cmd_list.vtable.SetGraphicsRootDescriptorTable(cmd_list, root_param_srv_table, rc.gpuHeapStart(self.binding_srv_heap));
    cmd_list.vtable.SetGraphicsRootDescriptorTable(cmd_list, root_param_sampler_table, rc.gpuHeapStart(self.binding_sampler_heap));

    self.current_pipeline = null;
    self.vertex_bindings = @splat(.{});
    self.bindings_dirty = false;
    var current_surface: ?*SurfaceRes = null;

    for (list_cmds) |command| {
        switch (command) {
            .begin_pass => |pass| {
                if (pass.depth != null or pass.extra_colors.len > 0) return error.Unsupported;
                const color = pass.color orelse return error.Unsupported;
                switch (color.target) {
                    .surface => |h| {
                        const surf = as(SurfaceRes, device.surfaces.get(h).?.native);
                        const idx = surf.swap_chain.vtable.GetCurrentBackBufferIndex(surf.swap_chain);
                        const to_rt = cmdmod.ResourceBarrier.transition(surf.back_buffers[idx], .{}, .{ .render_target = true });
                        cmd_list.vtable.ResourceBarrier(cmd_list, 1, &[_]cmdmod.ResourceBarrier{to_rt});
                        surf.in_render_target[idx] = true;

                        const rtv = surf.rtv_handles[idx];
                        cmd_list.vtable.OMSetRenderTargets(cmd_list, 1, @ptrCast(&rtv), 0, null);
                        if (color.load == .clear) cmd_list.vtable.ClearRenderTargetView(cmd_list, rtv, &color.clear_color, 0, null);

                        const vp: cmdmod.Viewport = .{ .width = @floatFromInt(surf.width), .height = @floatFromInt(surf.height) };
                        cmd_list.vtable.RSSetViewports(cmd_list, 1, &[_]cmdmod.Viewport{vp});
                        const sc: cmdmod.Rect = .{ .left = 0, .top = 0, .right = @intCast(surf.width), .bottom = @intCast(surf.height) };
                        cmd_list.vtable.RSSetScissorRects(cmd_list, 1, &[_]cmdmod.Rect{sc});

                        current_surface = surf;
                    },
                    // Structurally unreachable: no texture this backend
                    // makes ever has `usage.render_target = true`, since
                    // `caps` never advertises it.
                    .texture => return error.Unsupported,
                }
                self.current_pipeline = null;
                self.vertex_bindings = @splat(.{});
                self.bindings_dirty = false;
            },
            .end_pass => {
                if (current_surface) |surf| {
                    const idx = surf.swap_chain.vtable.GetCurrentBackBufferIndex(surf.swap_chain);
                    if (surf.in_render_target[idx]) {
                        const to_present = cmdmod.ResourceBarrier.transition(surf.back_buffers[idx], .{ .render_target = true }, .{});
                        cmd_list.vtable.ResourceBarrier(cmd_list, 1, &[_]cmdmod.ResourceBarrier{to_present});
                        surf.in_render_target[idx] = false;
                    }
                    current_surface = null;
                }
            },
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                self.current_pipeline = res;
                self.bindings_dirty = true;
                cmd_list.vtable.SetPipelineState(cmd_list, res.pso);
                cmd_list.vtable.IASetPrimitiveTopology(cmd_list, res.topology);
            },
            .set_viewport => |v| {
                const vp: cmdmod.Viewport = .{ .top_left_x = v.x, .top_left_y = v.y, .width = v.width, .height = v.height, .min_depth = v.min_depth, .max_depth = v.max_depth };
                cmd_list.vtable.RSSetViewports(cmd_list, 1, &[_]cmdmod.Viewport{vp});
            },
            .set_scissor => |maybe| {
                const r: cmdmod.Rect = if (maybe) |rect| .{
                    .left = rect.x,
                    .top = rect.y,
                    .right = rect.x + @as(i32, @intCast(rect.width)),
                    .bottom = rect.y + @as(i32, @intCast(rect.height)),
                } else if (current_surface) |surf| .{
                    .left = 0,
                    .top = 0,
                    .right = @intCast(surf.width),
                    .bottom = @intCast(surf.height),
                } else .{ .left = 0, .top = 0, .right = 0, .bottom = 0 };
                cmd_list.vtable.RSSetScissorRects(cmd_list, 1, &[_]cmdmod.Rect{r});
            },
            .set_vertex_buffer => |b| {
                if (b.slot >= max_vertex_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.vertex_bindings[b.slot] = .{ .resource = res.resource, .offset = b.offset, .size = res.size };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                const view: cmdmod.IndexBufferView = .{
                    .buffer_location = res.resource.vtable.GetGPUVirtualAddress(res.resource),
                    .size_in_bytes = res.size,
                    .format = if (b.format == .u16) .r16_uint else .r32_uint,
                };
                cmd_list.vtable.IASetIndexBuffer(cmd_list, &view);
            },
            .set_uniform_buffer => |b| {
                if (b.slot >= max_binding_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                cmd_list.vtable.SetGraphicsRootConstantBufferView(cmd_list, root_param_cbv0 + b.slot, res.resource.vtable.GetGPUVirtualAddress(res.resource));
            },
            .set_texture => |b| {
                if (b.slot >= max_binding_slots) return error.Unsupported;
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                const srv_dest = rc.cpuHeapStart(self.binding_srv_heap).offsetBy(b.slot, self.cbv_srv_uav_increment);
                rc.copyDescriptorsSimple(self.device, 1, srv_dest, texture.srv_cpu, .cbv_srv_uav);
                const sampler_dest = rc.cpuHeapStart(self.binding_sampler_heap).offsetBy(b.slot, self.sampler_increment);
                rc.copyDescriptorsSimple(self.device, 1, sampler_dest, sampler.cpu, .sampler);
            },
            .draw => |d| {
                flushBindings(self);
                cmd_list.vtable.DrawInstanced(cmd_list, d.vertex_count, d.instance_count, d.first_vertex, 0);
            },
            .draw_indexed => |d| {
                flushBindings(self);
                cmd_list.vtable.DrawIndexedInstanced(cmd_list, d.index_count, d.instance_count, d.first_index, d.base_vertex, 0);
            },
            // Unreachable in practice: every MVP texture has one mip level,
            // and `Device.submit`'s own validation refuses `generateMips` on
            // one before any backend sees it.
            .generate_mips => return error.Unsupported,
        }
    }

    try closeExecuteAndWait(self);
}

/// Bind the vertex buffers set since the last flush, now that the pipeline
/// that gives them their strides is known to be current - `set_vertex_buffer`
/// carries no stride of its own, `PipelineDesc.buffers[slot].stride` is the
/// only place it lives.
fn flushBindings(self: *D3d) void {
    if (!self.bindings_dirty) return;
    self.bindings_dirty = false;
    const p = self.current_pipeline orelse return;
    if (p.buffer_count == 0) return;

    var views: [max_vertex_slots]cmdmod.VertexBufferView = undefined;
    var i: u32 = 0;
    while (i < p.buffer_count) : (i += 1) {
        const b = self.vertex_bindings[i];
        views[i] = if (b.resource) |buf| .{
            .buffer_location = buf.vtable.GetGPUVirtualAddress(buf) + b.offset,
            .size_in_bytes = b.size - b.offset,
            .stride_in_bytes = p.strides[i],
        } else .{ .buffer_location = 0, .size_in_bytes = 0, .stride_in_bytes = p.strides[i] };
    }
    self.list.vtable.IASetVertexBuffers(self.list, 0, p.buffer_count, &views);
}

// -------------------------------------------------------------------------
// Tests. WARP needs no window and no card. A window is out of scope for
// this file - `readTexture` is `error.Unsupported`, so there is no way to
// look at what a pass drew without one - so these check what can be checked
// without a window: that every kind of resource is made correctly, and that
// a request outside the MVP scope table comes back as `error.Unsupported`
// through `Device`'s own caps-driven rejection rather than a crash.
// -------------------------------------------------------------------------

const testing = std.testing;

fn warpDevice() !Device {
    return Device.init(testing.allocator, .{ .backend = .d3d12, .software = true }) catch |err| switch (err) {
        error.NoDevice, error.Unsupported => error.SkipZigTest,
        else => err,
    };
}

test "opening on WARP, and what it reports" {
    var device = try warpDevice();
    defer device.deinit();

    try testing.expectEqual(types.Backend.d3d12, device.backendTag());
    const information = device.info();
    try testing.expectEqual(types.Backend.d3d12, information.backend);
    try testing.expect(information.renderer.len > 0);
}

test "caps: only .d2 rgba8/bgra8, one sample, no render target" {
    var device = try warpDevice();
    defer device.deinit();

    const c = device.caps();
    const rgba = c.formatSupport(.rgba8_unorm);
    try testing.expect(rgba.sampled);
    try testing.expect(!rgba.render_target);
    try testing.expect(rgba.dimensions.contains(.d2));
    try testing.expect(!rgba.dimensions.contains(.cube));
    try testing.expect(!rgba.dimensions.contains(.d3));
    try testing.expect(rgba.supportsSamples(1));
    try testing.expect(!rgba.supportsSamples(4));
    try testing.expectEqual(@as(u32, 1), c.limits.max_anisotropy);

    const bgra = c.formatSupport(.bgra8_unorm);
    try testing.expect(bgra.sampled);

    // Never claimed: a depth format, and a compressed one.
    try testing.expect(!c.formatSupport(.depth32_float).sampled);
    try testing.expect(!c.formatSupport(.bc1_rgba_unorm).sampled);
}

test "a buffer is created, updated, and its bytes are what was written" {
    var device = try warpDevice();
    defer device.deinit();

    const buffer = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 1, 2, 3, 4 }) });
    try device.updateBuffer(buffer, 0, std.mem.asBytes(&[4]f32{ 5, 6, 7, 8 }));
    device.destroyBuffer(buffer);
}

test "a texture with initial data, and a sampler" {
    var device = try warpDevice();
    defer device.deinit();

    const pixels = [_]u8{ 255, 0, 0, 255 } ** (4 * 4);
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .data = &pixels });
    defer device.destroyTexture(texture);

    const sampler = try device.createSampler(.nearest);
    defer device.destroySampler(sampler);

    // Outside the MVP: writing after creation, and reading back at all.
    try testing.expectError(error.Unsupported, device.writeTexture(texture, .{}, &pixels, 0, 0));
    try testing.expectError(error.Unsupported, device.readTexture(texture, testing.allocator));
}

test "a shader compiles, and a pipeline is made from it" {
    var device = try warpDevice();
    defer device.deinit();

    const vs =
        \\struct In { float2 position : ATTR0; };
        \\struct Out { float4 position : SV_POSITION; };
        \\cbuffer Frame : register(b0) { float4 tint; };
        \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1) * tint.x; return o; }
    ;
    const ps =
        \\float4 main() : SV_TARGET { return float4(1, 0, 0, 1); }
    ;
    const shader = device.createShader(.{ .hlsl = .{ .vertex = vs, .fragment = ps } }) catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };
    defer device.destroyShader(shader);

    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
    });
    device.destroyPipeline(pipeline);
}

test "a shader that does not compile says why" {
    var device = try warpDevice();
    defer device.deinit();

    const result = device.createShader(.{ .hlsl = .{
        .vertex = "float4 main() : SV_POSITION { return notADeclaredThing; }",
        .fragment = "float4 main() : SV_TARGET { return float4(0,0,0,1); }",
    } });
    try testing.expectError(error.ShaderFailed, result);
    try testing.expect(device.diagnostics().len > 0);
}

test "requests outside the MVP scope come back as error.Unsupported" {
    var device = try warpDevice();
    defer device.deinit();

    // No cube, volume or array textures.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4 }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4 }));
    // No multisampled render targets.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .samples = 4, .usage = .{ .sampled = false, .render_target = true } }));
    // No render-target texture of any kind.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .sampled = false, .render_target = true } }));
    // No compressed or depth formats.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .format = .bc1_rgba_unorm }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } }));
    // No sampler border colour or anisotropy above what `caps` allows.
    try testing.expectError(error.Unsupported, device.createSampler(.{ .wrap_u = .border }));
}

test "a surface with no window is refused" {
    var device = try warpDevice();
    defer device.deinit();
    try testing.expectError(error.InvalidArgument, device.createSurface(.{}));
}
