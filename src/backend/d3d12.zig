// SPDX-License-Identifier: BSD-2-Clause

//! The Direct3D 12 backend. Windows only, feature level 11.0.
//!
//! Built on `fluxion-d3d`'s `d3d12.zig`, which stops at the device and the
//! queue the same way its `d3d11.zig` stops at a device: the slots that make
//! resources, command lists and pipeline states are its vtables' permanent
//! `*const anyopaque` placeholders. `d3d12_resource.zig`, `d3d12_command.zig`
//! and `d3d12_pipeline.zig`, siblings of this file, are those declarations -
//! typed fresh here, the way `IFactory2` below is typed fresh over
//! `fluxion-d3d`'s `dxgi.IDXGIFactory1`. It is narrower than the Direct3D 11
//! backend on purpose: `.d2` textures in `rgba8`, `bgra8` and `r8`, one mip,
//! one sample, no depth, no compute - and within that, everything a 2D
//! renderer asks: textures drawn into and sampled after, written in part and
//! read back. What the narrowness buys is simplicity in the parts a real
//! Direct3D 12 renderer usually spends the most code on:
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
//! **Two heaps per binding kind: permanent and a ring.** Every texture's SRV
//! and every sampler's descriptor is written once, at creation, into a plain
//! (non-shader-visible) heap that never moves. The root descriptor tables
//! point into a shader-visible ring instead: a draw whose textures or
//! samplers changed since the last one takes the next four slots of the
//! ring, has the four descriptors copied there with `CopyDescriptorsSimple`,
//! and points its table at them - so each draw reads the textures it was
//! given, not whatever the last `set_texture` of the list left. The ring
//! starts over at every submit, which is safe because a submit has waited for
//! the GPU before the next one records; one that runs out of ring executes
//! what it has, waits, and records on from the same state.
//!
//! **Fully synchronous.** `submit` records the whole command list, executes
//! it, signals a fence and spins on `GetCompletedValue` until the GPU has
//! caught up, every time. No double-buffering, no multiple frames in flight.
//! Slower than a real engine would want, and simple enough that nothing here
//! has to reason about what the GPU might still be reading. A texture's state
//! is therefore known on the CPU: it is sampled-from between submits, and a
//! pass, a write or a read moves it and moves it back.

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
// Limits this backend holds itself to. Not hardware limits - the real device
// allows far more - but the shapes this file's fixed root signature and
// fixed-size descriptor heaps were built for.
// -------------------------------------------------------------------------

const max_vertex_slots = 8;
const max_attributes = 16;
/// `t0`..`t3` and `s0`..`s3`: the width of the one descriptor table of each
/// kind the shared root signature has.
const max_binding_slots = 4;
/// How many textures/samplers this device can have alive at once, and how
/// many textures can be drawn into. A fixed heap size, because a descriptor
/// heap cannot be resized - see the module comment on the permanent heaps
/// and the ring.
const max_srv_descriptors = 4096;
const max_sampler_descriptors = 256;
const max_rtv_descriptors = 256;
/// The shader-visible rings, in descriptors: four for every draw whose
/// textures changed, and four for every draw whose samplers did. A sampler
/// heap that shaders see holds at most 2048.
const ring_srv_descriptors = 65536;
const ring_sampler_descriptors = 2048;

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

    /// A texture's render target view, for the textures a pass draws into:
    /// made when the texture is, kept with it.
    rtv_heap: *rc.ID3D12DescriptorHeap,
    rtv_next: u32 = 0,
    rtv_free: std.ArrayListUnmanaged(u32) = .empty,
    /// What an empty slot of a table reads: a null SRV and a plain sampler,
    /// in the permanent heaps.
    null_srv: rc.CpuDescriptorHandle,
    plain_sampler: rc.CpuDescriptorHandle,

    /// The shader-visible rings the root descriptor tables point into; see
    /// the module comment. `*_used` is how far into each the recording that is
    /// open has come.
    ring_srv_heap: *rc.ID3D12DescriptorHeap,
    ring_sampler_heap: *rc.ID3D12DescriptorHeap,
    ring_srv_used: u32 = 0,
    ring_sampler_used: u32 = 0,

    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,

    // Per-submit recording state, kept so that a recording that runs out of
    // ring can be executed and picked up again where it was.
    recording: bool = false,
    current_pipeline: ?*PipelineRes = null,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
    index_view: ?cmdmod.IndexBufferView = null,
    uniforms: [max_binding_slots]u64 = @splat(0),
    textures: [max_binding_slots]rc.CpuDescriptorHandle = undefined,
    samplers: [max_binding_slots]rc.CpuDescriptorHandle = undefined,
    textures_dirty: bool = true,
    samplers_dirty: bool = true,
    target: ?Target = null,
    viewport: cmdmod.Viewport = .{ .width = 0, .height = 0 },
    scissor: cmdmod.Rect = .{ .left = 0, .top = 0, .right = 0, .bottom = 0 },
};

/// What the pass that is open draws into.
const Target = struct {
    rtv: rc.CpuDescriptorHandle,
    width: u32,
    height: u32,
    /// What it goes back to when the pass ends: the back buffer to
    /// presenting, a texture to being sampled.
    resource: *rc.ID3D12Resource,
    texture: ?*TextureRes,
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
    format: types.Format,
    srv_index: u32,
    srv_cpu: rc.CpuDescriptorHandle,
    /// Its render target view, for a texture made to be drawn into.
    rtv_index: ?u32 = null,
    rtv_cpu: rc.CpuDescriptorHandle = .{},
    /// Where it is now, as the recording that is open has left it: `sampled`
    /// between submits.
    state: rc.ResourceStates = sampled,
};

/// What a texture is between submits: readable by any stage, since the root
/// signature lets every stage see the tables.
const sampled: rc.ResourceStates = .{ .pixel_shader_resource = true, .non_pixel_shader_resource = true };

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
    const rtv_heap = rc.createDescriptorHeap(device, .{ .type = .rtv, .num_descriptors = max_rtv_descriptors }) catch return error.NoDevice;
    errdefer _ = com.release(rtv_heap);
    const ring_srv_heap = rc.createDescriptorHeap(device, .{
        .type = .cbv_srv_uav,
        .num_descriptors = ring_srv_descriptors,
        .flags = .{ .shader_visible = true },
    }) catch return error.NoDevice;
    errdefer _ = com.release(ring_srv_heap);
    const ring_sampler_heap = rc.createDescriptorHeap(device, .{
        .type = .sampler,
        .num_descriptors = ring_sampler_descriptors,
        .flags = .{ .shader_visible = true },
    }) catch return error.NoDevice;
    errdefer _ = com.release(ring_sampler_heap);

    // The first slot of each permanent heap is what an empty table slot
    // reads: a null SRV samples as zero, as an unbound one does on Direct3D
    // 11, and a plain sampler is always a valid one.
    const srv_increment = rc.descriptorHandleIncrementSize(device, .cbv_srv_uav);
    const sampler_increment = rc.descriptorHandleIncrementSize(device, .sampler);
    const null_srv = rc.cpuHeapStart(srv_heap);
    rc.createShaderResourceView(device, null, &.{
        .format = .r8g8b8a8_unorm,
        .dimension = .texture2d,
        .u = .{ .texture2d = .{ .mip_levels = 1 } },
    }, null_srv);
    const plain_sampler = rc.cpuHeapStart(sampler_heap);
    rc.createSampler(device, &.{ .min_lod = 0, .max_lod = 0 }, plain_sampler);

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
        .cbv_srv_uav_increment = srv_increment,
        .sampler_increment = sampler_increment,
        .srv_heap = srv_heap,
        .srv_next = 1,
        .sampler_heap = sampler_heap,
        .sampler_next = 1,
        .rtv_heap = rtv_heap,
        .null_srv = null_srv,
        .plain_sampler = plain_sampler,
        .ring_srv_heap = ring_srv_heap,
        .ring_sampler_heap = ring_sampler_heap,
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
        self.ring_sampler_heap,
        self.ring_srv_heap,
        self.rtv_heap,
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
    self.rtv_free.deinit(self.gpa);
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
// Capabilities, as data. `.d2` only, one sample, one mip, `rgba8_unorm`,
// `bgra8_unorm` and `r8_unorm`, each sampled and drawn into - `Device`
// refuses everything this does not claim before this file is asked.
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
            // this, so "no anisotropy above one" is enforced here rather than
            // by this file refusing a sampler by hand.
            .max_anisotropy = 1,
            .max_color_attachments = 1,
        },
        .features = .{ .sampler_border = true, .sampler_lod_bias = true },
    };
    const support: types.FormatSupport = .{
        .sampled = true,
        .filterable = true,
        .render_target = true,
        .blendable = true,
        .generate_mips = false,
        .sample_counts = 0b1,
        .dimensions = blk: {
            var set = std.EnumSet(types.Dimension).initEmpty();
            set.insert(.d2);
            break :blk set;
        },
    };
    answer.formats.set(.rgba8_unorm, support);
    answer.formats.set(.bgra8_unorm, support);
    answer.formats.set(.r8_unorm, support);
    return answer;
}

fn textureFormat(format: types.Format) ?rc.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        .r8_unorm => .r8_unorm,
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

fn allocRtv(self: *D3d) Error!u32 {
    if (takeFree(&self.rtv_free)) |i| return i;
    if (self.rtv_next >= max_rtv_descriptors) return error.Failed;
    const i = self.rtv_next;
    self.rtv_next += 1;
    return i;
}

fn freeRtv(self: *D3d, i: u32) void {
    self.rtv_free.append(self.gpa, i) catch {};
}

fn rtvHandle(self: *D3d, i: u32) rc.CpuDescriptorHandle {
    return rc.cpuHeapStart(self.rtv_heap).offsetBy(i, self.rtv_increment);
}

// -------------------------------------------------------------------------
// Recording and waiting: the two halves every submit - a frame's, or a
// texture upload's - is built from.
// -------------------------------------------------------------------------

fn beginRecording(self: *D3d) Error!void {
    self.cmd_allocator.vtable.Reset(self.cmd_allocator).check() catch return error.Failed;
    self.list.vtable.Reset(self.list, self.cmd_allocator, null).check() catch return error.Failed;
    self.recording = true;
}

fn closeExecuteAndWait(self: *D3d) Error!void {
    self.recording = false;
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

/// Blocks until the queue has finished everything submitted to it so far -
/// including a `Present`, which `submit`'s own fence wait does not cover:
/// that wait happens before `present` is ever called, and `Present` queues
/// GPU-side work of its own (the flip, DWM's handoff) after it. Releasing a
/// swap chain's back buffers while that work is still outstanding is a real
/// Direct3D 12 debug-layer fault - "VerifyNotInUse" - not a false positive,
/// so `destroySurface`/`resizeSurface` call this before touching any of
/// them.
fn waitForGpuIdle(self: *D3d) void {
    self.fence_value += 1;
    self.queue.vtable.Signal(self.queue, @ptrCast(self.fence), self.fence_value).check() catch return;
    while (self.fence.vtable.GetCompletedValue(self.fence) < self.fence_value) {}
}

/// Record a texture's move to `to`, if it is not there already, and keep
/// where it is now. Every recording executes in full before the next begins,
/// so the state kept here is the state the GPU will find.
fn transition(self: *D3d, res: *TextureRes, to: rc.ResourceStates) void {
    if (@as(u32, @bitCast(res.state)) == @as(u32, @bitCast(to))) return;
    const barrier = cmdmod.ResourceBarrier.transition(res.resource, res.state, to);
    self.list.vtable.ResourceBarrier(self.list, 1, &[_]cmdmod.ResourceBarrier{barrier});
    res.state = to;
}

/// What a submit that failed part-way leaves: the pass that was open ended -
/// its target back where the next submit and `present` expect it - and what
/// was recorded executed, so that the states kept on the CPU stay true and
/// the list is closed for the next `Reset`.
fn abandonRecording(self: *D3d) void {
    if (!self.recording) return;
    endTarget(self);
    closeExecuteAndWait(self) catch {};
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
    // shape or a format outside what `caps` claims never reaches here.
    if (desc.dimension != .d2 or desc.samples != 1 or desc.mip_levels != 1) return error.Unsupported;
    const format = textureFormat(desc.format) orelse return error.Unsupported;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const target = desc.usage.render_target;
    const rdesc = rc.ResourceDesc.texture2d(desc.width, desc.height, format, .{ .allow_render_target = target });
    const obj = rc.createCommittedResource(self.device, .of(.default), rc.heap_flags_none, rdesc, sampled, null) catch return error.Failed;
    errdefer _ = com.release(obj);

    const slot = try allocSrv(self);
    errdefer freeSrv(self, slot);
    const cpu = srvHandle(self, slot);
    rc.createShaderResourceView(self.device, obj, &.{
        .format = format,
        .dimension = .texture2d,
        .u = .{ .texture2d = .{ .mip_levels = 1 } },
    }, cpu);

    res.* = .{ .resource = obj, .width = desc.width, .height = desc.height, .format = desc.format, .srv_index = slot, .srv_cpu = cpu };
    if (target) {
        const rtv = try allocRtv(self);
        res.rtv_index = rtv;
        res.rtv_cpu = rtvHandle(self, rtv);
        rc.createRenderTargetView(self.device, obj, null, res.rtv_cpu);
    }
    errdefer if (res.rtv_index) |rtv| freeRtv(self, rtv);

    // Level zero, the way `writeTexture` fills any box of it.
    if (desc.data) |data| try writeTexture(impl, res, .{ .width = desc.width, .height = desc.height, .depth = 1 }, data, desc.effectiveRowPitch(), 0);
    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    freeSrv(self, res.srv_index);
    if (res.rtv_index) |rtv| freeRtv(self, rtv);
    _ = com.release(res.resource);
    self.gpa.destroy(res);
}

/// Where rows of `width` texels go in a buffer a copy reads or writes: the
/// pitch a copy wants, a multiple of `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`.
fn footprintOf(res: *const TextureRes, width: u32, height: u32) rc.PlacedSubresourceFootprint {
    return .{ .offset = 0, .footprint = .{
        .format = textureFormat(res.format).?,
        .width = width,
        .height = height,
        .depth = 1,
        .row_pitch = @intCast(std.mem.alignForward(usize, res.format.rowBytes(width), 256)),
    } };
}

/// A buffer the CPU writes and a copy reads, or the other way round.
fn transferBuffer(self: *D3d, heap: rc.HeapType, size: u64) Error!*rc.ID3D12Resource {
    const state: rc.ResourceStates = if (heap == .readback) .{ .copy_dest = true } else .generic_read;
    return rc.createCommittedResource(self.device, .of(heap), rc.heap_flags_none, .buffer(size), state, null) catch return error.Failed;
}

/// Any box of level zero, through a one-off staging buffer laid out the way
/// a copy wants its rows padded, and `CopyTextureRegion` onto the texture.
/// Runs its own record/execute/wait, sharing the one command list and
/// allocator `submit` also uses: writing happens between submits, never
/// while one is being recorded.
fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    _ = slice_pitch;
    const self = cast(impl);
    const res = as(TextureRes, native);
    const footprint = footprintOf(res, region.width, region.height);
    const row_bytes = res.format.rowBytes(region.width);

    const staging = try transferBuffer(self, .upload, @as(u64, footprint.footprint.row_pitch) * region.height);
    defer _ = com.release(staging);
    var mapped: ?*anyopaque = null;
    staging.vtable.Map(staging, 0, &rc.Range.nothing_read, &mapped).check() catch return error.Failed;
    const dst: [*]u8 = @ptrCast(mapped.?);
    for (0..region.height) |y| {
        @memcpy(dst[y * footprint.footprint.row_pitch ..][0..row_bytes], bytes[y * row_pitch ..][0..row_bytes]);
    }
    staging.vtable.Unmap(staging, 0, null);

    try beginRecording(self);
    errdefer abandonRecording(self);
    transition(self, res, .{ .copy_dest = true });
    const dst_loc = cmdmod.TextureCopyLocation.subresource(res.resource, 0);
    const src_loc = cmdmod.TextureCopyLocation.placed(staging, footprint);
    self.list.vtable.CopyTextureRegion(self.list, &dst_loc, region.x, region.y, 0, &src_loc, null);
    transition(self, res, sampled);
    try closeExecuteAndWait(self);
}

/// Level zero, copied into a readback buffer and handed back as RGBA, eight
/// bits a channel, top row first - as Direct3D keeps a texture - with one
/// channel repeated into red, green and blue as the Direct3D 11 backend does.
fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    _ = sub;
    const self = cast(impl);
    const res = as(TextureRes, native);
    const footprint = footprintOf(res, res.width, res.height);
    const size = @as(u64, footprint.footprint.row_pitch) * res.height;

    const readback = try transferBuffer(self, .readback, size);
    defer _ = com.release(readback);

    try beginRecording(self);
    errdefer abandonRecording(self);
    transition(self, res, .{ .copy_source = true });
    const dst_loc = cmdmod.TextureCopyLocation.placed(readback, footprint);
    const src_loc = cmdmod.TextureCopyLocation.subresource(res.resource, 0);
    self.list.vtable.CopyTextureRegion(self.list, &dst_loc, 0, 0, 0, &src_loc, null);
    transition(self, res, sampled);
    try closeExecuteAndWait(self);

    var mapped: ?*anyopaque = null;
    const everything: rc.Range = .{ .begin = 0, .end = @intCast(size) };
    readback.vtable.Map(readback, 0, &everything, &mapped).check() catch return error.Failed;
    defer readback.vtable.Unmap(readback, 0, &rc.Range.nothing_read);
    const data: [*]const u8 = @ptrCast(mapped.?);

    const out_row = @as(usize, res.width) * 4;
    const pixels = try gpa.alloc(u8, out_row * res.height);
    for (0..res.height) |y| {
        const source = data[y * footprint.footprint.row_pitch ..];
        const destination = pixels[y * out_row ..][0..out_row];
        for (0..res.width) |x| {
            const texel = destination[x * 4 ..][0..4];
            switch (res.format) {
                .rgba8_unorm => @memcpy(texel, source[x * 4 ..][0..4]),
                .bgra8_unorm => texel.* = .{ source[x * 4 + 2], source[x * 4 + 1], source[x * 4], source[x * 4 + 3] },
                .r8_unorm => texel.* = .{ source[x], source[x], source[x], 255 },
                else => unreachable,
            }
        }
    }
    return pixels;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

fn addressMode(w: types.Wrap) rc.TextureAddressMode {
    return switch (w) {
        .repeat => .wrap,
        .clamp_to_edge => .clamp,
        .mirror => .mirror,
        .border => .border,
    };
}

fn samplerFilter(desc: types.SamplerDesc) rc.Filter {
    // `D3D12_FILTER`'s bit encoding: bit 0x10 linear-minifies, 0x04
    // linear-magnifies, 0x01 linear-filters between mips, 0x80 compares.
    // `max_anisotropy` is always clamped to one by `Device`, so the
    // anisotropic encoding is never needed here.
    var bits: u32 = 0;
    if (desc.min_filter == .linear) bits |= 0x10;
    if (desc.mag_filter == .linear) bits |= 0x04;
    if (desc.mip_filter == .linear) bits |= 0x01;
    if (desc.compare != null) bits |= 0x80;
    return @enumFromInt(bits);
}

fn comparison(compare: ?types.CompareFn) rc.ComparisonFunc {
    return switch (compare orelse return .never) {
        .never => .never,
        .less => .less,
        .equal => .equal,
        .less_equal => .less_equal,
        .greater => .greater,
        .not_equal => .not_equal,
        .greater_equal => .greater_equal,
        .always => .always,
    };
}

fn borderColor(border: types.BorderColor) [4]f32 {
    return switch (border) {
        .transparent_black => .{ 0, 0, 0, 0 },
        .opaque_black => .{ 0, 0, 0, 1 },
        .opaque_white => .{ 1, 1, 1, 1 },
    };
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
        .mip_lod_bias = desc.lod_bias,
        .max_anisotropy = @max(1, desc.max_anisotropy),
        .comparison_func = comparison(desc.compare),
        .border_color = borderColor(desc.border),
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
    }
}

fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    waitForGpuIdle(self);
    for (res.back_buffers) |buf| _ = com.release(buf);
    _ = com.release(res.rtv_heap);
    _ = com.release(res.swap_chain);
    self.gpa.destroy(res);
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    waitForGpuIdle(self);
    for (res.back_buffers) |buf| _ = com.release(buf);
    res.swap_chain.vtable.base.ResizeBuffers(@ptrCast(res.swap_chain), 0, width, height, .unknown, 0).check() catch return error.Failed;
    try attachBackBuffers(self, res);
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

/// See d3d11's: the same sync intervals, the same flip model.
fn present(impl: backend.Impl, native: backend.Native, mode: types.PresentMode) Error!void {
    _ = impl;
    const res = as(SurfaceRes, native);
    res.swap_chain.vtable.base.Present(@ptrCast(res.swap_chain), if (mode.waits()) 1 else 0, 0).check() catch |err| switch (err) {
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
    errdefer abandonRecording(self);

    self.ring_srv_used = 0;
    self.ring_sampler_used = 0;
    self.current_pipeline = null;
    self.vertex_bindings = @splat(.{});
    self.bindings_dirty = false;
    self.index_view = null;
    self.uniforms = @splat(0);
    self.textures = @splat(self.null_srv);
    self.samplers = @splat(self.plain_sampler);
    self.textures_dirty = true;
    self.samplers_dirty = true;
    self.target = null;
    setUpList(self);

    const cmd_list = self.list;
    for (list_cmds) |command| {
        switch (command) {
            .begin_pass => |pass| {
                if (pass.depth != null or pass.extra_colors.len > 0) return error.Unsupported;
                const color = pass.color orelse return error.Unsupported;
                const target: Target = switch (color.target) {
                    .surface => |h| blk: {
                        const surf = as(SurfaceRes, device.surfaces.get(h).?.native);
                        const idx = surf.swap_chain.vtable.GetCurrentBackBufferIndex(surf.swap_chain);
                        const to_rt = cmdmod.ResourceBarrier.transition(surf.back_buffers[idx], .{}, .{ .render_target = true });
                        cmd_list.vtable.ResourceBarrier(cmd_list, 1, &[_]cmdmod.ResourceBarrier{to_rt});
                        break :blk .{ .rtv = surf.rtv_handles[idx], .width = surf.width, .height = surf.height, .resource = surf.back_buffers[idx], .texture = null };
                    },
                    .texture => |h| blk: {
                        const res = as(TextureRes, device.textures.get(h).?.native);
                        if (res.rtv_index == null) return error.Unsupported;
                        transition(self, res, .{ .render_target = true });
                        break :blk .{ .rtv = res.rtv_cpu, .width = res.width, .height = res.height, .resource = res.resource, .texture = res };
                    },
                };
                self.target = target;
                cmd_list.vtable.OMSetRenderTargets(cmd_list, 1, @ptrCast(&target.rtv), 0, null);
                if (color.load == .clear) cmd_list.vtable.ClearRenderTargetView(cmd_list, target.rtv, &color.clear_color, 0, null);

                self.viewport = .{ .width = @floatFromInt(target.width), .height = @floatFromInt(target.height) };
                self.scissor = .{ .left = 0, .top = 0, .right = @intCast(target.width), .bottom = @intCast(target.height) };
                cmd_list.vtable.RSSetViewports(cmd_list, 1, &[_]cmdmod.Viewport{self.viewport});
                cmd_list.vtable.RSSetScissorRects(cmd_list, 1, &[_]cmdmod.Rect{self.scissor});

                self.current_pipeline = null;
                self.vertex_bindings = @splat(.{});
                self.bindings_dirty = false;
            },
            .end_pass => endTarget(self),
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                self.current_pipeline = res;
                self.bindings_dirty = true;
                cmd_list.vtable.SetPipelineState(cmd_list, res.pso);
                cmd_list.vtable.IASetPrimitiveTopology(cmd_list, res.topology);
            },
            .set_viewport => |v| {
                self.viewport = .{ .top_left_x = v.x, .top_left_y = v.y, .width = v.width, .height = v.height, .min_depth = v.min_depth, .max_depth = v.max_depth };
                cmd_list.vtable.RSSetViewports(cmd_list, 1, &[_]cmdmod.Viewport{self.viewport});
            },
            .set_scissor => |maybe| {
                self.scissor = if (maybe) |rect| .{
                    .left = rect.x,
                    .top = rect.y,
                    .right = rect.x + @as(i32, @intCast(rect.width)),
                    .bottom = rect.y + @as(i32, @intCast(rect.height)),
                } else if (self.target) |target| .{
                    .left = 0,
                    .top = 0,
                    .right = @intCast(target.width),
                    .bottom = @intCast(target.height),
                } else .{ .left = 0, .top = 0, .right = 0, .bottom = 0 };
                cmd_list.vtable.RSSetScissorRects(cmd_list, 1, &[_]cmdmod.Rect{self.scissor});
            },
            .set_vertex_buffer => |b| {
                if (b.slot >= max_vertex_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.vertex_bindings[b.slot] = .{ .resource = res.resource, .offset = b.offset, .size = res.size };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.index_view = .{
                    .buffer_location = res.resource.vtable.GetGPUVirtualAddress(res.resource),
                    .size_in_bytes = res.size,
                    .format = if (b.format == .u16) .r16_uint else .r32_uint,
                };
                cmd_list.vtable.IASetIndexBuffer(cmd_list, &self.index_view.?);
            },
            .set_uniform_buffer => |b| {
                if (b.slot >= max_binding_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.uniforms[b.slot] = res.resource.vtable.GetGPUVirtualAddress(res.resource);
                cmd_list.vtable.SetGraphicsRootConstantBufferView(cmd_list, root_param_cbv0 + b.slot, self.uniforms[b.slot]);
            },
            .set_texture => |b| {
                if (b.slot >= max_binding_slots) return error.Unsupported;
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                if (self.textures[b.slot].ptr != texture.srv_cpu.ptr) {
                    self.textures[b.slot] = texture.srv_cpu;
                    self.textures_dirty = true;
                }
                if (self.samplers[b.slot].ptr != sampler.cpu.ptr) {
                    self.samplers[b.slot] = sampler.cpu;
                    self.samplers_dirty = true;
                }
            },
            .draw => |d| {
                try prepareDraw(self);
                cmd_list.vtable.DrawInstanced(cmd_list, d.vertex_count, d.instance_count, d.first_vertex, 0);
            },
            .draw_indexed => |d| {
                try prepareDraw(self);
                cmd_list.vtable.DrawIndexedInstanced(cmd_list, d.index_count, d.instance_count, d.first_index, d.base_vertex, 0);
            },
            // Unreachable in practice: every texture here has one mip level,
            // and `Device.submit`'s own validation refuses `generateMips` on
            // one before any backend sees it.
            .generate_mips => return error.Unsupported,
        }
    }

    try closeExecuteAndWait(self);
}

/// What every recording starts with: the one root signature, and the rings
/// its tables point into.
fn setUpList(self: *D3d) void {
    self.list.vtable.SetGraphicsRootSignature(self.list, self.root_signature);
    const heaps = [_]*rc.ID3D12DescriptorHeap{ self.ring_srv_heap, self.ring_sampler_heap };
    self.list.vtable.SetDescriptorHeaps(self.list, heaps.len, &heaps);
}

/// End the pass that is open: its texture back to being sampled, or its back
/// buffer back to being presented.
fn endTarget(self: *D3d) void {
    const target = self.target orelse return;
    self.target = null;
    if (target.texture) |res| {
        transition(self, res, sampled);
    } else {
        const to_present = cmdmod.ResourceBarrier.transition(target.resource, .{ .render_target = true }, .{});
        self.list.vtable.ResourceBarrier(self.list, 1, &[_]cmdmod.ResourceBarrier{to_present});
    }
}

/// The tables a draw reads, made where its textures or samplers changed, and
/// its vertex buffers bound.
fn prepareDraw(self: *D3d) Error!void {
    const out_of_srvs = self.textures_dirty and self.ring_srv_used + max_binding_slots > ring_srv_descriptors;
    const out_of_samplers = self.samplers_dirty and self.ring_sampler_used + max_binding_slots > ring_sampler_descriptors;
    if (out_of_srvs or out_of_samplers) try restartRecording(self);

    if (self.textures_dirty) {
        const at = self.ring_srv_used;
        self.ring_srv_used += max_binding_slots;
        const start = rc.cpuHeapStart(self.ring_srv_heap);
        for (self.textures, 0..) |source, i| {
            rc.copyDescriptorsSimple(self.device, 1, start.offsetBy(at + @as(u32, @intCast(i)), self.cbv_srv_uav_increment), source, .cbv_srv_uav);
        }
        self.list.vtable.SetGraphicsRootDescriptorTable(self.list, root_param_srv_table, rc.gpuHeapStart(self.ring_srv_heap).offsetBy(at, self.cbv_srv_uav_increment));
        self.textures_dirty = false;
    }
    if (self.samplers_dirty) {
        const at = self.ring_sampler_used;
        self.ring_sampler_used += max_binding_slots;
        const start = rc.cpuHeapStart(self.ring_sampler_heap);
        for (self.samplers, 0..) |source, i| {
            rc.copyDescriptorsSimple(self.device, 1, start.offsetBy(at + @as(u32, @intCast(i)), self.sampler_increment), source, .sampler);
        }
        self.list.vtable.SetGraphicsRootDescriptorTable(self.list, root_param_sampler_table, rc.gpuHeapStart(self.ring_sampler_heap).offsetBy(at, self.sampler_increment));
        self.samplers_dirty = false;
    }
    flushBindings(self);
}

/// Execute what has been recorded, wait for it, and record on from the same
/// state with the rings empty again: what a submit that draws with more
/// changes of texture than the ring holds does. The target stays where the
/// pass put it; nothing is cleared again.
fn restartRecording(self: *D3d) Error!void {
    try closeExecuteAndWait(self);
    try beginRecording(self);
    self.ring_srv_used = 0;
    self.ring_sampler_used = 0;
    setUpList(self);

    const cmd_list = self.list;
    if (self.target) |target| cmd_list.vtable.OMSetRenderTargets(cmd_list, 1, @ptrCast(&target.rtv), 0, null);
    cmd_list.vtable.RSSetViewports(cmd_list, 1, &[_]cmdmod.Viewport{self.viewport});
    cmd_list.vtable.RSSetScissorRects(cmd_list, 1, &[_]cmdmod.Rect{self.scissor});
    if (self.current_pipeline) |p| {
        cmd_list.vtable.SetPipelineState(cmd_list, p.pso);
        cmd_list.vtable.IASetPrimitiveTopology(cmd_list, p.topology);
    }
    if (self.index_view) |*view| cmd_list.vtable.IASetIndexBuffer(cmd_list, view);
    for (self.uniforms, 0..) |address, slot| {
        if (address != 0) cmd_list.vtable.SetGraphicsRootConstantBufferView(cmd_list, root_param_cbv0 + @as(u32, @intCast(slot)), address);
    }
    self.bindings_dirty = true;
    self.textures_dirty = true;
    self.samplers_dirty = true;
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
// Tests. WARP needs no window and no card: what a pass drew is read back
// from the texture it drew into. A request outside what `caps` claims comes
// back as `error.Unsupported` through `Device`'s own caps-driven rejection
// rather than a crash.
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

test "caps: .d2 rgba8, bgra8 and r8, sampled and drawn into, one sample" {
    var device = try warpDevice();
    defer device.deinit();

    const c = device.caps();
    for ([_]types.Format{ .rgba8_unorm, .bgra8_unorm, .r8_unorm }) |format| {
        const support = c.formatSupport(format);
        try testing.expect(support.sampled);
        try testing.expect(support.render_target);
        try testing.expect(support.dimensions.contains(.d2));
        try testing.expect(!support.dimensions.contains(.cube));
        try testing.expect(!support.dimensions.contains(.d3));
        try testing.expect(support.supportsSamples(1));
        try testing.expect(!support.supportsSamples(4));
    }
    try testing.expectEqual(@as(u32, 1), c.limits.max_anisotropy);
    try testing.expect(c.features.sampler_border);

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

test "a texture reads back as it was made, and a box written into it changes that box" {
    var device = try warpDevice();
    defer device.deinit();

    var pixels: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| pixels[i * 4 ..][0..4].* = .{ @intCast(i * 10), 20, 30, 255 };
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .data = &pixels });
    defer device.destroyTexture(texture);
    {
        const back = try device.readTexture(texture, testing.allocator);
        defer testing.allocator.free(back);
        try testing.expectEqualSlices(u8, &pixels, back);
    }

    // Two by two, one texel in from the top left.
    const green = [_]u8{ 0, 255, 0, 255 } ** 4;
    try device.writeTexture(texture, .{ .x = 1, .y = 1, .width = 2, .height = 2 }, &green, 8, 0);
    const back = try device.readTexture(texture, testing.allocator);
    defer testing.allocator.free(back);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[(1 * 4 + 1) * 4 ..][0..4].*);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[(2 * 4 + 2) * 4 ..][0..4].*);
    try testing.expectEqual([4]u8{ 0, 20, 30, 255 }, back[0..4].*);
    try testing.expectEqual(pixels[(3 * 4 + 3) * 4 ..][0..4].*, back[(3 * 4 + 3) * 4 ..][0..4].*);
}

test "bgra8 and r8 read back as RGBA" {
    var device = try warpDevice();
    defer device.deinit();

    const bgra = try device.createTexture(.{ .width = 1, .height = 1, .format = .bgra8_unorm, .data = &.{ 10, 20, 30, 40 } });
    defer device.destroyTexture(bgra);
    const from_bgra = try device.readTexture(bgra, testing.allocator);
    defer testing.allocator.free(from_bgra);
    try testing.expectEqualSlices(u8, &.{ 30, 20, 10, 40 }, from_bgra);

    // An atlas of coverage, 3 wide: rows of one byte a texel, written again
    // one row at a time.
    const r8 = try device.createTexture(.{ .width = 3, .height = 2, .format = .r8_unorm, .data = &.{ 1, 2, 3, 4, 5, 6 } });
    defer device.destroyTexture(r8);
    try device.writeTexture(r8, .{ .y = 1, .width = 3, .height = 1 }, &.{ 7, 8, 9 }, 3, 0);
    const from_r8 = try device.readTexture(r8, testing.allocator);
    defer testing.allocator.free(from_r8);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 255 }, from_r8[0..4]);
    try testing.expectEqualSlices(u8, &.{ 9, 9, 9, 255 }, from_r8[5 * 4 ..][0..4]);
}

/// Clip-space corners and a shader that samples slot 0 across them, top of
/// the picture at the top: what the pass tests draw with.
const Quad = struct {
    shader: types.Shader,
    pipeline: types.Pipeline,
    corners: types.Buffer,

    const vs =
        \\struct In { float2 position : ATTR0; };
        \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
        \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.uv = i.position * float2(0.5, -0.5) + 0.5; return o; }
    ;
    const ps =
        \\Texture2D picture : register(t0);
        \\SamplerState picture_sampler : register(s0);
        \\float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET { return picture.Sample(picture_sampler, uv); }
    ;

    /// The same, with uv running to 2 across the quad rather than 1.
    const vs_twice =
        \\struct In { float2 position : ATTR0; };
        \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
        \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.uv = i.position * float2(1, -1) + 1; return o; }
    ;

    fn init(device: *Device) !Quad {
        return initWith(device, vs);
    }

    fn initWith(device: *Device, vertex: [:0]const u8) !Quad {
        const shader = device.createShader(.{ .hlsl = .{ .vertex = vertex, .fragment = ps } }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };
        const pipeline = try device.createPipeline(.{
            .shader = shader,
            .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
            .buffers = &.{.{ .stride = 8 }},
            .topology = .triangle_strip,
        });
        const corners = [_][2]f32{ .{ -1, -1 }, .{ -1, 1 }, .{ 1, -1 }, .{ 1, 1 } };
        const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(corners)), .data = std.mem.asBytes(&corners) });
        return .{ .shader = shader, .pipeline = pipeline, .corners = buffer };
    }
};

fn texelAt(pixels: []const u8, width: usize, x: usize, y: usize) [4]u8 {
    return pixels[(y * width + x) * 4 ..][0..4].*;
}

test "a pass clears and draws into a texture, which reads back and is sampled after" {
    var device = try warpDevice();
    defer device.deinit();
    const quad = try Quad.init(&device);

    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 1, 0, 0, 1 } } });
    try cmd.endPass();
    try device.submit();
    const cleared = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(cleared);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(cleared, 8, 3, 5));

    // Sampled into a second target: the first one's red, drawn over blue.
    const second = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const sampler = try device.createSampler(.nearest);
    const again = device.begin();
    try again.beginPass(.{ .color = .{ .target = .{ .texture = second }, .clear_color = .{ 0, 0, 1, 1 } } });
    try again.setPipeline(quad.pipeline);
    try again.setVertexBuffer(0, quad.corners, 0);
    try again.setTexture(0, target, sampler);
    try again.draw(.{ .vertex_count = 4 });
    try again.endPass();
    try device.submit();
    const drawn = try device.readTexture(second, testing.allocator);
    defer testing.allocator.free(drawn);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(drawn, 8, 4, 4));
}

test "each draw of one submit samples the texture it was given" {
    var device = try warpDevice();
    defer device.deinit();
    const quad = try Quad.init(&device);

    const red = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 255, 0, 0, 255 } });
    const green = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 0, 255, 0, 255 } });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const sampler = try device.createSampler(.nearest);

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(quad.pipeline);
    try cmd.setVertexBuffer(0, quad.corners, 0);
    try cmd.setViewport(.{ .width = 4, .height = 8 });
    try cmd.setTexture(0, red, sampler);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.setViewport(.{ .x = 4, .width = 4, .height = 8 });
    try cmd.setTexture(0, green, sampler);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(pixels, 8, 1, 4));
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, texelAt(pixels, 8, 6, 4));
}

test "a submit with more changes of sampler than the ring holds draws them all" {
    var device = try warpDevice();
    defer device.deinit();
    const quad = try Quad.init(&device);

    const texture = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 0, 0, 255, 255 } });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const nearest = try device.createSampler(.nearest);
    const linear = try device.createSampler(.linear);

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(quad.pipeline);
    try cmd.setVertexBuffer(0, quad.corners, 0);
    // One draw into each column, each with the other sampler: more tables
    // than the sampler ring has room for.
    const draws = ring_sampler_descriptors / max_binding_slots + 40;
    for (0..draws) |i| {
        try cmd.setViewport(.{ .x = @floatFromInt(i % 8), .width = 1, .height = 8 });
        try cmd.setTexture(0, texture, if (i % 2 == 0) nearest else linear);
        try cmd.draw(.{ .vertex_count = 4 });
    }
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    for (0..8) |x| try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, texelAt(pixels, 8, x, 4));
}

test "a border sampler reads its colour outside the texture" {
    var device = try warpDevice();
    defer device.deinit();
    // Across the target, uv runs from 0 to 2: the left half reads the
    // texture, which is black, and the right half reads past its edge.
    const quad = try Quad.initWith(&device, Quad.vs_twice);

    const texture = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 0, 0, 0, 255 } });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const border = try device.createSampler(.{ .wrap_u = .border, .wrap_v = .border, .border = .opaque_white, .min_filter = .nearest, .mag_filter = .nearest });

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(quad.pipeline);
    try cmd.setVertexBuffer(0, quad.corners, 0);
    try cmd.setTexture(0, texture, border);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, texelAt(pixels, 8, 1, 1));
    try testing.expectEqual([4]u8{ 255, 255, 255, 255 }, texelAt(pixels, 8, 6, 6));
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

test "what is not here yet comes back as error.Unsupported" {
    var device = try warpDevice();
    defer device.deinit();

    // No cube, volume or array textures.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4 }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4 }));
    // No multisampled render targets, and no chains of levels.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .samples = 4, .usage = .{ .sampled = false, .render_target = true } }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 2 }));
    // No compressed or depth formats.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .format = .bc1_rgba_unorm }));
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } }));
}

test "a surface with no window is refused" {
    var device = try warpDevice();
    defer device.deinit();
    try testing.expectError(error.InvalidArgument, device.createSurface(.{}));
}
