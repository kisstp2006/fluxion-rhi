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
//! backend on purpose: `.d2` textures, no compute - and within that, what a
//! renderer asks: textures drawn into and sampled after, written in part and
//! read back, a level at a time, and their chains of levels filled by
//! `generateMips`, which Direct3D 12 has no call for: each level is drawn
//! from the one above with a linear filter (see `generateMips`). What the narrowness buys is simplicity in the parts a real
//! Direct3D 12 renderer usually spends the most code on:
//!
//! **Buffers are always upload-heap.** CPU-writable, mapped once at creation
//! and kept mapped for the buffer's life - `updateBuffer` is a `memcpy` into
//! memory the GPU already sees. One the GPU may still be reading is written
//! into another copy - a version - that starts as the buffer's bytes kept
//! on the CPU, the way a Direct3D 11 driver renames a buffer written with
//! discard (see below). No device-local buffer, no copy queue.
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
//! given, not whatever the last `set_texture` of the list left. Each
//! recording slot (below) has its own part of the ring, which starts over
//! when the slot is recorded into again - by then the GPU is done with it;
//! one that runs out executes what it has and records on from the same state
//! in the next slot.
//!
//! **Recordings in flight.** A submit, an upload or a readback records into
//! the next of `ring_size` slots - a command allocator and its part of the
//! descriptor rings - executes it and signals the fence with the next value,
//! without waiting; a slot is waited for, with `SetEventOnCompletion`, only
//! when it comes round again. What is destroyed while the GPU may still use
//! it, and a staging buffer, is released once the fence is past the last
//! value signalled; a readback waits for its own. The queue runs lists in
//! order, so a texture's state kept on the CPU is the state the next list
//! finds: sampled-from between submits, and a pass, a write or a read moves
//! it and moves it back.
//!
//! **One list a frame.** A `submit` or an upload goes on at the end of the
//! list that is open, rather than into one of its own, and the list is
//! executed when something needs it done: a present, a readback, a
//! surface's end, or `max_lists` lists. `ExecuteCommandLists` has a cost of
//! its own on the GPU, and a frame of many passes pays it once.

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
const readback = @import("readback.zig");
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
/// `b0`..`b7`: the root CBVs the shared root signature has.
const uniform_slots = 8;
/// `t0`..`t15` and `s0`..`s15`: the width of its one descriptor table of
/// each kind.
const texture_slots = types.max_texture_slots;
/// How many textures/samplers this device can have alive at once, and how
/// many textures can be drawn into. A fixed heap size, because a descriptor
/// heap cannot be resized - see the module comment on the permanent heaps
/// and the ring.
const max_srv_descriptors = 4096;
const max_sampler_descriptors = 256;
const max_rtv_descriptors = 256;
const max_dsv_descriptors = 64;
/// The shader-visible rings, in descriptors: a table's worth for every draw
/// whose textures changed, and one for every draw whose samplers did. A
/// sampler heap that shaders see holds at most 2048.
const ring_srv_descriptors = 65536;
const ring_sampler_descriptors = 2048;

/// How many recordings can be on their way to the GPU at once: one waits
/// for the one this many before it, and no other. Each has a part of the
/// descriptor rings.
const ring_size = 4;
const slot_srv_descriptors = ring_srv_descriptors / ring_size;
const slot_sampler_descriptors = ring_sampler_descriptors / ring_size;

/// The most `submit`s one list takes before it is executed: a program that
/// never presents still has its work done.
const max_lists = 64;

const root_param_cbv0 = 0;
const root_param_srv_table = uniform_slots;
const root_param_sampler_table = uniform_slots + 1;

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

/// `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING`, and `DXGI_PRESENT_ALLOW_TEARING`: a
/// swap chain made with the first may be presented with the second, which puts
/// a frame up at once, mid-refresh, where nothing composites the window.
const swap_chain_allow_tearing: u32 = 0x800;
const present_allow_tearing: u32 = 0x200;

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
    /// The one list, reset onto the allocator of the slot at `at`.
    list: *cmdmod.ID3D12GraphicsCommandList,
    slots: [ring_size]Slot,
    at: usize = 0,
    fence: *rc.ID3D12Fence,
    /// The value the last execute signalled, and the last the GPU is known
    /// to have reached.
    fence_value: u64 = 0,
    completed: u64 = 0,
    graveyard: std.ArrayListUnmanaged(Grave) = .empty,
    /// The buffers the submit being recorded binds: used again after a list
    /// that runs out of ring is executed and recording goes on.
    used_buffers: std.ArrayListUnmanaged(*BufferRes) = .empty,
    /// How many submits the open list holds.
    lists: u32 = 0,

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
    /// The same for a depth texture's depth-stencil view.
    dsv_heap: *rc.ID3D12DescriptorHeap,
    dsv_increment: u32,
    dsv_next: u32 = 0,
    dsv_free: std.ArrayListUnmanaged(u32) = .empty,
    /// What an empty slot of a table reads: a null SRV and a plain sampler,
    /// in the permanent heaps.
    null_srv: rc.CpuDescriptorHandle,
    plain_sampler: rc.CpuDescriptorHandle,
    /// What work on one level at a time draws with, in the permanent heaps
    /// and written again each time: a list reads a render target view, and
    /// copies a shader resource view into its ring, as it records the call
    /// that names it, so one of each serves every texture. The pass into a
    /// level other than the first has its own render target view, which the
    /// target is bound with again if the recording restarts.
    scratch_srv: rc.CpuDescriptorHandle,
    pass_rtv: rc.CpuDescriptorHandle,
    mip_rtv: rc.CpuDescriptorHandle,
    /// Linear, clamped to the edge: what a level is filtered down with.
    mip_sampler: rc.CpuDescriptorHandle,
    /// `generateMips`'s shader, compiled the first time it is asked for, and
    /// its pipeline for each format it has filled.
    mip_shader: ?MipShader = null,
    mip_pipelines: std.EnumArray(types.Format, ?*pl.ID3D12PipelineState) = .initFill(null),

    /// The shader-visible rings the root descriptor tables point into; see
    /// the module comment. `*_used` is how far into its slot's part of each
    /// the recording that is open has come.
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
    uniforms: [uniform_slots]u64 = @splat(0),
    textures: [texture_slots]rc.CpuDescriptorHandle = undefined,
    samplers: [texture_slots]rc.CpuDescriptorHandle = undefined,
    textures_dirty: bool = true,
    samplers_dirty: bool = true,
    target: ?Target = null,
    viewport: cmdmod.Viewport = .{ .width = 0, .height = 0 },
    scissor: cmdmod.Rect = .{ .left = 0, .top = 0, .right = 0, .bottom = 0 },
};

/// What the pass that is open draws into: a colour target, a depth one, or
/// both.
const Target = struct {
    rtv: ?rc.CpuDescriptorHandle,
    dsv: ?rc.CpuDescriptorHandle = null,
    width: u32,
    height: u32,
    /// What the colour target goes back to when the pass ends: the back
    /// buffer to presenting, a texture to being sampled. A depth texture
    /// stays where depth is written, unless it is sampled too.
    resource: ?*rc.ID3D12Resource,
    texture: ?*TextureRes,
    depth: ?*TextureRes = null,
    /// Where a multisampled colour target is resolved when the pass ends.
    resolve: ?Resolve = null,

    fn bind(self: Target, list: *cmdmod.ID3D12GraphicsCommandList) void {
        const count: u32 = if (self.rtv != null) 1 else 0;
        const rtvs: ?[*]const rc.CpuDescriptorHandle = if (self.rtv) |*rtv| @ptrCast(rtv) else null;
        const dsv: ?*const rc.CpuDescriptorHandle = if (self.dsv) |*held| held else null;
        list.vtable.OMSetRenderTargets(list, count, rtvs, 0, dsv);
    }
};

/// A target a pass's samples are averaged into: a texture, or a back buffer.
const Resolve = struct {
    texture: ?*TextureRes,
    resource: *rc.ID3D12Resource,
    format: rc.Format,
};

const VertexBinding = struct {
    resource: ?*rc.ID3D12Resource = null,
    offset: u32 = 0,
    size: u32 = 0,
};

/// One recording on its way to the GPU: the allocator its list used, and
/// the fence value its execute signalled - nought for none.
const Slot = struct {
    allocator: *cmdmod.ID3D12CommandAllocator,
    value: u64 = 0,
};

/// Released once the fence is past `value`.
const Grave = struct {
    value: u64,
    what: union(enum) {
        resource: *rc.ID3D12Resource,
        pso: *pl.ID3D12PipelineState,
    },
};

/// One copy of a buffer's memory, persistently mapped - an upload-heap
/// resource may stay mapped for its whole life - and the fence value of the
/// last execute that used it.
const Version = struct {
    resource: *rc.ID3D12Resource,
    mapped: [*]u8,
    busy: u64 = 0,
};

/// A buffer: the version draws are recorded with now, the ones a write moved
/// it off, and its bytes on the CPU, which a new version starts as.
const BufferRes = struct {
    current: Version,
    spare: std.ArrayListUnmanaged(Version) = .empty,
    shadow: []u8,
    /// How far anything was ever written: a new version copies no further.
    written: usize,
    size: u32,
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
    /// Its depth-stencil view, for one of a depth format.
    dsv_index: ?u32 = null,
    dsv_cpu: rc.CpuDescriptorHandle = .{},
    /// Where it is now, as the recording that is open has left it: `sampled`
    /// between submits.
    state: rc.ResourceStates = sampled,
    /// Samples a pixel: more than one for a target a pass resolves.
    samples: u32 = 1,
    /// A depth texture that is sampled as well: back to `sampled` when the
    /// pass that drew into it ends.
    readable: bool = false,
    /// Levels in its chain; its shader resource view reads them all.
    mip_levels: u32 = 1,

    /// How big level `mip` is, across and down.
    fn levelSize(self: *const TextureRes, mip: u32) [2]u32 {
        return .{ types.mipExtent(self.width, mip), types.mipExtent(self.height, mip) };
    }
};

/// What a texture is between submits: readable by any stage, since the root
/// signature lets every stage see the tables.
const sampled: rc.ResourceStates = .{ .pixel_shader_resource = true, .non_pixel_shader_resource = true };

/// The bytecode of the shader every level of a chain is drawn with.
const MipShader = struct { vertex: []u8, pixel: []u8 };

/// A triangle over the whole target, and the level above read at the middle
/// of each of its pixels: between four texels of the level above, which a
/// linear filter averages.
const mip_vertex =
    \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
    \\Out main(uint id : SV_VertexID) {
    \\    Out o;
    \\    float2 corner = float2((id << 1) & 2, id & 2);
    \\    o.uv = corner;
    \\    o.position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
    \\    return o;
    \\}
;
const mip_pixel =
    \\Texture2D above : register(t0);
    \\SamplerState linear_clamp : register(s0);
    \\float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET { return above.SampleLevel(linear_clamp, uv, 0); }
;

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
    /// `swap_chain_allow_tearing` where the system allows tearing, nought
    /// where it does not: what the chain was made with, and what every
    /// resize has to say again.
    flags: u32,
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

    var slots: [ring_size]Slot = undefined;
    var made: usize = 0;
    errdefer for (slots[0..made]) |one| {
        _ = com.release(one.allocator);
    };
    for (&slots) |*one| {
        one.* = .{ .allocator = cmdmod.createCommandAllocator(device, .direct) catch return error.NoDevice };
        made += 1;
    }

    const list = cmdmod.createGraphicsCommandList(device, 0, .direct, slots[0].allocator) catch return error.NoDevice;
    errdefer _ = com.release(list);
    // Real Direct3D 12 API requirement: a command list is created already
    // recording, and must be closed before its first `ExecuteCommandLists`.
    list.vtable.Close(list).check() catch return error.NoDevice;

    const fence = rc.createFence(device, 0, rc.fence_flags_none) catch return error.NoDevice;
    errdefer _ = com.release(fence);

    // The one root signature every pipeline shares: a root CBV a slot, one
    // CBV_SRV_UAV table, one sampler table. `.all` visibility throughout,
    // matching the Direct3D 11 backend's own choice to bind every stage
    // rather than assume only the pixel shader samples.
    const srv_ranges = [_]pl.DescriptorRange{
        .{ .range_type = .srv, .num_descriptors = texture_slots, .base_shader_register = 0 },
    };
    const sampler_ranges = [_]pl.DescriptorRange{
        .{ .range_type = .sampler, .num_descriptors = texture_slots, .base_shader_register = 0 },
    };
    var root_params: [uniform_slots + 2]pl.RootParameter = undefined;
    for (root_params[0..uniform_slots], 0..) |*param, slot| param.* = pl.RootParameter.cbv(@intCast(slot), .all);
    root_params[root_param_srv_table] = pl.RootParameter.table(&srv_ranges, .all);
    root_params[root_param_sampler_table] = pl.RootParameter.table(&sampler_ranges, .all);
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
    const dsv_heap = rc.createDescriptorHeap(device, .{ .type = .dsv, .num_descriptors = max_dsv_descriptors }) catch return error.NoDevice;
    errdefer _ = com.release(dsv_heap);
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
    // The second of each, and the first two render target views: what work
    // on one level draws with. See `D3d.scratch_srv`.
    const mip_sampler = rc.cpuHeapStart(sampler_heap).offsetBy(1, sampler_increment);
    rc.createSampler(device, &.{}, mip_sampler);
    const rtv_increment = rc.descriptorHandleIncrementSize(device, .rtv);

    const self = try gpa.create(D3d);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .library = library,
        .dxgi_library = dxgi_library,
        .factory = factory,
        .device = device,
        .queue = queue,
        .slots = slots,
        .list = list,
        .fence = fence,
        .root_signature = root_signature,
        .rtv_increment = rtv_increment,
        .cbv_srv_uav_increment = srv_increment,
        .sampler_increment = sampler_increment,
        .srv_heap = srv_heap,
        .srv_next = 2,
        .sampler_heap = sampler_heap,
        .sampler_next = 2,
        .rtv_heap = rtv_heap,
        .rtv_next = 2,
        .dsv_heap = dsv_heap,
        .dsv_increment = rc.descriptorHandleIncrementSize(device, .dsv),
        .null_srv = null_srv,
        .plain_sampler = plain_sampler,
        .scratch_srv = rc.cpuHeapStart(srv_heap).offsetBy(1, srv_increment),
        .pass_rtv = rc.cpuHeapStart(rtv_heap),
        .mip_rtv = rc.cpuHeapStart(rtv_heap).offsetBy(1, rtv_increment),
        .mip_sampler = mip_sampler,
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
    // made; what the GPU may still be using waits in the graveyard.
    waitForGpuIdle(self);
    self.graveyard.deinit(self.gpa);
    self.used_buffers.deinit(self.gpa);
    for (self.slots) |one| _ = com.release(one.allocator);
    for (self.mip_pipelines.values) |held| if (held) |pso| {
        _ = com.release(pso);
    };
    if (self.mip_shader) |shader| {
        self.gpa.free(shader.vertex);
        self.gpa.free(shader.pixel);
    }
    if (self.compiler) |*c| c.unload();
    com.releaseAll(.{
        self.ring_sampler_heap,
        self.ring_srv_heap,
        self.rtv_heap,
        self.dsv_heap,
        self.sampler_heap,
        self.srv_heap,
        self.root_signature,
        self.fence,
        self.list,
        self.queue,
        self.device,
    });
    self.srv_free.deinit(self.gpa);
    self.sampler_free.deinit(self.gpa);
    self.rtv_free.deinit(self.gpa);
    self.dsv_free.deinit(self.gpa);
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
// Capabilities. `.d2` only and one mip; of the uncompressed formats, what
// the driver says it samples, filters, draws into, blends and multisamples -
// `Device` refuses everything this does not claim before this file is
// asked. Depth is drawn into, never sampled.
// -------------------------------------------------------------------------

/// `D3D12_FEATURE_DATA_FORMAT_SUPPORT`.
const FormatSupportQuery = extern struct {
    format: rc.Format,
    support1: u32 = 0,
    support2: u32 = 0,
};

/// `D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS`.
const SampleQuery = extern struct {
    format: rc.Format,
    sample_count: u32,
    flags: u32 = 0,
    quality_levels: u32 = 0,
};

/// `D3D12_FORMAT_SUPPORT1`'s bits this file reads.
const support_texture2d: u32 = 0x20;
const support_shader_load: u32 = 0x80;
const support_shader_sample: u32 = 0x100;
const support_render_target: u32 = 0x4000;
const support_blendable: u32 = 0x8000;
const support_depth_stencil: u32 = 0x10000;
const support_multisample_resolve: u32 = 0x40000;
const support_multisample_render_target: u32 = 0x200000;

/// The formats this file makes: every uncompressed one with a DXGI format.
const offered = [_]types.Format{
    .r8_unorm,      .rg8_unorm,        .rgba8_unorm,   .bgra8_unorm,
    .r16_float,     .rg16_float,       .rgba16_float,  .r32_float,
    .rg32_float,    .rgba32_float,     .rgb10a2_unorm, .rg11b10_float,
    .depth16_unorm, .depth24_stencil8, .depth32_float, .depth32_float_stencil8,
};

/// What the device says it does with `format`: nothing, for a query it
/// refuses.
fn askFormat(device: *d3d12.ID3D12Device, format: rc.Format) u32 {
    var query: FormatSupportQuery = .{ .format = format };
    const result = device.vtable.CheckFeatureSupport(device, .format_support, &query, @sizeOf(FormatSupportQuery));
    result.check() catch return 0;
    return query.support1;
}

/// Whether `format` takes `count` samples a pixel.
fn askSamples(device: *d3d12.ID3D12Device, format: rc.Format, count: u32) bool {
    var query: SampleQuery = .{ .format = format, .sample_count = count };
    const result = device.vtable.CheckFeatureSupport(device, .multisample_quality_levels, &query, @sizeOf(SampleQuery));
    result.check() catch return false;
    return query.quality_levels > 0;
}

fn caps(impl: backend.Impl) types.Caps {
    const self = cast(impl);
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
    var flat = std.EnumSet(types.Dimension).initEmpty();
    flat.insert(.d2);
    for (offered) |format| {
        const native = textureFormat(format).?;
        const bits = askFormat(self.device, native);
        if (bits & support_texture2d == 0) continue;
        const depth = format.isDepth();
        const drawn = bits & (if (depth) support_depth_stencil else support_render_target) != 0;
        var counts: u8 = 0b1;
        // Several samples a pixel only where they can be drawn into, and a
        // colour one resolved into one.
        const multisampled = bits & support_multisample_render_target != 0 and (depth or bits & support_multisample_resolve != 0);
        if (drawn and multisampled) for ([_]u32{ 2, 4, 8 }) |count| {
            if (askSamples(self.device, native, count)) counts |= @as(u8, @intCast(count));
        };
        // A depth format is read through a view of another format, the
        // one `depthFormats` names, and asked about as that.
        const read_bits = if (depth) askFormat(self.device, depthFormats(format).read) else bits;
        answer.formats.set(format, .{
            .sampled = read_bits & (support_shader_load | support_shader_sample) != 0,
            // For a depth format: whether a sampler that compares filters it.
            .filterable = read_bits & support_shader_sample != 0,
            .render_target = drawn,
            .blendable = !depth and bits & support_blendable != 0,
            // Each level drawn from the one above, filtered.
            .generate_mips = !depth and drawn and bits & support_shader_sample != 0,
            .sample_counts = counts,
            .dimensions = flat,
        });
    }
    return answer;
}

/// What a depth texture that is also sampled - a shadow map - is made as:
/// typeless, so that depth is written through a view of one format and read
/// through a view of another.
const DepthFormats = struct { resource: rc.Format, read: rc.Format };

fn depthFormats(format: types.Format) DepthFormats {
    return switch (format) {
        .depth16_unorm => .{ .resource = .r16_typeless, .read = .r16_unorm },
        .depth24_stencil8 => .{ .resource = .r24g8_typeless, .read = .r24_unorm_x8_typeless },
        .depth32_float => .{ .resource = .r32_typeless, .read = .r32_float },
        .depth32_float_stencil8 => .{ .resource = .r32g8x24_typeless, .read = .r32_float_x8x24_typeless },
        else => unreachable,
    };
}

fn textureFormat(format: types.Format) ?rc.Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .bgra8_unorm => .b8g8r8a8_unorm,
        .r8_unorm => .r8_unorm,
        .rg8_unorm => .r8g8_unorm,
        .r16_float => .r16_float,
        .rg16_float => .r16g16_float,
        .rgba16_float => .r16g16b16a16_float,
        .r32_float => .r32_float,
        .rg32_float => .r32g32_float,
        .rgba32_float => .r32g32b32a32_float,
        .rgb10a2_unorm => .r10g10b10a2_unorm,
        .rg11b10_float => .r11g11b10_float,
        .depth16_unorm => .d16_unorm,
        .depth24_stencil8 => .d24_unorm_s8_uint,
        .depth32_float => .d32_float,
        .depth32_float_stencil8 => .d32_float_s8x24_uint,
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

fn allocDsv(self: *D3d) Error!u32 {
    if (takeFree(&self.dsv_free)) |i| return i;
    if (self.dsv_next >= max_dsv_descriptors) return error.Failed;
    const i = self.dsv_next;
    self.dsv_next += 1;
    return i;
}

fn freeDsv(self: *D3d, i: u32) void {
    self.dsv_free.append(self.gpa, i) catch {};
}

fn dsvHandle(self: *D3d, i: u32) rc.CpuDescriptorHandle {
    return rc.cpuHeapStart(self.dsv_heap).offsetBy(i, self.dsv_increment);
}

// -------------------------------------------------------------------------
// Recording and waiting: the two halves every submit - a frame's, or a
// texture upload's - is built from.
// -------------------------------------------------------------------------

/// Record into the next slot, once the GPU is done with what was recorded
/// there last, with its part of the rings empty.
fn beginRecording(self: *D3d) Error!void {
    self.at = (self.at + 1) % ring_size;
    const slot = &self.slots[self.at];
    try waitFor(self, slot.value);
    collect(self);
    slot.allocator.vtable.Reset(slot.allocator).check() catch return error.Failed;
    self.list.vtable.Reset(self.list, slot.allocator, null).check() catch return error.Failed;
    self.recording = true;
    self.ring_srv_used = 0;
    self.ring_sampler_used = 0;
    self.lists = 0;
    setUpList(self);
}

/// Execute the open list, if there is one.
fn flush(self: *D3d) Error!void {
    if (self.recording) try closeAndExecute(self);
}

/// Close what was recorded, execute it and signal the next fence value,
/// without waiting for it.
fn closeAndExecute(self: *D3d) Error!void {
    self.recording = false;
    self.list.vtable.Close(self.list).check() catch return error.Failed;
    const lists = [_]*cmdmod.ID3D12GraphicsCommandList{self.list};
    cmdmod.executeCommandLists(self.queue, &lists);
    self.fence_value += 1;
    self.queue.vtable.Signal(self.queue, @ptrCast(self.fence), self.fence_value).check() catch return error.Failed;
    self.slots[self.at].value = self.fence_value;
    if (self.debug) {
        self.device.vtable.GetDeviceRemovedReason(self.device).check() catch return error.DeviceLost;
    }
}

/// Wait until the GPU has reached `value`, sleeping rather than spinning.
fn waitFor(self: *D3d, value: u64) Error!void {
    if (value <= self.completed) return;
    self.completed = self.fence.vtable.GetCompletedValue(self.fence);
    if (value <= self.completed) return;
    self.fence.vtable.SetEventOnCompletion(self.fence, value, null).check() catch return error.DeviceLost;
    self.completed = @max(self.completed, value);
}

/// Release `what` once the GPU is past everything executed so far, or now
/// when it already is.
fn bury(self: *D3d, what: @FieldType(Grave, "what")) void {
    if (!self.recording and self.completed >= self.fence_value) return free(what);
    // The open list, when there is one, may use it too.
    const value = if (self.recording) self.fence_value + 1 else self.fence_value;
    self.graveyard.append(self.gpa, .{ .value = value, .what = what }) catch {
        // No room to wait: wait for it instead.
        waitForGpuIdle(self);
        free(what);
    };
}

fn free(what: @FieldType(Grave, "what")) void {
    switch (what) {
        .resource => |resource| _ = com.release(resource),
        .pso => |pso| _ = com.release(pso),
    }
}

/// Release what the GPU is done with.
fn collect(self: *D3d) void {
    self.completed = @max(self.completed, self.fence.vtable.GetCompletedValue(self.fence));
    var i: usize = 0;
    while (i < self.graveyard.items.len) {
        const grave = self.graveyard.items[i];
        if (grave.value > self.completed) {
            i += 1;
            continue;
        }
        _ = self.graveyard.swapRemove(i);
        free(grave.what);
    }
}

/// Bound by the submit being recorded: the version it has now is the one
/// the list reads, busy until the fence passes the value the list's execute
/// will signal.
fn use(self: *D3d, res: *BufferRes) Error!*rc.ID3D12Resource {
    self.used_buffers.append(self.gpa, res) catch return error.OutOfMemory;
    res.current.busy = self.fence_value + 1;
    return res.current.resource;
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
    flush(self) catch {};
    self.fence_value += 1;
    self.queue.vtable.Signal(self.queue, @ptrCast(self.fence), self.fence_value).check() catch return;
    waitFor(self, self.fence_value) catch return;
    collect(self);
}

/// Record a texture's move to `to`, if it is not there already, and keep
/// where it is now. The queue runs the lists in the order they were
/// recorded, so the state kept here is the state the GPU will find.
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
    closeAndExecute(self) catch {};
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) Error!backend.Native {
    const self = cast(impl);
    const res = try self.gpa.create(BufferRes);
    errdefer self.gpa.destroy(res);

    // A whole number of 256-byte blocks: a block bound from an offset is read
    // by the shader as far as it declares, and that stays in the buffer.
    const size: u32 = @intCast(if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 256) else desc.size);
    const shadow = try self.gpa.alloc(u8, @max(size, 1));
    errdefer self.gpa.free(shadow);
    @memset(shadow, 0);
    const version = try uploadVersion(self, size);
    errdefer _ = com.release(version.resource);
    @memset(version.mapped[0..size], 0);
    // Nought throughout, as it was made.
    res.* = .{ .current = version, .shadow = shadow, .written = size, .size = size };
    if (desc.data) |data| {
        @memcpy(version.mapped[0..data.len], data);
        @memcpy(shadow[0..data.len], data);
    }
    return res;
}

/// An upload-heap buffer's memory, mapped for its life.
fn uploadVersion(self: *D3d, size: u32) Error!Version {
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
    return .{ .resource = obj, .mapped = @ptrCast(mapped.?) };
}

fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    bury(self, .{ .resource = res.current.resource });
    for (res.spare.items) |version| bury(self, .{ .resource = version.resource });
    res.spare.deinit(self.gpa);
    self.gpa.free(res.shadow);
    self.gpa.destroy(res);
}

/// Write into the version the GPU is not reading: the current one when no
/// list it may still be running used it, another when one did.
fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) Error!void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    const end = offset + bytes.len;
    if (res.current.busy > self.completed) self.completed = self.fence.vtable.GetCompletedValue(self.fence);
    if (res.current.busy > self.completed) {
        res.spare.ensureUnusedCapacity(self.gpa, 1) catch return error.OutOfMemory;
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
fn spareVersion(self: *D3d, res: *BufferRes) Error!Version {
    for (res.spare.items, 0..) |version, i| {
        if (version.busy <= self.completed) return res.spare.swapRemove(i);
    }
    return uploadVersion(self, res.size);
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);
    // `Device` has already checked `caps`, so this is defence in depth: a
    // shape or a format outside what `caps` claims never reaches here.
    if (desc.dimension != .d2) return error.Unsupported;
    const format = textureFormat(desc.format) orelse return error.Unsupported;
    // A chain of levels; not of depth, which nothing here fills.
    const levels = desc.mip_levels;
    if (levels > 1 and desc.format.isDepth()) return error.Unsupported;

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    // A depth texture is made where depth is written, with a depth-stencil
    // view. One that is sampled too - a shadow map - is typeless, written
    // through a view of its depth format and read through one of the format
    // `depthFormats` names; one that is not has a null shader resource view.
    if (desc.format.isDepth()) {
        const readable = desc.usage.sampled;
        const formats = depthFormats(desc.format);
        var rdesc = rc.ResourceDesc.texture2d(desc.width, desc.height, if (readable) formats.resource else format, .{ .allow_depth_stencil = true, .deny_shader_resource = !readable });
        rdesc.sample = .{ .count = desc.samples };
        const clear: rc.ClearValue = .depthStencil(format, 1, 0);
        const writing: rc.ResourceStates = .{ .depth_write = true };
        const obj = rc.createCommittedResource(self.device, .of(.default), rc.heap_flags_none, rdesc, writing, &clear) catch return error.Failed;
        errdefer _ = com.release(obj);
        const dsv = try allocDsv(self);
        errdefer freeDsv(self, dsv);
        const slot = try allocSrv(self);
        const cpu = srvHandle(self, slot);
        rc.createShaderResourceView(self.device, if (readable) obj else null, &.{
            .format = if (readable) formats.read else .r8g8b8a8_unorm,
            .dimension = .texture2d,
            .u = .{ .texture2d = .{ .mip_levels = 1 } },
        }, cpu);
        res.* = .{
            .resource = obj,
            .width = desc.width,
            .height = desc.height,
            .format = desc.format,
            .srv_index = slot,
            .srv_cpu = cpu,
            .dsv_index = dsv,
            .dsv_cpu = dsvHandle(self, dsv),
            .state = writing,
            .samples = desc.samples,
            .readable = readable,
        };
        // A typeless resource's view has to say its format; any other's is
        // the resource's own.
        rc.createDepthStencilView(self.device, obj, if (readable) &.{ .format = format, .dimension = .texture2d } else null, res.dsv_cpu);
        return res;
    }

    // A chain `generateMips` can fill is drawn into a level at a time, by
    // that call if not by a pass.
    const target = desc.usage.render_target;
    const bits = askFormat(self.device, format);
    const drawn = target or (levels > 1 and bits & support_render_target != 0 and bits & support_shader_sample != 0);
    var rdesc = rc.ResourceDesc.texture2d(desc.width, desc.height, format, .{ .allow_render_target = drawn });
    rdesc.sample = .{ .count = desc.samples };
    rdesc.mip_levels = @intCast(levels);
    const clear: rc.ClearValue = .{ .format = format, .color = desc.clear_color };
    const obj = rc.createCommittedResource(self.device, .of(.default), rc.heap_flags_none, rdesc, sampled, if (drawn) &clear else null) catch return error.Failed;
    errdefer _ = com.release(obj);

    // A multisampled texture is drawn into and resolved, never sampled: its
    // shader resource view is the null one an empty slot reads.
    const slot = try allocSrv(self);
    errdefer freeSrv(self, slot);
    const cpu = srvHandle(self, slot);
    rc.createShaderResourceView(self.device, if (desc.samples == 1) obj else null, &.{
        .format = if (desc.samples == 1) format else .r8g8b8a8_unorm,
        .dimension = .texture2d,
        .u = .{ .texture2d = .{ .mip_levels = levels } },
    }, cpu);

    res.* = .{ .resource = obj, .width = desc.width, .height = desc.height, .format = desc.format, .srv_index = slot, .srv_cpu = cpu, .samples = desc.samples, .mip_levels = levels };
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

/// Its descriptors are free at once - a list copies a texture's into the
/// ring, and a target's, as it records - and the texture itself once the
/// GPU is done with it.
fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    freeSrv(self, res.srv_index);
    if (res.rtv_index) |rtv| freeRtv(self, rtv);
    if (res.dsv_index) |dsv| freeDsv(self, dsv);
    bury(self, .{ .resource = res.resource });
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

/// Any box of any level, through a one-off staging buffer laid out the way
/// a copy wants its rows padded, and `CopyTextureRegion` onto the texture.
/// Records into a slot of its own, as `submit` does, without waiting:
/// writing happens between submits, never while one is being recorded.
fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    _ = slice_pitch;
    const self = cast(impl);
    const res = as(TextureRes, native);
    const footprint = footprintOf(res, region.width, region.height);
    const row_bytes = res.format.rowBytes(region.width);

    const staging = try transferBuffer(self, .upload, @as(u64, footprint.footprint.row_pitch) * region.height);
    var buried = false;
    defer if (!buried) {
        _ = com.release(staging);
    };
    var mapped: ?*anyopaque = null;
    staging.vtable.Map(staging, 0, &rc.Range.nothing_read, &mapped).check() catch return error.Failed;
    const dst: [*]u8 = @ptrCast(mapped.?);
    for (0..region.height) |y| {
        @memcpy(dst[y * footprint.footprint.row_pitch ..][0..row_bytes], bytes[y * row_pitch ..][0..row_bytes]);
    }
    staging.vtable.Unmap(staging, 0, null);

    if (!self.recording) try beginRecording(self);
    errdefer abandonRecording(self);
    transition(self, res, .{ .copy_dest = true });
    const dst_loc = cmdmod.TextureCopyLocation.subresource(res.resource, region.mip);
    const src_loc = cmdmod.TextureCopyLocation.placed(staging, footprint);
    self.list.vtable.CopyTextureRegion(self.list, &dst_loc, region.x, region.y, 0, &src_loc, null);
    transition(self, res, sampled);
    // Read by the copy when the list runs: released once it has.
    bury(self, .{ .resource = staging });
    buried = true;
}

/// One level, copied into a readback buffer and handed back as RGBA, eight
/// bits a channel, top row first - as Direct3D keeps a texture - with one
/// channel repeated into red, green and blue as the Direct3D 11 backend does.
fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (readback.decodeOf(res.format) == null) return error.Unsupported;
    const level = res.levelSize(sub.mip);
    const footprint = footprintOf(res, level[0], level[1]);
    const size = @as(u64, footprint.footprint.row_pitch) * level[1];

    const landing = try transferBuffer(self, .readback, size);
    defer _ = com.release(landing);

    if (!self.recording) try beginRecording(self);
    errdefer abandonRecording(self);
    transition(self, res, .{ .copy_source = true });
    const dst_loc = cmdmod.TextureCopyLocation.placed(landing, footprint);
    const src_loc = cmdmod.TextureCopyLocation.subresource(res.resource, sub.mip);
    self.list.vtable.CopyTextureRegion(self.list, &dst_loc, 0, 0, 0, &src_loc, null);
    transition(self, res, sampled);
    try closeAndExecute(self);
    // The one wait a readback has: the bytes are wanted now.
    try waitFor(self, self.fence_value);

    var mapped: ?*anyopaque = null;
    const everything: rc.Range = .{ .begin = 0, .end = @intCast(size) };
    landing.vtable.Map(landing, 0, &everything, &mapped).check() catch return error.Failed;
    defer landing.vtable.Unmap(landing, 0, &rc.Range.nothing_read);
    const data: [*]const u8 = @ptrCast(mapped.?);

    const pixels = try gpa.alloc(u8, @as(usize, level[0]) * 4 * level[1]);
    readback.convert(res.format, readback.decodeOf(res.format).?, data, footprint.footprint.row_pitch, level[0], level[1], pixels);
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

    if (desc.extra_color_formats.len > 0) {
        log.writeAll("fluxion-rhi: the Direct3D 12 backend has no extra colour attachments yet") catch {};
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
    const color_format: ?rc.Format = if (desc.color_format) |format| textureFormat(format) orelse return error.Unsupported else null;
    const depth_format: rc.Format = if (desc.depth_format) |format| textureFormat(format) orelse return error.Unsupported else .unknown;

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
        .depth_bias = desc.depth.bias,
        .slope_scaled_depth_bias = desc.depth.slope_bias,
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
    if (color_format) |format| rtv_formats[0] = format;

    const pso_desc: pl.GraphicsPipelineStateDesc = .{
        .root_signature = self.root_signature,
        .vs = .of(shader_res.vertex),
        .ps = .of(shader_res.pixel),
        .blend_state = blend,
        .rasterizer_state = rasterizer,
        .depth_stencil_state = .{
            .depth_enable = if (desc.depth.test_enabled) 1 else 0,
            .depth_write_mask = if (desc.depth.write) .all else .zero,
            .depth_func = comparison(desc.depth.compare),
        },
        .input_layout = .{
            .elements = if (desc.attributes.len == 0) null else &elements,
            .count = @intCast(desc.attributes.len),
        },
        .primitive_topology_type = topology_type,
        .num_render_targets = if (color_format != null) 1 else 0,
        .rtv_formats = rtv_formats,
        .dsv_format = depth_format,
        .sample = .{ .count = desc.samples },
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
    bury(self, .{ .pso = res.pso });
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

    const flags: u32 = if (dxgi.allowsTearing(self.factory)) swap_chain_allow_tearing else 0;
    const chain_desc: SwapChainDesc1 = .{ .width = desc.width, .height = desc.height, .flags = flags };
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
        .flags = flags,
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
    res.swap_chain.vtable.base.ResizeBuffers(@ptrCast(res.swap_chain), 0, width, height, .unknown, res.flags).check() catch return error.Failed;
    try attachBackBuffers(self, res);
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

/// See d3d11's: the same sync intervals, the same flip model, the same
/// tearing for `disabled` where the system allows it.
fn present(impl: backend.Impl, native: backend.Native, mode: types.PresentMode) Error!void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    // What was drawn into it is recorded, and runs before the flip.
    try flush(self);
    const interval: u32 = if (mode.waits()) 1 else 0;
    const flags: u32 = if (mode == .disabled and res.flags & swap_chain_allow_tearing != 0) present_allow_tearing else 0;
    res.swap_chain.vtable.base.Present(@ptrCast(res.swap_chain), interval, flags).check() catch |err| switch (err) {
        error.DeviceRemoved, error.DeviceReset => return error.DeviceLost,
        else => return error.Failed,
    };
}

// -------------------------------------------------------------------------
// Submitting
// -------------------------------------------------------------------------

fn submit(impl: backend.Impl, device: *Device, list_cmds: []const commands.Command) Error!void {
    const self = cast(impl);
    self.used_buffers.clearRetainingCapacity();
    if (!self.recording) try beginRecording(self);
    errdefer abandonRecording(self);

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

    const cmd_list = self.list;
    for (list_cmds) |command| {
        switch (command) {
            .begin_pass => |pass| {
                if (pass.extra_colors.len > 0) return error.Unsupported;
                var target: Target = .{ .rtv = null, .width = 0, .height = 0, .resource = null, .texture = null };
                if (pass.color) |color| switch (color.target) {
                    .surface => |h| {
                        const surf = as(SurfaceRes, device.surfaces.get(h).?.native);
                        const idx = surf.swap_chain.vtable.GetCurrentBackBufferIndex(surf.swap_chain);
                        const to_rt = cmdmod.ResourceBarrier.transition(surf.back_buffers[idx], .{}, .{ .render_target = true });
                        cmd_list.vtable.ResourceBarrier(cmd_list, 1, &[_]cmdmod.ResourceBarrier{to_rt});
                        target = .{ .rtv = surf.rtv_handles[idx], .width = surf.width, .height = surf.height, .resource = surf.back_buffers[idx], .texture = null };
                    },
                    .texture => |h| {
                        const res = as(TextureRes, device.textures.get(h).?.native);
                        if (res.rtv_index == null) return error.Unsupported;
                        transition(self, res, .{ .render_target = true });
                        // A level below the first through a view of its own.
                        const level = res.levelSize(color.mip_level);
                        const rtv = if (color.mip_level == 0) res.rtv_cpu else blk: {
                            rc.createRenderTargetView(self.device, res.resource, &.{
                                .format = textureFormat(res.format).?,
                                .dimension = .texture2d,
                                .u = .{ .texture2d = .{ .mip_slice = color.mip_level } },
                            }, self.pass_rtv);
                            break :blk self.pass_rtv;
                        };
                        target = .{ .rtv = rtv, .width = level[0], .height = level[1], .resource = res.resource, .texture = res };
                    },
                };
                if (pass.color) |color| if (color.resolve) |into| {
                    if (target.texture == null) return error.Unsupported;
                    target.resolve = switch (into) {
                        .surface => |h| blk: {
                            const surf = as(SurfaceRes, device.surfaces.get(h).?.native);
                            const idx = surf.swap_chain.vtable.GetCurrentBackBufferIndex(surf.swap_chain);
                            break :blk .{ .texture = null, .resource = surf.back_buffers[idx], .format = surface_dxgi_format };
                        },
                        .texture => |h| blk: {
                            const res = as(TextureRes, device.textures.get(h).?.native);
                            break :blk .{ .texture = res, .resource = res.resource, .format = textureFormat(res.format).? };
                        },
                    };
                };
                if (pass.depth) |depth| {
                    const res = as(TextureRes, device.textures.get(depth.texture).?.native);
                    const dsv = res.dsv_index orelse return error.Unsupported;
                    transition(self, res, .{ .depth_write = true });
                    target.dsv = dsvHandle(self, dsv);
                    target.depth = res;
                    if (pass.color == null) {
                        target.width = res.width;
                        target.height = res.height;
                    }
                }
                self.target = target;
                target.bind(cmd_list);
                if (pass.color) |color| if (color.load == .clear) cmd_list.vtable.ClearRenderTargetView(cmd_list, target.rtv.?, &color.clear_color, 0, null);
                if (pass.depth) |depth| if (depth.load == .clear) {
                    const res = as(TextureRes, device.textures.get(depth.texture).?.native);
                    const flags: u32 = if (res.format.hasStencil()) 0x3 else 0x1;
                    cmd_list.vtable.ClearDepthStencilView(cmd_list, target.dsv.?, flags, depth.clear_depth, depth.clear_stencil, 0, null);
                };

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
                self.vertex_bindings[b.slot] = .{ .resource = try use(self, res), .offset = b.offset, .size = res.size };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                const resource = try use(self, res);
                self.index_view = .{
                    .buffer_location = resource.vtable.GetGPUVirtualAddress(resource),
                    .size_in_bytes = res.size,
                    .format = if (b.format == .u16) .r16_uint else .r32_uint,
                };
                cmd_list.vtable.IASetIndexBuffer(cmd_list, &self.index_view.?);
            },
            .set_uniform_buffer => |b| {
                if (b.slot >= uniform_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                const resource = try use(self, res);
                self.uniforms[b.slot] = resource.vtable.GetGPUVirtualAddress(resource) + b.offset;
                cmd_list.vtable.SetGraphicsRootConstantBufferView(cmd_list, root_param_cbv0 + b.slot, self.uniforms[b.slot]);
            },
            .set_texture => |b| {
                if (b.slot >= texture_slots) return error.Unsupported;
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
            .generate_mips => |h| try generateMips(self, as(TextureRes, device.textures.get(h).?.native)),
        }
    }

    self.lists += 1;
    if (self.lists >= max_lists) try closeAndExecute(self);
}

/// Every level below the first, each drawn from the one above it: a
/// triangle over the level, reading the level above with a linear filter at
/// the middle of each pixel - the average of the four texels under it. The
/// level being drawn is a render target and every other stays sampled,
/// barrier by barrier. Outside any pass; the pipeline and the textures the
/// submit had bound are bound again before its next draw.
fn generateMips(self: *D3d, res: *TextureRes) Error!void {
    const pso = try mipPipelineOf(self, res.format);
    const list = self.list;
    const format = textureFormat(res.format).?;
    transition(self, res, sampled);
    list.vtable.SetPipelineState(list, pso);
    list.vtable.IASetPrimitiveTopology(list, .triangle_list);
    const textures = self.textures;
    const samplers = self.samplers;
    for (1..res.mip_levels) |level| {
        const mip: u32 = @intCast(level);
        const size = res.levelSize(mip);
        rc.createShaderResourceView(self.device, res.resource, &.{
            .format = format,
            .dimension = .texture2d,
            .u = .{ .texture2d = .{ .most_detailed_mip = mip - 1, .mip_levels = 1 } },
        }, self.scratch_srv);
        rc.createRenderTargetView(self.device, res.resource, &.{
            .format = format,
            .dimension = .texture2d,
            .u = .{ .texture2d = .{ .mip_slice = mip } },
        }, self.mip_rtv);
        levelBarrier(self, res, mip, sampled, .{ .render_target = true });
        list.vtable.OMSetRenderTargets(list, 1, @ptrCast(&self.mip_rtv), 0, null);
        list.vtable.RSSetViewports(list, 1, &[_]cmdmod.Viewport{.{ .width = @floatFromInt(size[0]), .height = @floatFromInt(size[1]) }});
        list.vtable.RSSetScissorRects(list, 1, &[_]cmdmod.Rect{.{ .left = 0, .top = 0, .right = @intCast(size[0]), .bottom = @intCast(size[1]) }});
        self.textures[0] = self.scratch_srv;
        self.samplers[0] = self.mip_sampler;
        self.textures_dirty = true;
        self.samplers_dirty = true;
        // The ring takes its copy of the view now, so the next level can
        // write the same slot.
        try prepareDraw(self);
        list.vtable.DrawInstanced(list, 3, 1, 0, 0);
        levelBarrier(self, res, mip, .{ .render_target = true }, sampled);
    }
    self.textures = textures;
    self.samplers = samplers;
    self.textures_dirty = true;
    self.samplers_dirty = true;
    if (self.current_pipeline) |p| {
        list.vtable.SetPipelineState(list, p.pso);
        list.vtable.IASetPrimitiveTopology(list, p.topology);
    }
}

/// One level of a texture from one state to another; the texture as a whole
/// is kept in `res.state`, which every level is back in afterwards.
fn levelBarrier(self: *D3d, res: *TextureRes, mip: u32, before: rc.ResourceStates, after: rc.ResourceStates) void {
    const barrier: cmdmod.ResourceBarrier = .{
        .type = .transition,
        .u = .{ .transition = .{ .resource = res.resource, .subresource = mip, .state_before = before, .state_after = after } },
    };
    self.list.vtable.ResourceBarrier(self.list, 1, &[_]cmdmod.ResourceBarrier{barrier});
}

/// The pipeline that draws a level of `format` from the one above: its
/// shader compiled the first time, the pipeline made once for each format.
fn mipPipelineOf(self: *D3d, format: types.Format) Error!*pl.ID3D12PipelineState {
    if (self.mip_pipelines.get(format)) |held| return held;
    if (self.mip_shader == null) {
        var discard: Io.Writer.Discarding = .init(&.{});
        const compiler = try compilerOf(self, &discard.writer);
        const vs_blob = try compileStage(compiler, mip_vertex, "vs_5_0", &discard.writer);
        defer _ = com.release(vs_blob);
        const ps_blob = try compileStage(compiler, mip_pixel, "ps_5_0", &discard.writer);
        defer _ = com.release(ps_blob);
        const vertex = try self.gpa.dupe(u8, vs_blob.bytes());
        errdefer self.gpa.free(vertex);
        self.mip_shader = .{ .vertex = vertex, .pixel = try self.gpa.dupe(u8, ps_blob.bytes()) };
    }
    const shader = self.mip_shader.?;
    var blend: pl.BlendDesc = .{};
    blend.render_target[0] = .{ .render_target_write_mask = pl.color_write_all };
    var rtv_formats: [8]rc.Format = @splat(.unknown);
    rtv_formats[0] = textureFormat(format).?;
    const pso = pl.createGraphicsPipelineState(self.device, &.{
        .root_signature = self.root_signature,
        .vs = .of(shader.vertex),
        .ps = .of(shader.pixel),
        .blend_state = blend,
        .rasterizer_state = .{ .cull_mode = .none },
        .depth_stencil_state = .{ .depth_enable = 0, .depth_write_mask = .zero },
        .input_layout = .{ .elements = null, .count = 0 },
        .primitive_topology_type = .triangle,
        .num_render_targets = 1,
        .rtv_formats = rtv_formats,
        .dsv_format = .unknown,
        .sample = .{ .count = 1 },
    }) catch return error.Failed;
    self.mip_pipelines.set(format, pso);
    return pso;
}

/// What every recording starts with: the one root signature, and the rings
/// its tables point into.
fn setUpList(self: *D3d) void {
    self.list.vtable.SetGraphicsRootSignature(self.list, self.root_signature);
    const heaps = [_]*rc.ID3D12DescriptorHeap{ self.ring_srv_heap, self.ring_sampler_heap };
    self.list.vtable.SetDescriptorHeaps(self.list, heaps.len, &heaps);
}

/// End the pass that is open: its samples averaged into where it resolves,
/// and its texture back to being sampled, or its back buffer back to being
/// presented.
fn endTarget(self: *D3d) void {
    const target = self.target orelse return;
    self.target = null;
    if (target.resolve) |into| resolve(self, target.texture.?, into);
    if (target.depth) |res| if (res.readable) transition(self, res, sampled);
    if (target.texture) |res| {
        transition(self, res, sampled);
    } else if (target.resource) |back_buffer| {
        const to_present = cmdmod.ResourceBarrier.transition(back_buffer, .{ .render_target = true }, .{});
        self.list.vtable.ResourceBarrier(self.list, 1, &[_]cmdmod.ResourceBarrier{to_present});
    }
}

/// A multisampled texture's samples averaged into `into`, which is left as
/// it was found: a texture sampled, a back buffer presented.
fn resolve(self: *D3d, source: *TextureRes, into: Resolve) void {
    const list = self.list;
    transition(self, source, .{ .resolve_source = true });
    if (into.texture) |res| {
        transition(self, res, .{ .resolve_dest = true });
    } else {
        const to_dest = cmdmod.ResourceBarrier.transition(into.resource, .{}, .{ .resolve_dest = true });
        list.vtable.ResourceBarrier(list, 1, &[_]cmdmod.ResourceBarrier{to_dest});
    }
    list.vtable.ResolveSubresource(list, into.resource, 0, source.resource, 0, into.format);
    if (into.texture) |res| {
        transition(self, res, sampled);
    } else {
        const to_present = cmdmod.ResourceBarrier.transition(into.resource, .{ .resolve_dest = true }, .{});
        list.vtable.ResourceBarrier(list, 1, &[_]cmdmod.ResourceBarrier{to_present});
    }
}

/// The tables a draw reads, made where its textures or samplers changed, and
/// its vertex buffers bound.
fn prepareDraw(self: *D3d) Error!void {
    const out_of_srvs = self.textures_dirty and self.ring_srv_used + texture_slots > slot_srv_descriptors;
    const out_of_samplers = self.samplers_dirty and self.ring_sampler_used + texture_slots > slot_sampler_descriptors;
    if (out_of_srvs or out_of_samplers) try restartRecording(self);

    if (self.textures_dirty) {
        const at: u32 = @as(u32, @intCast(self.at)) * slot_srv_descriptors + self.ring_srv_used;
        self.ring_srv_used += texture_slots;
        const start = rc.cpuHeapStart(self.ring_srv_heap);
        for (self.textures, 0..) |source, i| {
            rc.copyDescriptorsSimple(self.device, 1, start.offsetBy(at + @as(u32, @intCast(i)), self.cbv_srv_uav_increment), source, .cbv_srv_uav);
        }
        self.list.vtable.SetGraphicsRootDescriptorTable(self.list, root_param_srv_table, rc.gpuHeapStart(self.ring_srv_heap).offsetBy(at, self.cbv_srv_uav_increment));
        self.textures_dirty = false;
    }
    if (self.samplers_dirty) {
        const at: u32 = @as(u32, @intCast(self.at)) * slot_sampler_descriptors + self.ring_sampler_used;
        self.ring_sampler_used += texture_slots;
        const start = rc.cpuHeapStart(self.ring_sampler_heap);
        for (self.samplers, 0..) |source, i| {
            rc.copyDescriptorsSimple(self.device, 1, start.offsetBy(at + @as(u32, @intCast(i)), self.sampler_increment), source, .sampler);
        }
        self.list.vtable.SetGraphicsRootDescriptorTable(self.list, root_param_sampler_table, rc.gpuHeapStart(self.ring_sampler_heap).offsetBy(at, self.sampler_increment));
        self.samplers_dirty = false;
    }
    flushBindings(self);
}

/// Execute what has been recorded and record on from the same state in the
/// next slot, with its part of the rings: what a submit that draws with more
/// changes of texture than a part holds does. The target stays where the
/// pass put it; nothing is cleared again.
fn restartRecording(self: *D3d) Error!void {
    try closeAndExecute(self);
    try beginRecording(self);
    // What this list bound is bound again below, in the next list.
    for (self.used_buffers.items) |res| res.current.busy = self.fence_value + 1;

    const cmd_list = self.list;
    if (self.target) |target| target.bind(cmd_list);
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

test "caps: .d2 only; the driver's formats, floats too, sampled, drawn into and multisampled" {
    var device = try warpDevice();
    defer device.deinit();

    const c = device.caps();
    for ([_]types.Format{ .rgba8_unorm, .bgra8_unorm, .r8_unorm, .rgba16_float, .rg11b10_float, .rgba32_float }) |format| {
        const support = c.formatSupport(format);
        try testing.expect(support.sampled);
        try testing.expect(support.render_target);
        try testing.expect(support.dimensions.contains(.d2));
        try testing.expect(!support.dimensions.contains(.cube));
        try testing.expect(!support.dimensions.contains(.d3));
        try testing.expect(support.supportsSamples(1));
    }
    // What a renderer draws light into before it is toned down: filtered,
    // blended and multisampled.
    const hdr = c.formatSupport(.rgba16_float);
    try testing.expect(hdr.filterable and hdr.blendable and hdr.supportsSamples(4));
    try testing.expect(c.formatSupport(.depth32_float).supportsSamples(4));
    try testing.expectEqual(@as(u32, 1), c.limits.max_anisotropy);
    try testing.expect(c.features.sampler_border);

    // A depth format is drawn into and not sampled; a compressed one is
    // never claimed.
    try testing.expect(c.formatSupport(.depth32_float).render_target);
    // ... and read, as a shadow map, filtered by a sampler that compares.
    try testing.expect(c.formatSupport(.depth32_float).sampled);
    try testing.expect(c.formatSupport(.depth32_float).filterable);
    try testing.expect(!c.formatSupport(.bc1_rgba_unorm).sampled);
}

test "a buffer is created, updated, and its bytes are what was written" {
    var device = try warpDevice();
    defer device.deinit();

    const buffer = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 1, 2, 3, 4 }) });
    try device.updateBuffer(buffer, 0, std.mem.asBytes(&[4]f32{ 5, 6, 7, 8 }));
    device.destroyBuffer(buffer);
}

test "a buffer the GPU may still read is written into another copy, which keeps the rest of its bytes" {
    var device = try warpDevice();
    defer device.deinit();
    const self = cast(device.impl);

    const handle = try device.createBuffer(.{ .kind = .vertex, .size = 8, .data = "abcdefgh" });
    const res = as(BufferRes, device.buffers.get(handle).?.native);
    const first = res.current;
    // As if a list the GPU has not done yet drew with it.
    res.current.busy = self.fence_value + 1;
    try device.updateBuffer(handle, 2, "XY");
    try testing.expect(res.current.resource != first.resource);
    try testing.expectEqualSlices(u8, "abcdefgh", first.mapped[0..8]);
    try testing.expectEqualSlices(u8, "abXYefgh", res.current.mapped[0..8]);

    // One no list uses is written where it is.
    const second = res.current.resource;
    try device.updateBuffer(handle, 0, "Z");
    try testing.expectEqual(second, res.current.resource);
    try testing.expectEqualSlices(u8, "ZbXYefgh", res.current.mapped[0..8]);
}

test "submits go into one list, executed when needed, and what they used is released once the GPU is done" {
    var device = try warpDevice();
    defer device.deinit();
    const self = cast(device.impl);

    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true }, .clear_color = .{ 0, 1, 0, 1 } });
    try flush(self);
    const before = self.fence_value;
    for (0..12) |_| {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 1, 0, 1 } } });
        try cmd.endPass();
        try device.submit();
    }
    // Twelve submits, and nothing executed yet.
    try testing.expectEqual(before, self.fence_value);
    try testing.expect(self.recording);
    const doomed = try device.createTexture(.{ .width = 2, .height = 2 });
    const graves = self.graveyard.items.len;
    device.destroyTexture(doomed);
    try testing.expectEqual(graves + 1, self.graveyard.items.len);
    try flush(self);
    try testing.expectEqual(before + 1, self.fence_value);
    waitForGpuIdle(self);
    try testing.expectEqual(@as(usize, 0), self.graveyard.items.len);

    const back = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(back);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, back[0..4].*);
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
    const draws = ring_sampler_descriptors / texture_slots + 40;
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
    // No compressed formats.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 4, .height = 4, .format = .bc1_rgba_unorm }));
}

test "a depth texture drawn into is read by a sampler that compares, and a bias pushes what is drawn back" {
    var device = try warpDevice();
    defer device.deinit();

    // A plane at depth one half, drawn into two shadow maps: once as it is,
    // and once pushed back by 2^18 of a 32-bit float's steps at one half,
    // 2^-24 each - a sixty-fourth.
    const nothing =
        \\float4 main() : SV_TARGET { return 0; }
    ;
    const flat = try device.createShader(.{ .hlsl = .{ .vertex = Flat.vs, .fragment = nothing } });
    const plane = try device.createBuffer(.{ .kind = .vertex, .size = 72, .data = std.mem.sliceAsBytes(&[_]f32{
        -1, -1, 0.5, 1, -1, 0.5, -1, 1, 0.5,
        -1, 1,  0.5, 1, -1, 0.5, 1,  1, 0.5,
    }) });
    var maps: [2]types.Texture = undefined;
    for (&maps, [_]i32{ 0, 1 << 18 }) |*map, bias| {
        const pipeline = try device.createPipeline(.{
            .shader = flat,
            .attributes = &.{.{ .location = 0, .format = .float3, .offset = 0 }},
            .buffers = &.{.{ .stride = 12 }},
            .color_format = null,
            .depth_format = .depth32_float,
            .depth = .{ .test_enabled = true, .write = true, .compare = .less, .bias = bias },
        });
        map.* = try device.createTexture(.{ .width = 4, .height = 4, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
        const cmd = device.begin();
        try cmd.beginPass(.{ .depth = .{ .texture = map.* } });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, plane, 0);
        try cmd.draw(.{ .vertex_count = 6 });
        try cmd.endPass();
        try device.submit();
    }

    // Read through a sampler that compares: lit where the depth asked about
    // is at or before what is stored.
    const comparing =
        \\cbuffer Params : register(b0) { float4 param; };
        \\Texture2D<float> map : register(t0);
        \\SamplerComparisonState map_sampler : register(s0);
        \\float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET {
        \\    float v = map.SampleCmpLevelZero(map_sampler, uv, param.x);
        \\    return float4(v, v, v, 1);
        \\}
    ;
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = Quad.vs, .fragment = comparing } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
    });
    const quad = try Quad.init(&device);
    const compare = try device.createSampler(.{ .compare = .less_equal });
    const params = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    const out = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .sampled = true, .render_target = true } });

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
        try cmd.setVertexBuffer(0, quad.corners, 0);
        try cmd.setUniformBuffer(0, params);
        try cmd.setTexture(0, maps[case.map], compare);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        const pixels = try device.readTexture(out, testing.allocator);
        defer testing.allocator.free(pixels);
        const want: [4]u8 = if (case.lit) .{ 255, 255, 255, 255 } else .{ 0, 0, 0, 255 };
        try testing.expectEqual(want, texelAt(pixels, 4, 1, 2));
    }
}

/// A triangle in one colour, at `samples` a pixel, with a depth test where
/// `depth` is given.
const Flat = struct {
    shader: types.Shader,
    pipeline: types.Pipeline,

    const vs =
        \\float4 main(float3 position : ATTR0) : SV_POSITION { return float4(position, 1); }
    ;
    const ps =
        \\cbuffer Look : register(b0) { float4 colour; };
        \\float4 main() : SV_TARGET { return colour; }
    ;

    fn init(device: *Device, samples: u32, format: types.Format, depth: ?types.Format) !Flat {
        const shader = device.createShader(.{ .hlsl = .{ .vertex = vs, .fragment = ps } }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };
        const pipeline = try device.createPipeline(.{
            .shader = shader,
            .attributes = &.{.{ .location = 0, .format = .float3, .offset = 0 }},
            .buffers = &.{.{ .stride = 12 }},
            .topology = .triangles,
            .samples = samples,
            .color_format = format,
            .depth_format = depth,
            .depth = if (depth != null) .{ .test_enabled = true, .write = true, .compare = .less } else .{},
        });
        return .{ .shader = shader, .pipeline = pipeline };
    }
};

test "a float target keeps light over one, which a pass sampling it halves" {
    var device = try warpDevice();
    defer device.deinit();

    // Cleared to more than white: two in red.
    const bright = try device.createTexture(.{ .width = 4, .height = 4, .format = .rgba16_float, .usage = .{ .sampled = true, .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = bright }, .clear_color = .{ 2, 0.25, 0, 1 } } });
    try cmd.endPass();
    try device.submit();
    const clamped = try device.readTexture(bright, testing.allocator);
    defer testing.allocator.free(clamped);
    try testing.expectEqual([4]u8{ 255, 64, 0, 255 }, texelAt(clamped, 4, 1, 1));

    // Halved into another: one and an eighth, so the red was kept.
    const halving =
        \\Texture2D picture : register(t0);
        \\SamplerState picture_sampler : register(s0);
        \\float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET { return picture.Sample(picture_sampler, uv) * 0.5; }
    ;
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = Quad.vs, .fragment = halving } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
        .color_format = .rgba16_float,
    });
    const quad = try Quad.init(&device);
    const half = try device.createTexture(.{ .width = 4, .height = 4, .format = .rgba16_float, .usage = .{ .sampled = true, .render_target = true } });
    const sampler = try device.createSampler(.nearest);
    const again = device.begin();
    try again.beginPass(.{ .color = .{ .target = .{ .texture = half } } });
    try again.setPipeline(pipeline);
    try again.setVertexBuffer(0, quad.corners, 0);
    try again.setTexture(0, bright, sampler);
    try again.draw(.{ .vertex_count = 4 });
    try again.endPass();
    try device.submit();
    const halved = try device.readTexture(half, testing.allocator);
    defer testing.allocator.free(halved);
    try testing.expectEqual([4]u8{ 255, 32, 0, 128 }, texelAt(halved, 4, 2, 2));
}

test "a multisampled target, with multisampled depth, is resolved into a texture at the end of the pass" {
    var device = try warpDevice();
    defer device.deinit();

    const flat = try Flat.init(&device, 4, .rgba16_float, .depth32_float);
    const msaa = try device.createTexture(.{ .width = 32, .height = 32, .format = .rgba16_float, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const depth = try device.createTexture(.{ .width = 32, .height = 32, .format = .depth32_float, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const resolved = try device.createTexture(.{ .width = 32, .height = 32, .format = .rgba16_float, .usage = .{ .sampled = true, .render_target = true } });

    // A red triangle over blue, with slanted sides, so that some pixels are
    // only partly covered.
    const triangle = [_][3]f32{ .{ -0.8, -0.8, 0.5 }, .{ 0, 0.8, 0.5 }, .{ 0.8, -0.8, 0.5 } };
    const corners = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(triangle)), .data = std.mem.asBytes(&triangle) });
    const red = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 1, 0, 0, 1 }) });
    const cmd = device.begin();
    try cmd.beginPass(.{
        .color = .{ .target = .{ .texture = msaa }, .clear_color = .{ 0, 0, 1, 1 }, .resolve = .{ .texture = resolved } },
        .depth = .{ .texture = depth },
    });
    try cmd.setPipeline(flat.pipeline);
    try cmd.setUniformBuffer(0, red);
    try cmd.setVertexBuffer(0, corners, 0);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(resolved, testing.allocator);
    defer testing.allocator.free(pixels);
    // Inside is red, outside blue, and the edge neither: its samples averaged.
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(pixels, 32, 16, 20));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, texelAt(pixels, 32, 1, 1));
    var blended: usize = 0;
    for (0..32 * 32) |i| {
        const p = pixels[i * 4 ..][0..4];
        if (p[0] > 0 and p[0] < 255 and p[2] > 0 and p[2] < 255) blended += 1;
    }
    try testing.expect(blended > 10);
    // The multisampled texture itself is not read.
    try testing.expectError(error.InvalidArgument, device.readTexture(msaa, testing.allocator));
}

test "the last slot of each kind is bound: the eighth uniform buffer and the sixteenth texture" {
    var device = try warpDevice();
    defer device.deinit();

    const last =
        \\cbuffer Look : register(b7) { float4 tint; };
        \\Texture2D picture : register(t15);
        \\SamplerState picture_sampler : register(s15);
        \\float4 main(float4 position : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET { return picture.Sample(picture_sampler, uv) * tint; }
    ;
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = Quad.vs, .fragment = last } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
        .buffers = &.{.{ .stride = 8 }},
        .topology = .triangle_strip,
    });
    const quad = try Quad.init(&device);
    const white = try device.createTexture(.{ .width = 1, .height = 1, .data = &.{ 255, 255, 255, 255 } });
    const green = try device.createBuffer(.{ .kind = .uniform, .size = 16, .data = std.mem.asBytes(&[4]f32{ 0, 1, 0, 1 }) });
    const target = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    const sampler = try device.createSampler(.nearest);
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, quad.corners, 0);
    try cmd.setUniformBuffer(7, green);
    try cmd.setTexture(15, white, sampler);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();
    const drawn = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(drawn);
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, texelAt(drawn, 4, 2, 2));
}

test "two draws read two parts of one uniform buffer" {
    var device = try warpDevice();
    defer device.deinit();
    const flat = try Flat.init(&device, 1, .rgba8_unorm, null);
    // The left half, then the right half.
    const halves = [_][3]f32{
        .{ -1, -1, 0 }, .{ -1, 1, 0 }, .{ 0, -1, 0 }, .{ 0, -1, 0 }, .{ -1, 1, 0 }, .{ 0, 1, 0 },
        .{ 0, -1, 0 },  .{ 0, 1, 0 },  .{ 1, -1, 0 }, .{ 1, -1, 0 }, .{ 0, 1, 0 },  .{ 1, 1, 0 },
    };
    const corners = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(halves)), .data = std.mem.asBytes(&halves) });
    const step = device.caps().limits.uniform_offset_alignment;
    const colours = try device.createBuffer(.{ .kind = .uniform, .size = step + 16 });
    try device.updateBuffer(colours, 0, std.mem.asBytes(&[4]f32{ 1, 0, 0, 1 }));
    try device.updateBuffer(colours, step, std.mem.asBytes(&[4]f32{ 0, 0, 1, 1 }));
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(flat.pipeline);
    try cmd.setVertexBuffer(0, corners, 0);
    try cmd.setUniformBufferRange(0, colours, 0, 16);
    try cmd.draw(.{ .vertex_count = 6 });
    try cmd.setUniformBufferRange(0, colours, step, 0);
    try cmd.draw(.{ .vertex_count = 6, .first_vertex = 6 });
    try cmd.endPass();
    try device.submit();
    const drawn = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(drawn);
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(drawn, 8, 1, 4));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, texelAt(drawn, 8, 6, 4));
}

test "a depth texture keeps what is nearer, whichever is drawn last, and a pass can write depth alone" {
    var device = try warpDevice();
    defer device.deinit();

    const vs =
        \\struct In { float3 position : ATTR0; float4 color : ATTR1; };
        \\struct Out { float4 position : SV_POSITION; float4 color : COLOR0; };
        \\Out main(In i) { Out o; o.position = float4(i.position, 1); o.color = i.color; return o; }
    ;
    const ps =
        \\struct In { float4 position : SV_POSITION; float4 color : COLOR0; };
        \\float4 main(In i) : SV_TARGET { return i.color; }
    ;
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = vs, .fragment = ps } });
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
    const depth = try device.createTexture(.{ .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } });
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
        try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, texelAt(pixels, 8, 4, 4));
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
    try testing.expectEqual([4]u8{ 0, 255, 0, 255 }, texelAt(pixels, 8, 4, 4));
}

const level_palette = [_][4]u8{
    .{ 255, 0, 0, 255 },
    .{ 0, 255, 0, 255 },
    .{ 0, 0, 255, 255 },
    .{ 255, 255, 0, 255 },
};

fn solidLevel(buffer: []u8, colour: [4]u8) []u8 {
    for (0..buffer.len / 4) |i| buffer[i * 4 ..][0..4].* = colour;
    return buffer;
}

fn expectSolidLevel(pixels: []const u8, colour: [4]u8) !void {
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) try testing.expectEqualSlices(u8, &colour, pixels[i..][0..4]);
}

fn expectNearTexel(expected: [4]u8, actual: [4]u8, tolerance: u8) !void {
    for (expected, actual) |e, a| {
        const distance = if (e > a) e - a else a - e;
        if (distance > tolerance) {
            std.debug.print("expected {any}, found {any}\n", .{ expected, actual });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

test "each level of a chain is written and read on its own" {
    var device = try warpDevice();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0 });
    try testing.expectEqual(@as(u32, 4), (try device.textureInfo(texture)).mip_levels);
    var buffer: [8 * 8 * 4]u8 = undefined;
    for (0..4) |mip| {
        const size: usize = types.mipExtent(8, @intCast(mip));
        try device.writeTexture(texture, .{ .mip = @intCast(mip) }, solidLevel(buffer[0 .. size * size * 4], level_palette[mip]), 0, 0);
    }
    for (0..4) |mip| {
        const size: usize = types.mipExtent(8, @intCast(mip));
        const pixels = try device.readSubresource(texture, .{ .mip = @intCast(mip) }, testing.allocator);
        defer testing.allocator.free(pixels);
        try testing.expectEqual(size * size * 4, pixels.len);
        try expectSolidLevel(pixels, level_palette[mip]);
    }

    // A box in level one, and the rest of that level, and the others, keep theirs.
    try device.writeTexture(texture, .{ .mip = 1, .x = 1, .y = 1, .width = 2, .height = 2 }, solidLevel(buffer[0..16], level_palette[3]), 0, 0);
    const one = try device.readSubresource(texture, .{ .mip = 1 }, testing.allocator);
    defer testing.allocator.free(one);
    try testing.expectEqualSlices(u8, &level_palette[3], one[(1 * 4 + 1) * 4 ..][0..4]);
    try testing.expectEqualSlices(u8, &level_palette[3], one[(2 * 4 + 2) * 4 ..][0..4]);
    try testing.expectEqualSlices(u8, &level_palette[1], one[0..4]);
    const zero = try device.readSubresource(texture, .{}, testing.allocator);
    defer testing.allocator.free(zero);
    try expectSolidLevel(zero, level_palette[0]);
}

test "generateMips draws each level from the one above" {
    var device = try warpDevice();
    defer device.deinit();
    try testing.expect(device.caps().formatSupport(.rgba8_unorm).generate_mips);

    // A 4x4 checkerboard of black and white: every level below is grey.
    var checker: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| checker[i * 4 ..][0..4].* = if ((i % 4 + i / 4) % 2 == 0) .{ 255, 255, 255, 255 } else .{ 0, 0, 0, 255 };
    // Only sampled, and filled all the same.
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0, .data = &checker });

    const cmd = device.begin();
    try cmd.generateMips(texture);
    try device.submit();

    const zero = try device.readSubresource(texture, .{}, testing.allocator);
    defer testing.allocator.free(zero);
    try testing.expectEqualSlices(u8, &checker, zero);
    for (1..3) |mip| {
        const pixels = try device.readSubresource(texture, .{ .mip = @intCast(mip) }, testing.allocator);
        defer testing.allocator.free(pixels);
        for (0..pixels.len / 4) |i| try expectNearTexel(.{ 127, 127, 127, 255 }, pixels[i * 4 ..][0..4].*, 2);
    }

    // And a draw after it in the same submit has its own pipeline and textures.
    const quad = try Quad.init(&device);
    const out = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    const red = try device.createTexture(.{ .width = 1, .height = 1, .data = &level_palette[0] });
    const sampler = try device.createSampler(.{});
    const again = device.begin();
    try again.generateMips(texture);
    try again.beginPass(.{ .color = .{ .target = .{ .texture = out } } });
    try again.setPipeline(quad.pipeline);
    try again.setVertexBuffer(0, quad.corners, 0);
    try again.setTexture(0, red, sampler);
    try again.draw(.{ .vertex_count = 4 });
    try again.endPass();
    try device.submit();
    const drawn = try device.readTexture(out, testing.allocator);
    defer testing.allocator.free(drawn);
    try expectSolidLevel(drawn, level_palette[0]);
}

test "a sampler reads the level its bias and its range pick" {
    var device = try warpDevice();
    defer device.deinit();

    // Four texels wide, three levels, each one colour: which one a sampler reads shows in the pixel.
    const chain = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0 });
    var buffer: [4 * 4 * 4]u8 = undefined;
    for (0..3) |mip| {
        const size: usize = types.mipExtent(4, @intCast(mip));
        try device.writeTexture(chain, .{ .mip = @intCast(mip) }, solidLevel(buffer[0 .. size * size * 4], level_palette[mip]), 0, 0);
    }
    const quad = try Quad.init(&device);
    // As big as level zero: one texel a pixel, which is level zero unbiased.
    const out = try device.createTexture(.{ .width = 4, .height = 4, .usage = .{ .render_target = true } });
    const cases = [_]struct { types.SamplerDesc, [4]u8 }{
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest }, level_palette[0] },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_bias = 1 }, level_palette[1] },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_bias = 1, .lod_max = 0 }, level_palette[0] },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 2 }, level_palette[2] },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .none, .lod_bias = 1 }, level_palette[0] },
    };
    for (cases) |case| {
        const sampler = try device.createSampler(case[0]);
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = out } } });
        try cmd.setPipeline(quad.pipeline);
        try cmd.setVertexBuffer(0, quad.corners, 0);
        try cmd.setTexture(0, chain, sampler);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        const pixels = try device.readTexture(out, testing.allocator);
        defer testing.allocator.free(pixels);
        try expectSolidLevel(pixels, case[1]);
    }
}

test "a pass draws into one level of a chain, and the others keep theirs" {
    var device = try warpDevice();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 4, .mip_levels = 0, .usage = .{ .render_target = true } });
    var buffer: [8 * 4 * 4]u8 = undefined;
    for (0..4) |mip| {
        const size = [2]usize{ types.mipExtent(8, @intCast(mip)), types.mipExtent(4, @intCast(mip)) };
        try device.writeTexture(texture, .{ .mip = @intCast(mip) }, solidLevel(buffer[0 .. size[0] * size[1] * 4], level_palette[0]), 0, 0);
    }
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = texture }, .mip_level = 2, .clear_color = .{ 0, 0, 1, 1 } } });
    try cmd.endPass();
    try device.submit();

    for (0..4) |mip| {
        const pixels = try device.readSubresource(texture, .{ .mip = @intCast(mip) }, testing.allocator);
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, types.mipExtent(8, @intCast(mip)) * types.mipExtent(4, @intCast(mip)) * 4), pixels.len);
        try expectSolidLevel(pixels, if (mip == 2) level_palette[2] else level_palette[0]);
    }
}

test "a surface with no window is refused" {
    var device = try warpDevice();
    defer device.deinit();
    try testing.expectError(error.InvalidArgument, device.createSurface(.{}));
}

test "a surface presents in every mode and is resized, made to tear where the system allows it" {
    // A window that is never shown, for the swap chain to belong to.
    const window = user32.CreateWindowExA(0, "STATIC", "fluxion-rhi", user32.ws_popup, 0, 0, 32, 32, null, null, null, null) orelse return error.SkipZigTest;
    defer _ = user32.DestroyWindow(window);

    var device = try warpDevice();
    defer device.deinit();
    const surface = device.createSurface(.{ .native_window = @intFromPtr(window), .width = 32, .height = 32, .present_mode = .disabled }) catch return error.SkipZigTest;
    const res = as(SurfaceRes, device.surfaces.get(surface).?.native);
    const allowed = dxgi.allowsTearing(cast(device.impl).factory);
    try testing.expectEqual(if (allowed) swap_chain_allow_tearing else 0, res.flags);

    // Each mode presents - `disabled` with the tearing flag, which a chain
    // made without it would refuse - before a resize and after it.
    for (0..2) |round| {
        if (round == 1) try device.resizeSurface(surface, 48, 40);
        for ([_]types.PresentMode{ .disabled, .enabled, .adaptive, .mailbox, .disabled }) |mode| {
            try device.setPresentMode(surface, mode);
            const cmd = device.begin();
            try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface }, .clear_color = .{ 0, 1, 0, 1 } } });
            try cmd.endPass();
            try device.submit();
            try device.present(surface);
        }
    }
    try testing.expectEqual(@as(u32, 48), res.width);
    try testing.expectEqual(if (allowed) swap_chain_allow_tearing else 0, res.flags);
}

const user32 = struct {
    extern "user32" fn CreateWindowExA(u32, [*:0]const u8, [*:0]const u8, u32, i32, i32, i32, i32, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.winapi) ?*anyopaque;
    extern "user32" fn DestroyWindow(?*anyopaque) callconv(.winapi) c_int;
    /// `WS_POPUP`: no frame, and with no `WS_VISIBLE` never shown.
    const ws_popup: u32 = 0x80000000;
};
