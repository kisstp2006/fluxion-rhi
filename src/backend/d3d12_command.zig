// SPDX-License-Identifier: BSD-2-Clause

//! Command allocators and command lists: `ID3D12CommandAllocator`,
//! `ID3D12CommandList`, `ID3D12GraphicsCommandList`, and the value structs
//! recording a frame needs - viewports, vertex/index buffer views, resource
//! barriers.
//!
//! `ID3D12GraphicsCommandList`'s vtable has around fifty slots; this backend
//! calls perhaps fifteen of them. Every slot keeps its place in the real
//! order, and only the ones this MVP actually calls are typed - the same
//! "declare it here with a signature" rule `d3d11.zig` follows for
//! `ID3D11Device`.

const std = @import("std");
const testing = std.testing;

const d3d = @import("fluxion_d3d");
const com = d3d.com;
const Guid = d3d.Guid;
const Hresult = d3d.Hresult;
const Error = d3d.Error;
const d3d12 = d3d.d3d12;
const dxgi = d3d.dxgi;
const resource = @import("d3d12_resource.zig");
const pipeline = @import("d3d12_pipeline.zig");

const ID3D12Device = d3d12.ID3D12Device;
const ID3D12DeviceChild = d3d12.ID3D12DeviceChild;
const ID3D12Pageable = d3d12.ID3D12Pageable;
const CommandListType = d3d12.CommandListType;
const ID3D12Resource = resource.ID3D12Resource;
const ID3D12DescriptorHeap = resource.ID3D12DescriptorHeap;
const CpuDescriptorHandle = resource.CpuDescriptorHandle;
const GpuDescriptorHandle = resource.GpuDescriptorHandle;
const Format = resource.Format;

/// A `BOOL`: zero is false, anything else is true.
const Bool = c_int;

// -------------------------------------------------------------------------
// Value types
// -------------------------------------------------------------------------

/// `D3D12_VIEWPORT`. Its own struct, distinct from Direct3D 11's `D3D11_VIEWPORT`
/// in name only - both are six `FLOAT`s in the same order.
pub const Viewport = extern struct {
    top_left_x: f32 = 0,
    top_left_y: f32 = 0,
    width: f32,
    height: f32,
    min_depth: f32 = 0,
    max_depth: f32 = 1,
};

/// `D3D12_RECT`, which is a `RECT`: four `LONG`s, left/top/right/bottom.
pub const Rect = extern struct {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
};

/// `D3D12_BOX`.
pub const Box = extern struct {
    left: u32,
    top: u32 = 0,
    front: u32 = 0,
    right: u32,
    bottom: u32 = 1,
    back: u32 = 1,
};

/// `D3D12_PRIMITIVE_TOPOLOGY`. What `IASetPrimitiveTopology` takes - not to be
/// confused with `PrimitiveTopologyType` in `d3d12_pipeline.zig`, a coarser
/// enum a pipeline state is built with.
pub const PrimitiveTopology = enum(u32) {
    undefined = 0,
    point_list = 1,
    line_list = 2,
    line_strip = 3,
    triangle_list = 4,
    triangle_strip = 5,
    _,
};

/// `D3D12_VERTEX_BUFFER_VIEW`.
pub const VertexBufferView = extern struct {
    buffer_location: u64,
    size_in_bytes: u32,
    stride_in_bytes: u32,
};

/// `D3D12_INDEX_BUFFER_VIEW`.
pub const IndexBufferView = extern struct {
    buffer_location: u64,
    size_in_bytes: u32,
    format: Format,
};

/// `D3D12_RESOURCE_BARRIER_TYPE`.
pub const ResourceBarrierType = enum(u32) { transition = 0, aliasing = 1, uav = 2 };

/// `D3D12_RESOURCE_BARRIER_FLAGS`.
pub const ResourceBarrierFlags = enum(u32) { none = 0, begin_only = 0x1, end_only = 0x2 };

/// `D3D12_RESOURCE_TRANSITION_BARRIER`.
pub const ResourceTransitionBarrier = extern struct {
    resource: *ID3D12Resource,
    subresource: u32 = transition_barrier_all_subresources,
    state_before: resource.ResourceStates,
    state_after: resource.ResourceStates,
};

/// `D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES`.
pub const transition_barrier_all_subresources: u32 = 0xFFFFFFFF;

/// `D3D12_RESOURCE_ALIASING_BARRIER`. Declared for layout completeness - the
/// union `ResourceBarrier` below is sized against it - but never populated.
pub const ResourceAliasingBarrier = extern struct {
    resource_before: ?*ID3D12Resource,
    resource_after: ?*ID3D12Resource,
};

/// `D3D12_RESOURCE_UAV_BARRIER`. Same reason as `ResourceAliasingBarrier`;
/// this backend has no unordered access views.
pub const ResourceUavBarrier = extern struct {
    resource: ?*ID3D12Resource,
};

/// `D3D12_RESOURCE_BARRIER`. `Type`/`Flags` lead an anonymous union whose
/// widest member is `Transition` (holds a pointer, rounds up to 24 bytes on
/// 8-byte alignment) - modelled as a real `extern union` of the three
/// variants so Zig lays it out exactly as C does.
pub const ResourceBarrier = extern struct {
    type: ResourceBarrierType,
    flags: ResourceBarrierFlags = .none,
    u: extern union {
        transition: ResourceTransitionBarrier,
        aliasing: ResourceAliasingBarrier,
        uav: ResourceUavBarrier,
    },

    pub fn transition(res: *ID3D12Resource, before: resource.ResourceStates, after: resource.ResourceStates) ResourceBarrier {
        return .{
            .type = .transition,
            .u = .{ .transition = .{ .resource = res, .state_before = before, .state_after = after } },
        };
    }
};

// -------------------------------------------------------------------------
// The interfaces
// -------------------------------------------------------------------------

/// `ID3D12CommandAllocator`.
pub const ID3D12CommandAllocator = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{6102DEE4-AF59-4B09-B999-B44D73F09B24}");

    pub const VTable = extern struct {
        base: ID3D12Pageable.VTable,
        Reset: *const fn (*ID3D12CommandAllocator) callconv(.winapi) Hresult,
    };
};

/// `ID3D12CommandList`: the base every kind of list shares. `GetType` is the
/// one slot it adds over `ID3D12DeviceChild`.
pub const ID3D12CommandList = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{7116D91C-E7E4-47CE-B8C6-EC8168F437E5}");

    pub const VTable = extern struct {
        base: ID3D12DeviceChild.VTable,
        GetType: *const fn (*ID3D12CommandList) callconv(.winapi) CommandListType,
    };
};

/// `ID3D12GraphicsCommandList`. See the module comment: every slot keeps its
/// real position, and only the ones this backend calls are given a real
/// signature.
pub const ID3D12GraphicsCommandList = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{5B160D0F-AC1B-4185-8BA8-B3AE42A5A455}");

    pub const VTable = extern struct {
        base: ID3D12CommandList.VTable,
        Close: *const fn (*ID3D12GraphicsCommandList) callconv(.winapi) Hresult,
        Reset: *const fn (*ID3D12GraphicsCommandList, *ID3D12CommandAllocator, ?*anyopaque) callconv(.winapi) Hresult,
        ClearState: *const anyopaque,
        DrawInstanced: *const fn (*ID3D12GraphicsCommandList, u32, u32, u32, u32) callconv(.winapi) void,
        DrawIndexedInstanced: *const fn (*ID3D12GraphicsCommandList, u32, u32, u32, i32, u32) callconv(.winapi) void,
        Dispatch: *const anyopaque,
        CopyBufferRegion: *const fn (*ID3D12GraphicsCommandList, *ID3D12Resource, u64, *ID3D12Resource, u64, u64) callconv(.winapi) void,
        CopyTextureRegion: *const fn (*ID3D12GraphicsCommandList, *const TextureCopyLocation, u32, u32, u32, *const TextureCopyLocation, ?*const Box) callconv(.winapi) void,
        CopyResource: *const fn (*ID3D12GraphicsCommandList, *ID3D12Resource, *ID3D12Resource) callconv(.winapi) void,
        CopyTiles: *const anyopaque,
        ResolveSubresource: *const anyopaque,
        IASetPrimitiveTopology: *const fn (*ID3D12GraphicsCommandList, PrimitiveTopology) callconv(.winapi) void,
        RSSetViewports: *const fn (*ID3D12GraphicsCommandList, u32, [*]const Viewport) callconv(.winapi) void,
        RSSetScissorRects: *const fn (*ID3D12GraphicsCommandList, u32, [*]const Rect) callconv(.winapi) void,
        OMSetBlendFactor: *const anyopaque,
        OMSetStencilRef: *const anyopaque,
        SetPipelineState: *const fn (*ID3D12GraphicsCommandList, *pipeline.ID3D12PipelineState) callconv(.winapi) void,
        ResourceBarrier: *const fn (*ID3D12GraphicsCommandList, u32, [*]const ResourceBarrier) callconv(.winapi) void,
        ExecuteBundle: *const anyopaque,
        SetDescriptorHeaps: *const fn (*ID3D12GraphicsCommandList, u32, [*]const *ID3D12DescriptorHeap) callconv(.winapi) void,
        SetComputeRootSignature: *const anyopaque,
        SetGraphicsRootSignature: *const fn (*ID3D12GraphicsCommandList, *pipeline.ID3D12RootSignature) callconv(.winapi) void,
        SetComputeRootDescriptorTable: *const anyopaque,
        SetGraphicsRootDescriptorTable: *const fn (*ID3D12GraphicsCommandList, u32, GpuDescriptorHandle) callconv(.winapi) void,
        SetComputeRoot32BitConstant: *const anyopaque,
        SetGraphicsRoot32BitConstant: *const anyopaque,
        SetComputeRoot32BitConstants: *const anyopaque,
        SetGraphicsRoot32BitConstants: *const anyopaque,
        SetComputeRootConstantBufferView: *const anyopaque,
        SetGraphicsRootConstantBufferView: *const fn (*ID3D12GraphicsCommandList, u32, u64) callconv(.winapi) void,
        SetComputeRootShaderResourceView: *const anyopaque,
        SetGraphicsRootShaderResourceView: *const anyopaque,
        SetComputeRootUnorderedAccessView: *const anyopaque,
        SetGraphicsRootUnorderedAccessView: *const anyopaque,
        IASetIndexBuffer: *const fn (*ID3D12GraphicsCommandList, ?*const IndexBufferView) callconv(.winapi) void,
        IASetVertexBuffers: *const fn (*ID3D12GraphicsCommandList, u32, u32, [*]const VertexBufferView) callconv(.winapi) void,
        SOSetTargets: *const anyopaque,
        OMSetRenderTargets: *const fn (*ID3D12GraphicsCommandList, u32, ?[*]const CpuDescriptorHandle, Bool, ?*const CpuDescriptorHandle) callconv(.winapi) void,
        ClearDepthStencilView: *const anyopaque,
        ClearRenderTargetView: *const fn (*ID3D12GraphicsCommandList, CpuDescriptorHandle, *const [4]f32, u32, ?[*]const Rect) callconv(.winapi) void,
        ClearUnorderedAccessViewUint: *const anyopaque,
        ClearUnorderedAccessViewFloat: *const anyopaque,
        DiscardResource: *const anyopaque,
        BeginQuery: *const anyopaque,
        EndQuery: *const anyopaque,
        ResolveQueryData: *const anyopaque,
        SetPredication: *const anyopaque,
        SetMarker: *const anyopaque,
        BeginEvent: *const anyopaque,
        EndEvent: *const anyopaque,
        ExecuteIndirect: *const anyopaque,
    };
};

/// `D3D12_TEXTURE_COPY_TYPE`.
pub const TextureCopyType = enum(u32) { subresource_index = 0, placed_footprint = 1 };

/// `D3D12_TEXTURE_COPY_LOCATION`. `pResource` leads, then `Type`, then a union
/// of `PlacedFootprint` (32 bytes, 8-byte aligned for its own `UINT64 Offset`)
/// and `SubresourceIndex` (a bare `UINT`) - modelled the same way
/// `ResourceBarrier`'s union is, as an `extern union` of the two real shapes.
pub const TextureCopyLocation = extern struct {
    resource: *ID3D12Resource,
    type: TextureCopyType,
    u: extern union {
        placed_footprint: resource.PlacedSubresourceFootprint,
        subresource_index: u32,
    },

    pub fn subresource(res: *ID3D12Resource, index: u32) TextureCopyLocation {
        return .{ .resource = res, .type = .subresource_index, .u = .{ .subresource_index = index } };
    }

    pub fn placed(res: *ID3D12Resource, footprint: resource.PlacedSubresourceFootprint) TextureCopyLocation {
        return .{ .resource = res, .type = .placed_footprint, .u = .{ .placed_footprint = footprint } };
    }
};

// -------------------------------------------------------------------------
// The device/queue calls that make these
// -------------------------------------------------------------------------

pub fn createCommandAllocator(device: *ID3D12Device, kind: CommandListType) Error!*ID3D12CommandAllocator {
    const create = resource.slot(*const fn (*ID3D12Device, CommandListType, *const Guid, *?*anyopaque) callconv(.winapi) Hresult, device.vtable.CreateCommandAllocator);
    var raw: ?*anyopaque = null;
    const result = create(device, kind, com.iidOf(ID3D12CommandAllocator), &raw);
    return com.received(ID3D12CommandAllocator, result, raw);
}

/// Made with no initial pipeline state, which is the shape every backend here
/// wants: `submit` sets the pipeline itself, per draw call, from the command
/// list it is walking.
pub fn createGraphicsCommandList(device: *ID3D12Device, node_mask: u32, kind: CommandListType, allocator: *ID3D12CommandAllocator) Error!*ID3D12GraphicsCommandList {
    const create = resource.slot(*const fn (
        *ID3D12Device,
        u32,
        CommandListType,
        *ID3D12CommandAllocator,
        ?*pipeline.ID3D12PipelineState,
        *const Guid,
        *?*anyopaque,
    ) callconv(.winapi) Hresult, device.vtable.CreateCommandList);
    var raw: ?*anyopaque = null;
    const result = create(device, node_mask, kind, allocator, null, com.iidOf(ID3D12GraphicsCommandList), &raw);
    return com.received(ID3D12GraphicsCommandList, result, raw);
}

pub fn executeCommandLists(queue: *d3d12.ID3D12CommandQueue, lists: []const *ID3D12GraphicsCommandList) void {
    const execute = resource.slot(*const fn (*d3d12.ID3D12CommandQueue, u32, [*]const *ID3D12CommandList) callconv(.winapi) void, queue.vtable.ExecuteCommandLists);
    const base: [*]const *ID3D12CommandList = @ptrCast(lists.ptr);
    execute(queue, @intCast(lists.len), base);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "the descriptions the runtime reads are shaped as it expects" {
    try testing.expectEqual(@as(usize, 24), @sizeOf(Viewport));
    try testing.expectEqual(@as(usize, 16), @sizeOf(Rect));
    try testing.expectEqual(@as(usize, 16), @sizeOf(VertexBufferView));
    try testing.expectEqual(@as(usize, 16), @sizeOf(IndexBufferView));
    try testing.expectEqual(@as(usize, 24), @sizeOf(ResourceTransitionBarrier));
    try testing.expectEqual(@as(usize, 32), @sizeOf(ResourceBarrier));
    try testing.expectEqual(@as(usize, 48), @sizeOf(TextureCopyLocation));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(PrimitiveTopology.triangle_list));
}

const D3d12 = d3d.D3d12;

fn loadOrSkip() !D3d12 {
    return D3d12.load() catch |err| switch (err) {
        error.LibraryNotFound => error.SkipZigTest,
        else => err,
    };
}

fn warpDeviceOrSkip(lib: D3d12, dxgi_lib: *d3d.Dxgi) !*ID3D12Device {
    const factory = try dxgi_lib.createFactory(dxgi.IDXGIFactory1, .{});
    defer _ = com.release(factory);
    const warp = dxgi.warpAdapter(factory) catch return error.SkipZigTest;
    defer _ = com.release(warp);
    return lib.createDevice(.{ .adapter = @ptrCast(warp) }) catch return error.SkipZigTest;
}

test "an allocator, a command list made from it, recorded and closed" {
    var lib = try loadOrSkip();
    defer lib.unload();
    var dxgi_lib = d3d.Dxgi.load() catch return error.SkipZigTest;
    defer dxgi_lib.unload();
    const device = try warpDeviceOrSkip(lib, &dxgi_lib);
    defer _ = com.release(device);

    const allocator = try createCommandAllocator(device, .direct);
    defer _ = com.release(allocator);

    const list = try createGraphicsCommandList(device, 0, .direct, allocator);
    defer _ = com.release(list);

    // Real Direct3D 12 API requirement: a command list is created already
    // recording, and must be closed before its first `ExecuteCommandLists`.
    try list.vtable.Close(list).check();

    // `Reset` opens it again on the same allocator.
    try list.vtable.Reset(list, allocator, null).check();
    list.vtable.IASetPrimitiveTopology(list, .triangle_list);
    try list.vtable.Close(list).check();

    // A command list knows what kind it is.
    try testing.expectEqual(CommandListType.direct, list.vtable.base.GetType(@ptrCast(list)));
}
