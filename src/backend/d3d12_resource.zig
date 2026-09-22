// SPDX-License-Identifier: BSD-2-Clause

//! Resources, descriptor heaps and fences: `ID3D12Resource`,
//! `ID3D12DescriptorHeap`, `ID3D12Fence`, and the `ID3D12Device` calls and
//! value structs that make them.
//!
//! `fluxion-d3d` leaves these slots as `*const anyopaque` (see `d3d11.zig`'s
//! module comment for why); `slot()` gives one a signature at the point of
//! use, the same way `d3d11.zig` reaches `ID3D11Device`'s own scattered
//! slots. Sibling of `d3d12.zig` - everything here is what a renderer
//! allocates once it has a device.
//!
//! **The one real ABI trap in this file** is
//! `ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart` (and its GPU
//! twin): both return a struct - `D3D12_CPU_DESCRIPTOR_HANDLE`/
//! `D3D12_GPU_DESCRIPTOR_HANDLE`, each one `SIZE_T`/`UINT64` wide - by value.
//! MSVC's C++ ABI returns any class-typed value, however small, through a
//! hidden pointer the caller allocates and passes in - and for a non-static
//! member function that hidden pointer comes *after* the implicit `this`, not
//! before it: the real vtable slot is `RetType *Method(Self *this, RetType
//! *hiddenReturn)`. Several other bindings of this exact call document the
//! opposite order and are wrong about it - this one was checked the
//! reliable way, by trying both orders against a real WARP device and
//! keeping the one that does not crash: swapping the order in `cpuHeapStart`/
//! `gpuHeapStart` turns "a descriptor heap, and the CPU handle at its start"
//! below from a segfault on `Release` (a corrupted vtable pointer from
//! writing the handle into the wrong slot) into a passing test.

const std = @import("std");
const testing = std.testing;

const d3d = @import("fluxion_d3d");
const com = d3d.com;
const Guid = d3d.Guid;
const Hresult = d3d.Hresult;
const Error = d3d.Error;
const d3d12 = d3d.d3d12;
const dxgi = d3d.dxgi;

const ID3D12Device = d3d12.ID3D12Device;
const ID3D12Object = d3d12.ID3D12Object;
const ID3D12DeviceChild = d3d12.ID3D12DeviceChild;
const ID3D12Pageable = d3d12.ID3D12Pageable;

/// Give a type to a slot `fluxion-d3d` left opaque. `pub` because
/// `d3d12_command.zig` and `d3d12_pipeline.zig` need it too.
pub fn slot(comptime Fn: type, pointer: *const anyopaque) Fn {
    return @ptrCast(@alignCast(pointer));
}

// -------------------------------------------------------------------------
// Small value types the calls below take and return
// -------------------------------------------------------------------------

/// `DXGI_FORMAT`, the numbers this backend needs. Not the whole enum - see
/// `d3d11.zig`'s own `Format` for the rest of the table; the two libraries do
/// not share a type.
pub const Format = enum(u32) {
    unknown = 0,
    r32g32b32a32_float = 2,
    r32g32_float = 16,
    r32g32b32_float = 6,
    r8g8b8a8_unorm = 28,
    r8g8b8a8_uint = 30,
    r32_float = 41,
    r32_uint = 42,
    r32_sint = 43,
    r16_uint = 57,
    b8g8r8a8_unorm = 87,
    _,
};

/// `D3D12_CPU_DESCRIPTOR_HANDLE`. See the module comment for why the methods
/// that hand one back are not simply "return this".
pub const CpuDescriptorHandle = extern struct {
    ptr: usize = 0,

    pub fn offsetBy(self: CpuDescriptorHandle, descriptors: u32, increment: u32) CpuDescriptorHandle {
        return .{ .ptr = self.ptr + @as(usize, descriptors) * increment };
    }
};

/// `D3D12_GPU_DESCRIPTOR_HANDLE`. `UINT64`, not `SIZE_T`: a GPU address is
/// sixty-four bits even in a 32-bit process.
pub const GpuDescriptorHandle = extern struct {
    ptr: u64 = 0,

    pub fn offsetBy(self: GpuDescriptorHandle, descriptors: u32, increment: u32) GpuDescriptorHandle {
        return .{ .ptr = self.ptr + @as(u64, descriptors) * increment };
    }
};

/// `D3D12_DESCRIPTOR_HEAP_TYPE`.
pub const DescriptorHeapType = enum(u32) {
    cbv_srv_uav = 0,
    sampler = 1,
    rtv = 2,
    dsv = 3,
};

/// `D3D12_DESCRIPTOR_HEAP_FLAGS`.
pub const DescriptorHeapFlags = packed struct(u32) {
    /// The heap a shader can see through a descriptor table. An RTV or a DSV
    /// heap is never this - only a CBV_SRV_UAV or a SAMPLER heap may be.
    shader_visible: bool = false,
    _reserved: u31 = 0,
};

/// `D3D12_DESCRIPTOR_HEAP_DESC`.
pub const DescriptorHeapDesc = extern struct {
    type: DescriptorHeapType,
    num_descriptors: u32,
    flags: DescriptorHeapFlags = .{},
    node_mask: u32 = 0,
};

/// `D3D12_HEAP_TYPE`.
pub const HeapType = enum(u32) {
    default = 1,
    upload = 2,
    readback = 3,
    custom = 4,
};

pub const CpuPageProperty = enum(u32) { unknown = 0, not_available = 1, write_combine = 2, write_back = 3 };
pub const MemoryPool = enum(u32) { unknown = 0, l0 = 1, l1 = 2 };

/// `D3D12_HEAP_PROPERTIES`.
pub const HeapProperties = extern struct {
    type: HeapType,
    cpu_page_property: CpuPageProperty = .unknown,
    memory_pool_preference: MemoryPool = .unknown,
    creation_node_mask: u32 = 0,
    visible_node_mask: u32 = 0,

    /// The ordinary shape: no custom heap, one node. `UPLOAD` for a buffer the
    /// CPU writes and the GPU reads; `DEFAULT` for one only the GPU touches.
    pub fn of(kind: HeapType) HeapProperties {
        return .{ .type = kind };
    }
};

/// `D3D12_HEAP_FLAGS`. Unused by this backend - a committed resource with the
/// default flags is a heap and a resource made together - so this stays a
/// bare `u32` rather than a `packed struct` with thirty reserved bits.
pub const HeapFlags = u32;
pub const heap_flags_none: HeapFlags = 0;

/// `D3D12_RESOURCE_DIMENSION`.
pub const ResourceDimension = enum(u32) {
    unknown = 0,
    buffer = 1,
    texture1d = 2,
    texture2d = 3,
    texture3d = 4,
};

/// `D3D12_TEXTURE_LAYOUT`.
pub const TextureLayout = enum(u32) {
    /// Let the driver choose. What every texture in this backend uses.
    unknown = 0,
    /// What a buffer must be, and the only layout that is ever explicit here.
    row_major = 1,
    undefined_swizzle_64kb = 2,
    standard_swizzle_64kb = 3,
};

/// `D3D12_RESOURCE_FLAGS`.
pub const ResourceFlags = packed struct(u32) {
    allow_render_target: bool = false,
    allow_depth_stencil: bool = false,
    allow_unordered_access: bool = false,
    deny_shader_resource: bool = false,
    allow_cross_adapter: bool = false,
    allow_simultaneous_access: bool = false,
    video_decode_reference_only: bool = false,
    _reserved: u25 = 0,
};

/// `DXGI_SAMPLE_DESC`.
pub const SampleDesc = extern struct {
    count: u32 = 1,
    quality: u32 = 0,
};

/// `D3D12_RESOURCE_DESC`. Field order and widths are the header's exactly:
/// `Width` is a `UINT64` but `Height` is a plain `UINT`, and
/// `DepthOrArraySize`/`MipLevels` are both `UINT16` - narrower than every
/// other count in this file, and easy to get wrong by widening them to match.
pub const ResourceDesc = extern struct {
    dimension: ResourceDimension,
    alignment: u64 = 0,
    width: u64,
    height: u32 = 1,
    depth_or_array_size: u16 = 1,
    mip_levels: u16 = 1,
    format: Format = .unknown,
    sample: SampleDesc = .{},
    layout: TextureLayout = .unknown,
    flags: ResourceFlags = .{},

    /// A buffer of `size` bytes: the shape every field beyond `Width` is
    /// pinned to when `Dimension` is `BUFFER`.
    pub fn buffer(size: u64) ResourceDesc {
        return .{ .dimension = .buffer, .width = size, .layout = .row_major };
    }

    /// A plain 2D texture, one sample - the shape every texture this backend
    /// makes is.
    pub fn texture2d(width: u32, height: u32, format: Format, flags: ResourceFlags) ResourceDesc {
        return .{
            .dimension = .texture2d,
            .width = width,
            .height = height,
            .format = format,
            .flags = flags,
        };
    }
};

/// `D3D12_RESOURCE_STATES`. Not exhaustive, but every state this backend does
/// ask for has the header's number: two states silently swapped is a hazard
/// the driver does not catch.
pub const ResourceStates = packed struct(u32) {
    vertex_and_constant_buffer: bool = false,
    index_buffer: bool = false,
    render_target: bool = false,
    unordered_access: bool = false,
    depth_write: bool = false,
    depth_read: bool = false,
    non_pixel_shader_resource: bool = false,
    pixel_shader_resource: bool = false,
    stream_out: bool = false,
    indirect_argument: bool = false,
    copy_dest: bool = false,
    copy_source: bool = false,
    resolve_dest: bool = false,
    resolve_source: bool = false,
    _reserved: u18 = 0,

    /// `D3D12_RESOURCE_STATE_COMMON`/`D3D12_RESOURCE_STATE_PRESENT`: both zero,
    /// the same state under two names.
    pub const common: ResourceStates = .{};
    pub const present: ResourceStates = .{};
    /// `D3D12_RESOURCE_STATE_GENERIC_READ`, built from the fields themselves
    /// (`0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800 == 0xAC3`) rather than a
    /// second hand-computed literal that could drift from them.
    pub const generic_read: ResourceStates = .{
        .vertex_and_constant_buffer = true,
        .index_buffer = true,
        .non_pixel_shader_resource = true,
        .pixel_shader_resource = true,
        .indirect_argument = true,
        .copy_source = true,
    };
};

/// `D3D12_CLEAR_VALUE`. Only the colour branch of the union: this backend
/// never creates a depth-stencil committed resource.
pub const ClearValue = extern struct {
    format: Format,
    color: [4]f32 = .{ 0, 0, 0, 0 },
};

/// `D3D12_RANGE`. `Begin == End` means "nothing is read", the pattern an
/// upload buffer that is only ever written maps with.
pub const Range = extern struct {
    begin: usize = 0,
    end: usize = 0,

    pub const nothing_read: Range = .{};
};

/// `D3D12_CONSTANT_BUFFER_VIEW_DESC`.
pub const ConstantBufferViewDesc = extern struct {
    buffer_location: u64,
    size_in_bytes: u32,
};

/// `D3D12_SRV_DIMENSION`.
pub const SrvDimension = enum(u32) {
    unknown = 0,
    buffer = 1,
    texture1d = 2,
    texture1d_array = 3,
    texture2d = 4,
    texture2d_array = 5,
    texture2dms = 6,
    texture2dms_array = 7,
    texture3d = 8,
    texturecube = 9,
    texturecube_array = 10,
    raytracing_acceleration_structure = 11,
};

/// `D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3)`: the identity swizzle
/// every ordinary shader resource view wants, computed by hand once. `0x1688
/// == 5768`: bits 0-2/3-5/... pick source channel per component, and bit 12
/// is always set (the SDK's `ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES`) so an
/// all-zero mapping is never mistaken for a valid one.
pub const shader_4_component_mapping_identity: u32 = 5768;

/// `D3D12_TEX2D_SRV`, the shape this backend actually populates: MVP
/// textures are always `.d2`, one mip level.
pub const Tex2dSrv = extern struct {
    most_detailed_mip: u32 = 0,
    mip_levels: u32 = 1,
    plane_slice: u32 = 0,
    resource_min_lod_clamp: f32 = 0,
};

/// `D3D12_BUFFER_SRV`. Never populated - this backend has no buffer SRVs -
/// but its `FirstElement` is a `UINT64`, and that is exactly why this union
/// cannot be narrowed the way Direct3D 11's three view descriptions are (see
/// `d3d11.zig`): a `UINT64` anywhere in a C union forces the *whole* union to
/// 8-byte alignment and to a size that is a multiple of 8, which every
/// Direct3D 11 view union is exempt from and every Direct3D 12 one is not.
/// Leaving this branch out would silently shift `Tex2dSrv`'s fields four
/// bytes short of where the runtime expects them - exactly the bug this
/// comment exists to stop from coming back.
pub const BufferSrv = extern struct {
    first_element: u64 = 0,
    num_elements: u32 = 0,
    structure_byte_stride: u32 = 0,
    flags: u32 = 0,
};

/// `D3D12_TEX2D_ARRAY_SRV`. Also never populated, also declared only so the
/// union below is the runtime's true size (24 bytes) rather than
/// `Tex2dSrv`'s alone (16) - tied with `BufferSrv` for the union's widest
/// member.
pub const Tex2dArraySrv = extern struct {
    most_detailed_mip: u32 = 0,
    mip_levels: u32 = 1,
    first_array_slice: u32 = 0,
    array_size: u32 = 1,
    plane_slice: u32 = 0,
    resource_min_lod_clamp: f32 = 0,
};

/// `D3D12_SHADER_RESOURCE_VIEW_DESC`. A real `extern union` of every branch
/// that affects layout - not just `Texture2D`, the one this backend ever
/// populates - so Zig computes the union's true size and alignment. See
/// `BufferSrv`'s doc comment for why guessing goes wrong here specifically.
pub const ShaderResourceViewDesc = extern struct {
    format: Format,
    dimension: SrvDimension,
    shader_4_component_mapping: u32 = shader_4_component_mapping_identity,
    u: extern union {
        buffer: BufferSrv,
        texture2d: Tex2dSrv,
        texture2d_array: Tex2dArraySrv,
    } = .{ .texture2d = .{} },
};

/// `D3D12_RTV_DIMENSION`.
pub const RtvDimension = enum(u32) {
    unknown = 0,
    buffer = 1,
    texture1d = 2,
    texture1d_array = 3,
    texture2d = 4,
    texture2d_array = 5,
    texture2dms = 6,
    texture2dms_array = 7,
    texture3d = 8,
};

/// `D3D12_TEX2D_RTV`: the only render target view this backend ever makes -
/// one of a swap chain's back buffers.
pub const Tex2dRtv = extern struct {
    mip_slice: u32 = 0,
    plane_slice: u32 = 0,
};

/// `D3D12_BUFFER_RTV`. Never populated; declared, like `BufferSrv` above, only
/// because its `UINT64 FirstElement` forces the whole union - and so the
/// struct after it - to 8-byte alignment.
pub const BufferRtv = extern struct {
    first_element: u64 = 0,
    num_elements: u32 = 0,
};

/// `D3D12_RENDER_TARGET_VIEW_DESC`. A real `extern union` of the branches
/// that affect layout, for the same reason as `ShaderResourceViewDesc`'s.
pub const RenderTargetViewDesc = extern struct {
    format: Format,
    dimension: RtvDimension,
    u: extern union {
        buffer: BufferRtv,
        texture2d: Tex2dRtv,
    } = .{ .texture2d = .{} },
};

/// `D3D12_FILTER`. The same bit encoding as Direct3D 11's `FilterMode` (see
/// `d3d11.zig`).
pub const Filter = enum(u32) {
    min_mag_mip_point = 0x00,
    min_mag_point_mip_linear = 0x01,
    min_point_mag_linear_mip_point = 0x04,
    min_point_mag_mip_linear = 0x05,
    min_linear_mag_mip_point = 0x10,
    min_linear_mag_point_mip_linear = 0x11,
    min_mag_linear_mip_point = 0x14,
    min_mag_mip_linear = 0x15,
    anisotropic = 0x55,
    _,
};

/// `D3D12_TEXTURE_ADDRESS_MODE`.
pub const TextureAddressMode = enum(u32) { wrap = 1, mirror = 2, clamp = 3, border = 4, mirror_once = 5 };

/// `D3D12_COMPARISON_FUNC`.
pub const ComparisonFunc = enum(u32) {
    never = 1,
    less = 2,
    equal = 3,
    less_equal = 4,
    greater = 5,
    not_equal = 6,
    greater_equal = 7,
    always = 8,
};

/// `D3D12_SAMPLER_DESC`.
pub const SamplerDesc = extern struct {
    filter: Filter = .min_mag_mip_linear,
    address_u: TextureAddressMode = .clamp,
    address_v: TextureAddressMode = .clamp,
    address_w: TextureAddressMode = .clamp,
    mip_lod_bias: f32 = 0,
    max_anisotropy: u32 = 1,
    comparison_func: ComparisonFunc = .never,
    border_color: [4]f32 = .{ 0, 0, 0, 0 },
    min_lod: f32 = 0,
    max_lod: f32 = std.math.floatMax(f32),
};

/// `D3D12_SUBRESOURCE_FOOTPRINT`.
pub const SubresourceFootprint = extern struct {
    format: Format,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
};

/// `D3D12_PLACED_SUBRESOURCE_FOOTPRINT`.
pub const PlacedSubresourceFootprint = extern struct {
    offset: u64,
    footprint: SubresourceFootprint,
};

/// `D3D12_FENCE_FLAGS`. None of the bits (shared, cross-adapter, non-monitored)
/// are used by an in-process, single-GPU fence.
pub const FenceFlags = u32;
pub const fence_flags_none: FenceFlags = 0;

// -------------------------------------------------------------------------
// The interfaces
// -------------------------------------------------------------------------

/// `ID3D12Resource`. `GetDesc`, `WriteToSubresource`, `ReadFromSubresource`
/// and `GetHeapProperties` are left opaque - the first returns a struct by
/// value (see the module comment), and this backend never calls the rest.
/// `GetGPUVirtualAddress` is safe to declare plainly: its return is
/// `D3D12_GPU_VIRTUAL_ADDRESS`, a `typedef UINT64`, not a class type, so MSVC
/// returns it in a register like any other scalar.
pub const ID3D12Resource = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{696442BE-A72E-4059-BC79-5B5C98040FAD}");

    pub const VTable = extern struct {
        base: ID3D12Pageable.VTable,
        Map: *const fn (*ID3D12Resource, u32, ?*const Range, ?*?*anyopaque) callconv(.winapi) Hresult,
        Unmap: *const fn (*ID3D12Resource, u32, ?*const Range) callconv(.winapi) void,
        GetDesc: *const anyopaque,
        GetGPUVirtualAddress: *const fn (*ID3D12Resource) callconv(.winapi) u64,
        WriteToSubresource: *const anyopaque,
        ReadFromSubresource: *const anyopaque,
        GetHeapProperties: *const anyopaque,
    };
};

/// `ID3D12DescriptorHeap`. `GetDesc` is left opaque for the same reason as
/// `ID3D12Resource`'s; `cpuHeapStart`/`gpuHeapStart` below are the safe way to
/// reach the two slots after it.
pub const ID3D12DescriptorHeap = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{8EFB471D-616C-4F49-90F7-127BB763FA51}");

    pub const VTable = extern struct {
        base: ID3D12Pageable.VTable,
        GetDesc: *const anyopaque,
        /// The hidden-return-pointer trap: see the module comment. Called
        /// through `cpuHeapStart`/`gpuHeapStart`, never directly.
        GetCPUDescriptorHandleForHeapStart: *const fn (*ID3D12DescriptorHeap, *CpuDescriptorHandle) callconv(.winapi) *CpuDescriptorHandle,
        GetGPUDescriptorHandleForHeapStart: *const fn (*ID3D12DescriptorHeap, *GpuDescriptorHandle) callconv(.winapi) *GpuDescriptorHandle,
    };
};

/// The heap's first descriptor, on the CPU side - where `CreateRenderTargetView`
/// and friends write, and where a non-shader-visible heap's descriptors are
/// read from at command-list recording time.
pub fn cpuHeapStart(heap: *ID3D12DescriptorHeap) CpuDescriptorHandle {
    var out: CpuDescriptorHandle = undefined;
    _ = heap.vtable.GetCPUDescriptorHandleForHeapStart(heap, &out);
    return out;
}

/// The same heap's first descriptor as the GPU addresses it: what
/// `SetGraphicsRootDescriptorTable` is given. Only a shader-visible heap has
/// a meaningful one.
pub fn gpuHeapStart(heap: *ID3D12DescriptorHeap) GpuDescriptorHandle {
    var out: GpuDescriptorHandle = undefined;
    _ = heap.vtable.GetGPUDescriptorHandleForHeapStart(heap, &out);
    return out;
}

/// `ID3D12Fence`. `GetCompletedValue`/`Signal` are the poll-and-signal pair a
/// fully synchronous backend needs; `SetEventOnCompletion` and the Win32
/// event it would need are not declared - poll-spin is fine here.
pub const ID3D12Fence = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{0A753DCF-C4D8-4B91-ADF6-BE5A60D95A76}");

    pub const VTable = extern struct {
        base: ID3D12Pageable.VTable,
        GetCompletedValue: *const fn (*ID3D12Fence) callconv(.winapi) u64,
        SetEventOnCompletion: *const anyopaque,
        Signal: *const fn (*ID3D12Fence, u64) callconv(.winapi) Hresult,
    };
};

// -------------------------------------------------------------------------
// The device calls that make these
// -------------------------------------------------------------------------

pub fn createDescriptorHeap(device: *ID3D12Device, desc: DescriptorHeapDesc) Error!*ID3D12DescriptorHeap {
    const create = slot(*const fn (*ID3D12Device, *const DescriptorHeapDesc, *const Guid, *?*anyopaque) callconv(.winapi) Hresult, device.vtable.CreateDescriptorHeap);
    var raw: ?*anyopaque = null;
    const result = create(device, &desc, com.iidOf(ID3D12DescriptorHeap), &raw);
    return com.received(ID3D12DescriptorHeap, result, raw);
}

pub fn createCommittedResource(
    device: *ID3D12Device,
    heap_properties: HeapProperties,
    heap_flags: HeapFlags,
    desc: ResourceDesc,
    initial_state: ResourceStates,
    clear_value: ?*const ClearValue,
) Error!*ID3D12Resource {
    const create = slot(*const fn (
        *ID3D12Device,
        *const HeapProperties,
        HeapFlags,
        *const ResourceDesc,
        ResourceStates,
        ?*const ClearValue,
        *const Guid,
        *?*anyopaque,
    ) callconv(.winapi) Hresult, device.vtable.CreateCommittedResource);
    var raw: ?*anyopaque = null;
    const result = create(device, &heap_properties, heap_flags, &desc, initial_state, clear_value, com.iidOf(ID3D12Resource), &raw);
    return com.received(ID3D12Resource, result, raw);
}

pub fn createFence(device: *ID3D12Device, initial_value: u64, flags: FenceFlags) Error!*ID3D12Fence {
    const create = slot(*const fn (*ID3D12Device, u64, FenceFlags, *const Guid, *?*anyopaque) callconv(.winapi) Hresult, device.vtable.CreateFence);
    var raw: ?*anyopaque = null;
    const result = create(device, initial_value, flags, com.iidOf(ID3D12Fence), &raw);
    return com.received(ID3D12Fence, result, raw);
}

pub fn createConstantBufferView(device: *ID3D12Device, desc: ?*const ConstantBufferViewDesc, dest: CpuDescriptorHandle) void {
    const create = slot(*const fn (*ID3D12Device, ?*const ConstantBufferViewDesc, CpuDescriptorHandle) callconv(.winapi) void, device.vtable.CreateConstantBufferView);
    create(device, desc, dest);
}

pub fn createShaderResourceView(device: *ID3D12Device, resource: ?*ID3D12Resource, desc: ?*const ShaderResourceViewDesc, dest: CpuDescriptorHandle) void {
    const create = slot(*const fn (*ID3D12Device, ?*ID3D12Resource, ?*const ShaderResourceViewDesc, CpuDescriptorHandle) callconv(.winapi) void, device.vtable.CreateShaderResourceView);
    create(device, resource, desc, dest);
}

pub fn createRenderTargetView(device: *ID3D12Device, resource: ?*ID3D12Resource, desc: ?*const RenderTargetViewDesc, dest: CpuDescriptorHandle) void {
    const create = slot(*const fn (*ID3D12Device, ?*ID3D12Resource, ?*const RenderTargetViewDesc, CpuDescriptorHandle) callconv(.winapi) void, device.vtable.CreateRenderTargetView);
    create(device, resource, desc, dest);
}

pub fn createSampler(device: *ID3D12Device, desc: *const SamplerDesc, dest: CpuDescriptorHandle) void {
    const create = slot(*const fn (*ID3D12Device, *const SamplerDesc, CpuDescriptorHandle) callconv(.winapi) void, device.vtable.CreateSampler);
    create(device, desc, dest);
}

/// Copy `count` consecutive descriptors of `kind` starting at `src` into the
/// ones starting at `dest`. CPU-side and immediate - not a command-list
/// call - so it must happen between frames or before the command list that
/// will read the destination is submitted, never concurrently with the GPU
/// still reading the heap it writes into.
pub fn copyDescriptorsSimple(device: *ID3D12Device, count: u32, dest: CpuDescriptorHandle, src: CpuDescriptorHandle, kind: DescriptorHeapType) void {
    const copy = slot(*const fn (*ID3D12Device, u32, CpuDescriptorHandle, CpuDescriptorHandle, DescriptorHeapType) callconv(.winapi) void, device.vtable.CopyDescriptorsSimple);
    copy(device, count, dest, src, kind);
}

/// How big one descriptor of `kind` is on this device - a driver constant,
/// never zero. Unlike the rest of this file, `GetDescriptorHandleIncrementSize`
/// already has a real signature in `fluxion-d3d`, so no `slot()` cast is
/// needed here.
pub fn descriptorHandleIncrementSize(device: *ID3D12Device, kind: DescriptorHeapType) u32 {
    return device.vtable.GetDescriptorHandleIncrementSize(device, @intFromEnum(kind));
}

/// The staging layout, row count and pitch `CopyTextureRegion` needs to
/// upload into `resource`'s subresources from a buffer. Every output pointer
/// may be null except that asking for none of them is pointless;
/// `total_bytes` is the size the staging buffer must be at least.
pub fn getCopyableFootprints(
    device: *ID3D12Device,
    desc: *const ResourceDesc,
    first_subresource: u32,
    num_subresources: u32,
    base_offset: u64,
    layouts: ?[*]PlacedSubresourceFootprint,
    num_rows: ?[*]u32,
    row_size_bytes: ?[*]u64,
    total_bytes: ?*u64,
) void {
    const get = slot(*const fn (
        *ID3D12Device,
        *const ResourceDesc,
        u32,
        u32,
        u64,
        ?[*]PlacedSubresourceFootprint,
        ?[*]u32,
        ?[*]u64,
        ?*u64,
    ) callconv(.winapi) void, device.vtable.GetCopyableFootprints);
    get(device, desc, first_subresource, num_subresources, base_offset, layouts, num_rows, row_size_bytes, total_bytes);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "the descriptions the runtime reads are shaped as it expects" {
    try testing.expectEqual(@as(usize, 8), @sizeOf(usize)); // this file assumes a 64-bit host, as fluxion-d3d does
    try testing.expectEqual(@as(usize, 16), @sizeOf(DescriptorHeapDesc));
    try testing.expectEqual(@as(usize, 20), @sizeOf(HeapProperties));
    try testing.expectEqual(@as(usize, 56), @sizeOf(ResourceDesc));
    try testing.expectEqual(@as(u32, 0xAC3), @as(u32, @bitCast(ResourceStates.generic_read)));
    try testing.expectEqual(@as(usize, 16), @sizeOf(ConstantBufferViewDesc));
    // Both wider than a naive narrowing would suggest: `BufferSrv`'s and
    // `BufferRtv`'s `UINT64 FirstElement` forces the whole union - and so
    // the struct after it - to 8-byte alignment. See `BufferSrv`'s doc
    // comment; this is the test that would have caught the bug it describes.
    try testing.expectEqual(@as(usize, 40), @sizeOf(ShaderResourceViewDesc));
    try testing.expectEqual(@as(usize, 24), @sizeOf(RenderTargetViewDesc));
    try testing.expectEqual(@as(usize, 16), @offsetOf(ShaderResourceViewDesc, "u"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(RenderTargetViewDesc, "u"));
    try testing.expectEqual(@as(usize, 52), @sizeOf(SamplerDesc));
    try testing.expectEqual(@as(usize, 32), @sizeOf(PlacedSubresourceFootprint));
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

test "a descriptor heap, and the CPU handle at its start" {
    var lib = try loadOrSkip();
    defer lib.unload();
    var dxgi_lib = d3d.Dxgi.load() catch return error.SkipZigTest;
    defer dxgi_lib.unload();
    const device = try warpDeviceOrSkip(lib, &dxgi_lib);
    defer _ = com.release(device);

    const heap = try createDescriptorHeap(device, .{ .type = .rtv, .num_descriptors = 4 });
    defer _ = com.release(heap);

    // A real heap start is never the null pointer - if the hidden-return-
    // pointer order in `GetCPUDescriptorHandleForHeapStart` were backwards,
    // this would either fault or read zero.
    const start = cpuHeapStart(heap);
    try testing.expect(start.ptr != 0);

    const stride = descriptorHandleIncrementSize(device, .rtv);
    try testing.expect(stride > 0);
    const second = start.offsetBy(1, stride);
    try testing.expectEqual(start.ptr + stride, second.ptr);
}

test "an upload buffer, mapped and written" {
    var lib = try loadOrSkip();
    defer lib.unload();
    var dxgi_lib = d3d.Dxgi.load() catch return error.SkipZigTest;
    defer dxgi_lib.unload();
    const device = try warpDeviceOrSkip(lib, &dxgi_lib);
    defer _ = com.release(device);

    const res_obj = try createCommittedResource(
        device,
        .of(.upload),
        heap_flags_none,
        .buffer(256),
        .generic_read,
        null,
    );
    defer _ = com.release(res_obj);

    try testing.expect(res_obj.vtable.GetGPUVirtualAddress(res_obj) != 0);

    var mapped: ?*anyopaque = null;
    try res_obj.vtable.Map(res_obj, 0, &Range.nothing_read, &mapped).check();
    const bytes: [*]u8 = @ptrCast(mapped.?);
    bytes[0] = 0xAB;
    res_obj.vtable.Unmap(res_obj, 0, null);
}

test "a fence starts at its initial value and is signalled by the CPU" {
    var lib = try loadOrSkip();
    defer lib.unload();
    var dxgi_lib = d3d.Dxgi.load() catch return error.SkipZigTest;
    defer dxgi_lib.unload();
    const device = try warpDeviceOrSkip(lib, &dxgi_lib);
    defer _ = com.release(device);

    const fence = try createFence(device, 0, fence_flags_none);
    defer _ = com.release(fence);

    try testing.expectEqual(@as(u64, 0), fence.vtable.GetCompletedValue(fence));
    try fence.vtable.Signal(fence, 3).check();
    try testing.expectEqual(@as(u64, 3), fence.vtable.GetCompletedValue(fence));
}
