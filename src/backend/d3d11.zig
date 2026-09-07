// SPDX-License-Identifier: BSD-2-Clause

//! The Direct3D 11 backend. Windows only, feature level 11.0.
//!
//! `fluxion-d3d` stops at a device on purpose: the slots that make buffers
//! and shaders are in its vtables at their true indices, typed as opaque
//! pointers, and reaching one means declaring it here with a signature. This
//! file is those declarations - the same ones its drawing examples make -
//! plus the backend built on them. Nothing is linked: `d3d11.dll`, `dxgi.dll`
//! and `d3dcompiler_47.dll` are found at run time, and a machine without one
//! gets `error.NoDevice` rather than a program that does not start.
//!
//! **HLSL is compiled here**, at `createShader`, with `d3dcompiler_47`. A
//! program that would rather ship bytecode is a `ShaderDesc` field away, and
//! the compiler is loaded lazily so a program that never compiles never pays
//! for it.
//!
//! **Dynamic buffers keep a shadow copy.** Direct3D 11 cannot update part of
//! a constant buffer, and a `MAP_WRITE_DISCARD` hands back fresh memory with
//! nothing in it. So a dynamic buffer - which every uniform buffer is - keeps
//! its bytes on the CPU, an update writes there, and the whole thing goes to
//! the GPU in one map. That is also what makes `updateBuffer(offset, ...)`
//! mean the same thing here as on OpenGL.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const d3d = @import("fluxion_d3d");
const com = d3d.com;
const d3d11 = d3d.d3d11;
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
    if (builtin.os.tag != .windows) @compileError("fluxion-rhi: the Direct3D 11 backend is Windows only");
}

// -------------------------------------------------------------------------
// The values the calls take
// -------------------------------------------------------------------------

const Bool = c_int;

/// Give a type to a slot the loader left opaque.
fn slot(comptime Fn: type, pointer: *const anyopaque) Fn {
    return @ptrCast(@alignCast(pointer));
}

/// `DXGI_FORMAT`, as far as this backend needs it.
const Format = enum(u32) {
    unknown = 0,
    r32g32b32a32_float = 2,
    r32g32b32_float = 6,
    r16g16b16a16_float = 10,
    r32g32_float = 16,
    r8g8b8a8_unorm = 28,
    r8g8b8a8_unorm_srgb = 29,
    r8g8b8a8_uint = 30,
    d32_float = 40,
    r32_float = 41,
    r32_uint = 42,
    r32_sint = 43,
    d24_unorm_s8_uint = 45,
    r16_uint = 57,
    r8_unorm = 61,
    b8g8r8a8_unorm = 87,
    _,
};

const Usage = enum(u32) { default = 0, immutable = 1, dynamic = 2, staging = 3 };

const BindFlags = packed struct(u32) {
    vertex_buffer: bool = false,
    index_buffer: bool = false,
    constant_buffer: bool = false,
    shader_resource: bool = false,
    stream_output: bool = false,
    render_target: bool = false,
    depth_stencil: bool = false,
    unordered_access: bool = false,
    decoder: bool = false,
    video_encoder: bool = false,
    _reserved: u22 = 0,
};

const CpuAccess = packed struct(u32) {
    _reserved0: u16 = 0,
    write: bool = false,
    read: bool = false,
    _reserved18: u14 = 0,
};

const Map = enum(u32) { read = 1, write = 2, read_write = 3, write_discard = 4, write_no_overwrite = 5 };

const Topology = enum(u32) {
    undefined = 0,
    point_list = 1,
    line_list = 2,
    line_strip = 3,
    triangle_list = 4,
    triangle_strip = 5,
    _,
};

const InputClass = enum(u32) { per_vertex = 0, per_instance = 1 };

const Comparison = enum(u32) {
    never = 1,
    less = 2,
    equal = 3,
    less_equal = 4,
    greater = 5,
    not_equal = 6,
    greater_equal = 7,
    always = 8,
};

const DepthWriteMask = enum(u32) { zero = 0, all = 1 };

const StencilOp = enum(u32) {
    keep = 1,
    zero = 2,
    replace = 3,
    increment_saturate = 4,
    decrement_saturate = 5,
    invert = 6,
    increment = 7,
    decrement = 8,
};

const ClearFlags = packed struct(u32) {
    depth: bool = false,
    stencil: bool = false,
    _reserved: u30 = 0,
};

const Blend = enum(u32) {
    zero = 1,
    one = 2,
    src_color = 3,
    inv_src_color = 4,
    src_alpha = 5,
    inv_src_alpha = 6,
    dest_alpha = 7,
    inv_dest_alpha = 8,
    dest_color = 9,
    inv_dest_color = 10,
    src_alpha_sat = 11,
    blend_factor = 14,
    inv_blend_factor = 15,
};

const BlendOp = enum(u32) { add = 1, subtract = 2, rev_subtract = 3, min = 4, max = 5 };

const FillMode = enum(u32) { wireframe = 2, solid = 3 };

const CullMode = enum(u32) { none = 1, front = 2, back = 3 };

const FilterMode = enum(u32) {
    min_mag_mip_point = 0x00,
    min_mag_point_mip_linear = 0x01,
    min_point_mag_linear_mip_point = 0x04,
    min_point_mag_mip_linear = 0x05,
    min_linear_mag_mip_point = 0x10,
    min_linear_mag_point_mip_linear = 0x11,
    min_mag_linear_mip_point = 0x14,
    min_mag_mip_linear = 0x15,
};

const AddressMode = enum(u32) { wrap = 1, mirror = 2, clamp = 3, border = 4 };

const SwapEffect = enum(u32) { discard = 0, sequential = 1, flip_sequential = 3, flip_discard = 4 };

const Scaling = enum(u32) { stretch = 0, none = 1, aspect_ratio_stretch = 2 };

const AlphaMode = enum(u32) { unspecified = 0, premultiplied = 1, straight = 2, ignore = 3 };

const usage_render_target_output: u32 = 1 << 5;

const SampleDesc = extern struct {
    count: u32 = 1,
    quality: u32 = 0,
};

const SwapChainDesc1 = extern struct {
    width: u32 = 0,
    height: u32 = 0,
    format: Format = .r8g8b8a8_unorm,
    stereo: Bool = 0,
    sample: SampleDesc = .{},
    buffer_usage: u32 = usage_render_target_output,
    buffer_count: u32 = 2,
    scaling: Scaling = .none,
    swap_effect: SwapEffect = .flip_discard,
    alpha_mode: AlphaMode = .unspecified,
    flags: u32 = 0,
};

const BufferDesc = extern struct {
    byte_width: u32,
    usage: Usage = .default,
    bind: BindFlags = .{},
    cpu_access: CpuAccess = .{},
    misc: u32 = 0,
    structure_stride: u32 = 0,
};

const Texture2DDesc = extern struct {
    width: u32,
    height: u32,
    mip_levels: u32 = 1,
    array_size: u32 = 1,
    format: Format,
    sample: SampleDesc = .{},
    usage: Usage = .default,
    bind: BindFlags = .{},
    cpu_access: CpuAccess = .{},
    misc: u32 = 0,
};

const SubresourceData = extern struct {
    memory: *const anyopaque,
    row_pitch: u32 = 0,
    slice_pitch: u32 = 0,
};

const MappedSubresource = extern struct {
    data: ?[*]u8 = null,
    row_pitch: u32 = 0,
    depth_pitch: u32 = 0,
};

const Box = extern struct {
    left: u32,
    top: u32 = 0,
    front: u32 = 0,
    right: u32,
    bottom: u32 = 1,
    back: u32 = 1,
};

const Viewport = extern struct {
    left: f32 = 0,
    top: f32 = 0,
    width: f32,
    height: f32,
    min_depth: f32 = 0,
    max_depth: f32 = 1,
};

const RectL = extern struct {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
};

const InputElement = extern struct {
    semantic_name: [*:0]const u8,
    semantic_index: u32 = 0,
    format: Format,
    input_slot: u32 = 0,
    aligned_byte_offset: u32 = 0,
    input_slot_class: InputClass = .per_vertex,
    instance_step_rate: u32 = 0,
};

const StencilOpDesc = extern struct {
    fail: StencilOp = .keep,
    depth_fail: StencilOp = .keep,
    pass: StencilOp = .keep,
    function: Comparison = .always,
};

const DepthStencilDesc = extern struct {
    depth_enable: Bool = 1,
    depth_write_mask: DepthWriteMask = .all,
    depth_function: Comparison = .less,
    stencil_enable: Bool = 0,
    stencil_read_mask: u8 = 0xFF,
    stencil_write_mask: u8 = 0xFF,
    front_face: StencilOpDesc = .{},
    back_face: StencilOpDesc = .{},
};

const RenderTargetBlendDesc = extern struct {
    blend_enable: Bool = 0,
    src_blend: Blend = .one,
    dest_blend: Blend = .zero,
    blend_op: BlendOp = .add,
    src_blend_alpha: Blend = .one,
    dest_blend_alpha: Blend = .zero,
    blend_op_alpha: BlendOp = .add,
    render_target_write_mask: u8 = 0x0F,
};

const BlendDesc = extern struct {
    alpha_to_coverage_enable: Bool = 0,
    independent_blend_enable: Bool = 0,
    render_target: [8]RenderTargetBlendDesc = @splat(.{}),
};

const RasterizerDesc = extern struct {
    fill_mode: FillMode = .solid,
    cull_mode: CullMode = .none,
    front_counter_clockwise: Bool = 1,
    depth_bias: i32 = 0,
    depth_bias_clamp: f32 = 0,
    slope_scaled_depth_bias: f32 = 0,
    depth_clip_enable: Bool = 1,
    scissor_enable: Bool = 1,
    multisample_enable: Bool = 0,
    antialiased_line_enable: Bool = 0,
};

const SamplerStateDesc = extern struct {
    filter: FilterMode = .min_mag_mip_linear,
    address_u: AddressMode = .clamp,
    address_v: AddressMode = .clamp,
    address_w: AddressMode = .clamp,
    mip_lod_bias: f32 = 0,
    max_anisotropy: u32 = 1,
    comparison: Comparison = .never,
    border_color: [4]f32 = .{ 0, 0, 0, 0 },
    min_lod: f32 = -std.math.floatMax(f32),
    max_lod: f32 = std.math.floatMax(f32),
};

// -------------------------------------------------------------------------
// The interfaces
// -------------------------------------------------------------------------

fn DeviceChild(comptime identifier: []const u8) type {
    return extern struct {
        vtable: *const d3d11.ID3D11DeviceChild.VTable,
        pub const iid = Guid.parseComptime(identifier);
    };
}

const IBuffer = DeviceChild("{48570B85-D1EE-4FCD-A250-EB350722B037}");
const ITexture2D = DeviceChild("{6F15AAF2-D208-4E89-9AB4-489535D34F9C}");
const IRenderTargetView = DeviceChild("{DFDBA067-0B8D-4865-875B-D7B4516CC164}");
const IDepthStencilView = DeviceChild("{9FDAC92A-1876-48C3-AFAD-25B94F84A9B6}");
const IShaderResourceView = DeviceChild("{B0E06FE0-8192-4E1A-B1CA-36D7414710B2}");
const IInputLayout = DeviceChild("{E4819DDC-4CF0-4025-BD26-5DE82A3E07B7}");
const IVertexShader = DeviceChild("{3B301D64-D678-4289-8897-22F8928B72F3}");
const IPixelShader = DeviceChild("{EA82E40D-51DC-4F33-93D4-DB7C9125AE8C}");
const IDepthStencilState = DeviceChild("{03823EFB-8D8F-4E1C-9AA2-F64BB2CBFDF1}");
const IBlendState = DeviceChild("{75B68FAA-347D-4159-8F45-A0640F01CD9A}");
const IRasterizerState = DeviceChild("{9BB4AB81-AB1A-4D8F-B506-FC04200B6EE7}");
const ISamplerState = DeviceChild("{DA6FEA51-564C-4487-9810-F0D0F9B4E3A5}");
const IResource = DeviceChild("{DC8E63F3-D12B-4952-B47B-5E45026A862D}");

fn asResource(object: anytype) *IResource {
    return @ptrCast(object);
}

/// `ID3D11DeviceContext`, the drawing half. The loader declares the four
/// slots this inherits and stops; everything below follows them in order,
/// and the list stops at `Flush`, the last one anything here calls.
const IDeviceContext = extern struct {
    vtable: *const VTable,

    pub const iid = d3d11.ID3D11DeviceContext.iid;

    pub const VTable = extern struct {
        base: d3d11.ID3D11DeviceContext.VTable,
        VSSetConstantBuffers: *const fn (*IDeviceContext, u32, u32, [*]const ?*IBuffer) callconv(.winapi) void,
        PSSetShaderResources: *const fn (*IDeviceContext, u32, u32, [*]const ?*IShaderResourceView) callconv(.winapi) void,
        PSSetShader: *const fn (*IDeviceContext, ?*IPixelShader, ?[*]const ?*IUnknown, u32) callconv(.winapi) void,
        PSSetSamplers: *const fn (*IDeviceContext, u32, u32, [*]const ?*ISamplerState) callconv(.winapi) void,
        VSSetShader: *const fn (*IDeviceContext, ?*IVertexShader, ?[*]const ?*IUnknown, u32) callconv(.winapi) void,
        DrawIndexed: *const fn (*IDeviceContext, u32, u32, i32) callconv(.winapi) void,
        Draw: *const fn (*IDeviceContext, u32, u32) callconv(.winapi) void,
        Map: *const fn (*IDeviceContext, *IResource, u32, Map, u32, *MappedSubresource) callconv(.winapi) Hresult,
        Unmap: *const fn (*IDeviceContext, *IResource, u32) callconv(.winapi) void,
        PSSetConstantBuffers: *const fn (*IDeviceContext, u32, u32, [*]const ?*IBuffer) callconv(.winapi) void,
        IASetInputLayout: *const fn (*IDeviceContext, ?*IInputLayout) callconv(.winapi) void,
        IASetVertexBuffers: *const fn (*IDeviceContext, u32, u32, [*]const ?*IBuffer, [*]const u32, [*]const u32) callconv(.winapi) void,
        IASetIndexBuffer: *const fn (*IDeviceContext, ?*IBuffer, Format, u32) callconv(.winapi) void,
        DrawIndexedInstanced: *const fn (*IDeviceContext, u32, u32, u32, i32, u32) callconv(.winapi) void,
        DrawInstanced: *const fn (*IDeviceContext, u32, u32, u32, u32) callconv(.winapi) void,
        GSSetConstantBuffers: *const anyopaque,
        GSSetShader: *const anyopaque,
        IASetPrimitiveTopology: *const fn (*IDeviceContext, Topology) callconv(.winapi) void,
        VSSetShaderResources: *const fn (*IDeviceContext, u32, u32, [*]const ?*IShaderResourceView) callconv(.winapi) void,
        VSSetSamplers: *const fn (*IDeviceContext, u32, u32, [*]const ?*ISamplerState) callconv(.winapi) void,
        Begin: *const anyopaque,
        End: *const anyopaque,
        GetData: *const anyopaque,
        SetPredication: *const anyopaque,
        GSSetShaderResources: *const anyopaque,
        GSSetSamplers: *const anyopaque,
        OMSetRenderTargets: *const fn (*IDeviceContext, u32, ?[*]const ?*IRenderTargetView, ?*IDepthStencilView) callconv(.winapi) void,
        OMSetRenderTargetsAndUnorderedAccessViews: *const anyopaque,
        OMSetBlendState: *const fn (*IDeviceContext, ?*IBlendState, ?*const [4]f32, u32) callconv(.winapi) void,
        OMSetDepthStencilState: *const fn (*IDeviceContext, ?*IDepthStencilState, u32) callconv(.winapi) void,
        SOSetTargets: *const anyopaque,
        DrawAuto: *const anyopaque,
        DrawIndexedInstancedIndirect: *const anyopaque,
        DrawInstancedIndirect: *const anyopaque,
        Dispatch: *const anyopaque,
        DispatchIndirect: *const anyopaque,
        RSSetState: *const fn (*IDeviceContext, ?*IRasterizerState) callconv(.winapi) void,
        RSSetViewports: *const fn (*IDeviceContext, u32, ?[*]const Viewport) callconv(.winapi) void,
        RSSetScissorRects: *const fn (*IDeviceContext, u32, ?[*]const RectL) callconv(.winapi) void,
        CopySubresourceRegion: *const anyopaque,
        CopyResource: *const fn (*IDeviceContext, *IResource, *IResource) callconv(.winapi) void,
        UpdateSubresource: *const fn (*IDeviceContext, *IResource, u32, ?*const Box, *const anyopaque, u32, u32) callconv(.winapi) void,
        CopyStructureCount: *const anyopaque,
        ClearRenderTargetView: *const fn (*IDeviceContext, *IRenderTargetView, *const [4]f32) callconv(.winapi) void,
        ClearUnorderedAccessViewUint: *const anyopaque,
        ClearUnorderedAccessViewFloat: *const anyopaque,
        ClearDepthStencilView: *const fn (*IDeviceContext, *IDepthStencilView, ClearFlags, f32, u8) callconv(.winapi) void,
        GenerateMips: *const anyopaque,
        SetResourceMinLOD: *const anyopaque,
        GetResourceMinLOD: *const anyopaque,
        ResolveSubresource: *const anyopaque,
        ExecuteCommandList: *const anyopaque,
        HSSetShaderResources: *const anyopaque,
        HSSetShader: *const anyopaque,
        HSSetSamplers: *const anyopaque,
        HSSetConstantBuffers: *const anyopaque,
        DSSetShaderResources: *const anyopaque,
        DSSetShader: *const anyopaque,
        DSSetSamplers: *const anyopaque,
        DSSetConstantBuffers: *const anyopaque,
        CSSetShaderResources: *const anyopaque,
        CSSetUnorderedAccessViews: *const anyopaque,
        CSSetShader: *const anyopaque,
        CSSetSamplers: *const anyopaque,
        CSSetConstantBuffers: *const anyopaque,
        VSGetConstantBuffers: *const anyopaque,
        PSGetShaderResources: *const anyopaque,
        PSGetShader: *const anyopaque,
        PSGetSamplers: *const anyopaque,
        VSGetShader: *const anyopaque,
        PSGetConstantBuffers: *const anyopaque,
        IAGetInputLayout: *const anyopaque,
        IAGetVertexBuffers: *const anyopaque,
        IAGetIndexBuffer: *const anyopaque,
        GSGetConstantBuffers: *const anyopaque,
        GSGetShader: *const anyopaque,
        IAGetPrimitiveTopology: *const anyopaque,
        VSGetShaderResources: *const anyopaque,
        VSGetSamplers: *const anyopaque,
        GetPredication: *const anyopaque,
        GSGetShaderResources: *const anyopaque,
        GSGetSamplers: *const anyopaque,
        OMGetRenderTargets: *const anyopaque,
        OMGetRenderTargetsAndUnorderedAccessViews: *const anyopaque,
        OMGetBlendState: *const anyopaque,
        OMGetDepthStencilState: *const anyopaque,
        SOGetTargets: *const anyopaque,
        RSGetState: *const anyopaque,
        RSGetViewports: *const anyopaque,
        RSGetScissorRects: *const anyopaque,
        HSGetShaderResources: *const anyopaque,
        HSGetShader: *const anyopaque,
        HSGetSamplers: *const anyopaque,
        HSGetConstantBuffers: *const anyopaque,
        DSGetShaderResources: *const anyopaque,
        DSGetShader: *const anyopaque,
        DSGetSamplers: *const anyopaque,
        DSGetConstantBuffers: *const anyopaque,
        CSGetShaderResources: *const anyopaque,
        CSGetUnorderedAccessViews: *const anyopaque,
        CSGetShader: *const anyopaque,
        CSGetSamplers: *const anyopaque,
        CSGetConstantBuffers: *const anyopaque,
        ClearState: *const fn (*IDeviceContext) callconv(.winapi) void,
        Flush: *const fn (*IDeviceContext) callconv(.winapi) void,
    };
};

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
        ResizeBuffers: *const fn (*ISwapChain, u32, u32, u32, Format, u32) callconv(.winapi) Hresult,
        ResizeTarget: *const anyopaque,
        GetContainingOutput: *const anyopaque,
        GetFrameStatistics: *const anyopaque,
        GetLastPresentCount: *const anyopaque,
    };
};

const IFactory2 = extern struct {
    vtable: *const VTable,

    pub const iid = dxgi.IDXGIFactory2.iid;

    pub const VTable = extern struct {
        base: dxgi.IDXGIFactory1.VTable,
        IsWindowedStereoEnabled: *const anyopaque,
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

/// `ID3D11Texture2D::GetDesc`: the first of the texture's own slots, after
/// the five it inherits from `ID3D11Resource`.
const ITexture2DFull = extern struct {
    base: d3d11.ID3D11DeviceChild.VTable,
    GetType: *const anyopaque,
    SetEvictionPriority: *const anyopaque,
    GetEvictionPriority: *const anyopaque,
    GetDesc: *const fn (*ITexture2D, *Texture2DDesc) callconv(.winapi) void,
};

fn describeTexture(texture: *ITexture2D) Texture2DDesc {
    const full: *const ITexture2DFull = @ptrCast(texture.vtable);
    var desc: Texture2DDesc = undefined;
    full.GetDesc(texture, &desc);
    return desc;
}

// -------------------------------------------------------------------------
// The device's creation slots, with signatures
// -------------------------------------------------------------------------

const Raw = struct {
    device: *d3d11.ID3D11Device,

    fn createBuffer(self: Raw, desc: BufferDesc, initial: ?[]const u8) d3d.Error!*IBuffer {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const BufferDesc, ?*const SubresourceData, *?*IBuffer) callconv(.winapi) Hresult, self.device.vtable.CreateBuffer);
        var data: SubresourceData = undefined;
        if (initial) |bytes| data = .{ .memory = bytes.ptr };
        var buffer: ?*IBuffer = null;
        return com.received(IBuffer, create(self.device, &desc, if (initial == null) null else &data, &buffer), buffer);
    }

    fn createTexture2D(self: Raw, desc: Texture2DDesc, initial: ?SubresourceData) d3d.Error!*ITexture2D {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const Texture2DDesc, ?*const SubresourceData, *?*ITexture2D) callconv(.winapi) Hresult, self.device.vtable.CreateTexture2D);
        var texture: ?*ITexture2D = null;
        return com.received(ITexture2D, create(self.device, &desc, if (initial) |*d| d else null, &texture), texture);
    }

    fn createShaderResourceView(self: Raw, resource: *IResource) d3d.Error!*IShaderResourceView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const anyopaque, *?*IShaderResourceView) callconv(.winapi) Hresult, self.device.vtable.CreateShaderResourceView);
        var view: ?*IShaderResourceView = null;
        return com.received(IShaderResourceView, create(self.device, resource, null, &view), view);
    }

    fn createRenderTargetView(self: Raw, resource: *IResource) d3d.Error!*IRenderTargetView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const anyopaque, *?*IRenderTargetView) callconv(.winapi) Hresult, self.device.vtable.CreateRenderTargetView);
        var view: ?*IRenderTargetView = null;
        return com.received(IRenderTargetView, create(self.device, resource, null, &view), view);
    }

    fn createDepthStencilView(self: Raw, resource: *IResource) d3d.Error!*IDepthStencilView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const anyopaque, *?*IDepthStencilView) callconv(.winapi) Hresult, self.device.vtable.CreateDepthStencilView);
        var view: ?*IDepthStencilView = null;
        return com.received(IDepthStencilView, create(self.device, resource, null, &view), view);
    }

    fn createInputLayout(self: Raw, elements: []const InputElement, code: []const u8) d3d.Error!*IInputLayout {
        const create = slot(*const fn (*d3d11.ID3D11Device, [*]const InputElement, u32, [*]const u8, usize, *?*IInputLayout) callconv(.winapi) Hresult, self.device.vtable.CreateInputLayout);
        var layout: ?*IInputLayout = null;
        return com.received(IInputLayout, create(self.device, elements.ptr, @intCast(elements.len), code.ptr, code.len, &layout), layout);
    }

    fn createVertexShader(self: Raw, code: []const u8) d3d.Error!*IVertexShader {
        const create = slot(*const fn (*d3d11.ID3D11Device, [*]const u8, usize, ?*IUnknown, *?*IVertexShader) callconv(.winapi) Hresult, self.device.vtable.CreateVertexShader);
        var shader: ?*IVertexShader = null;
        return com.received(IVertexShader, create(self.device, code.ptr, code.len, null, &shader), shader);
    }

    fn createPixelShader(self: Raw, code: []const u8) d3d.Error!*IPixelShader {
        const create = slot(*const fn (*d3d11.ID3D11Device, [*]const u8, usize, ?*IUnknown, *?*IPixelShader) callconv(.winapi) Hresult, self.device.vtable.CreatePixelShader);
        var shader: ?*IPixelShader = null;
        return com.received(IPixelShader, create(self.device, code.ptr, code.len, null, &shader), shader);
    }

    fn createBlendState(self: Raw, desc: BlendDesc) d3d.Error!*IBlendState {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const BlendDesc, *?*IBlendState) callconv(.winapi) Hresult, self.device.vtable.CreateBlendState);
        var state: ?*IBlendState = null;
        return com.received(IBlendState, create(self.device, &desc, &state), state);
    }

    fn createDepthStencilState(self: Raw, desc: DepthStencilDesc) d3d.Error!*IDepthStencilState {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const DepthStencilDesc, *?*IDepthStencilState) callconv(.winapi) Hresult, self.device.vtable.CreateDepthStencilState);
        var state: ?*IDepthStencilState = null;
        return com.received(IDepthStencilState, create(self.device, &desc, &state), state);
    }

    fn createRasterizerState(self: Raw, desc: RasterizerDesc) d3d.Error!*IRasterizerState {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const RasterizerDesc, *?*IRasterizerState) callconv(.winapi) Hresult, self.device.vtable.CreateRasterizerState);
        var state: ?*IRasterizerState = null;
        return com.received(IRasterizerState, create(self.device, &desc, &state), state);
    }

    fn createSamplerState(self: Raw, desc: SamplerStateDesc) d3d.Error!*ISamplerState {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const SamplerStateDesc, *?*ISamplerState) callconv(.winapi) Hresult, self.device.vtable.CreateSamplerState);
        var state: ?*ISamplerState = null;
        return com.received(ISamplerState, create(self.device, &desc, &state), state);
    }
};

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------

const D3d = struct {
    gpa: Allocator,
    library: d3d.D3d11,
    dxgi_library: d3d.Dxgi,
    factory: *dxgi.IDXGIFactory1,
    compiler: ?d3d.Compiler = null,
    device: d3d11.Device,
    raw: Raw,
    context: *IDeviceContext,
    debug: bool,
    renderer: [128]u8 = undefined,
    renderer_len: usize = 0,

    // Per-submit state.
    pipeline: ?*PipelineRes = null,
    target_width: u32 = 0,
    target_height: u32 = 0,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
};

const max_vertex_slots = 8;

const VertexBinding = struct {
    buffer: ?*IBuffer = null,
    offset: u32 = 0,
};

const BufferRes = struct {
    buffer: *IBuffer,
    size: usize,
    /// The CPU's copy of a dynamic buffer. See the module comment.
    shadow: ?[]u8,
};

const TextureRes = struct {
    texture: *ITexture2D,
    srv: ?*IShaderResourceView,
    rtv: ?*IRenderTargetView,
    dsv: ?*IDepthStencilView,
    width: u32,
    height: u32,
    format: types.Format,
};

const SamplerRes = struct {
    state: *ISamplerState,
};

const ShaderRes = struct {
    vertex: *IVertexShader,
    pixel: *IPixelShader,
    /// Kept for `CreateInputLayout`, which checks a layout against the
    /// signature the shader was compiled with.
    vertex_code: []u8,
};

const PipelineRes = struct {
    shader: *ShaderRes,
    layout: ?*IInputLayout,
    blend: *IBlendState,
    rasterizer: *IRasterizerState,
    depth: *IDepthStencilState,
    topology: Topology,
    strides: [max_vertex_slots]u32,
};

const SurfaceRes = struct {
    swap_chain: *ISwapChain,
    rtv: *IRenderTargetView,
    width: u32,
    height: u32,
};

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!struct { backend.Impl, *const backend.Vtable } {
    var library = d3d.D3d11.load() catch return error.NoDevice;
    errdefer library.unload();
    var dxgi_library = d3d.Dxgi.load() catch return error.NoDevice;
    errdefer dxgi_library.unload();

    const factory = dxgi_library.createFactory(dxgi.IDXGIFactory1, .{ .debug = desc.debug }) catch
        dxgi_library.createFactory(dxgi.IDXGIFactory1, .{}) catch return error.NoDevice;
    errdefer _ = com.release(factory);

    var device = library.createDevice(.{
        .driver = if (desc.software) .warp else .hardware,
        .flags = .{ .debug = desc.debug, .bgra_support = true },
    }) catch |err| switch (err) {
        // The debug layer is an optional Windows feature; a program that asked
        // for it on a machine without it still gets a device.
        else => if (desc.debug)
            library.createDevice(.{
                .driver = if (desc.software) .warp else .hardware,
                .flags = .{ .bgra_support = true },
            }) catch return error.NoDevice
        else
            return error.NoDevice,
    };
    errdefer device.release();

    const self = try gpa.create(D3d);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .library = library,
        .dxgi_library = dxgi_library,
        .factory = factory,
        .device = device,
        .raw = .{ .device = device.device },
        .context = @ptrCast(device.context),
        .debug = desc.debug,
    };

    // The adapter's name, for the log line. WARP has no adapter worth asking.
    if (desc.software) {
        const name = "Microsoft Basic Render Driver (WARP)";
        @memcpy(self.renderer[0..name.len], name);
        self.renderer_len = name.len;
    } else if (blk: {
        var adapters = dxgi.adapters(factory);
        break :blk adapters.next() catch null;
    }) |adapter| {
        defer _ = com.release(adapter);
        if (dxgi.describe(adapter)) |description| {
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
    .createBuffer = createBuffer,
    .destroyBuffer = destroyBuffer,
    .updateBuffer = updateBuffer,
    .createTexture = createTexture,
    .destroyTexture = destroyTexture,
    .updateTexture = updateTexture,
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
    self.context.vtable.ClearState(self.context);
    self.context.vtable.Flush(self.context);
    if (self.compiler) |*compiler| compiler.unload();
    self.device.release();
    _ = com.release(self.factory);
    self.dxgi_library.unload();
    self.library.unload();
    self.gpa.destroy(self);
}

fn info(impl: backend.Impl) types.Info {
    const self = cast(impl);
    return .{ .backend = .d3d11, .renderer = self.renderer[0..self.renderer_len] };
}

// -------------------------------------------------------------------------
// Buffers
// -------------------------------------------------------------------------

fn createBuffer(impl: backend.Impl, desc: types.BufferDesc) Error!backend.Native {
    const self = cast(impl);

    const res = try self.gpa.create(BufferRes);
    errdefer self.gpa.destroy(res);

    const dynamic = desc.dynamic or desc.kind == .uniform;
    const size: u32 = @intCast(if (desc.kind == .uniform) std.mem.alignForward(usize, desc.size, 16) else desc.size);

    var shadow: ?[]u8 = null;
    errdefer if (shadow) |s| self.gpa.free(s);
    var initial: ?[]const u8 = desc.data;
    if (dynamic) {
        // The shadow is the whole buffer, so the first upload is complete.
        const bytes = try self.gpa.alloc(u8, size);
        @memset(bytes, 0);
        if (desc.data) |data| @memcpy(bytes[0..data.len], data);
        shadow = bytes;
        initial = bytes;
    } else if (desc.data) |data| if (data.len < size) {
        // Immutable wants the full size; pad a short initial copy.
        const bytes = try self.gpa.alloc(u8, size);
        defer self.gpa.free(bytes);
        @memset(bytes, 0);
        @memcpy(bytes[0..data.len], data);
        res.* = .{
            .buffer = self.raw.createBuffer(bufferDesc(desc.kind, size, false, true), bytes) catch return error.Failed,
            .size = size,
            .shadow = null,
        };
        return res;
    };

    res.* = .{
        .buffer = self.raw.createBuffer(bufferDesc(desc.kind, size, dynamic, initial != null and !dynamic), initial) catch return error.Failed,
        .size = size,
        .shadow = shadow,
    };
    return res;
}

fn bufferDesc(kind: types.BufferKind, size: u32, dynamic: bool, immutable: bool) BufferDesc {
    return .{
        .byte_width = size,
        .usage = if (dynamic) .dynamic else if (immutable) .immutable else .default,
        .bind = switch (kind) {
            .vertex => .{ .vertex_buffer = true },
            .index => .{ .index_buffer = true },
            .uniform => .{ .constant_buffer = true },
        },
        .cpu_access = .{ .write = dynamic },
    };
}

fn destroyBuffer(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    _ = com.release(res.buffer);
    if (res.shadow) |s| self.gpa.free(s);
    self.gpa.destroy(res);
}

fn updateBuffer(impl: backend.Impl, native: backend.Native, offset: usize, bytes: []const u8) Error!void {
    const self = cast(impl);
    const res = as(BufferRes, native);
    const context = self.context;
    const resource = asResource(res.buffer);

    if (res.shadow) |shadow| {
        @memcpy(shadow[offset..][0..bytes.len], bytes);
        var mapped: MappedSubresource = .{};
        context.vtable.Map(context, resource, 0, .write_discard, 0, &mapped).check() catch return error.Failed;
        defer context.vtable.Unmap(context, resource, 0);
        const data = mapped.data orelse return error.Failed;
        @memcpy(data[0..shadow.len], shadow);
    } else {
        const box: Box = .{ .left = @intCast(offset), .right = @intCast(offset + bytes.len) };
        context.vtable.UpdateSubresource(context, resource, 0, &box, bytes.ptr, 0, 0);
    }
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

fn dxgiFormat(format: types.Format) Format {
    return switch (format) {
        .rgba8_unorm => .r8g8b8a8_unorm,
        .rgba8_unorm_srgb => .r8g8b8a8_unorm_srgb,
        .bgra8_unorm => .b8g8r8a8_unorm,
        .r8_unorm => .r8_unorm,
        .rgba16_float => .r16g16b16a16_float,
        .rgba32_float => .r32g32b32a32_float,
        .depth24_stencil8 => .d24_unorm_s8_uint,
        .depth32_float => .d32_float,
    };
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const depth = desc.format.isDepth();
    var initial: ?SubresourceData = null;
    if (desc.data) |data| initial = .{ .memory = data.ptr, .row_pitch = @intCast(desc.effectiveRowPitch()) };

    const texture = self.raw.createTexture2D(.{
        .width = desc.width,
        .height = desc.height,
        .format = dxgiFormat(desc.format),
        .bind = .{
            .shader_resource = desc.usage.sampled and !depth,
            .render_target = desc.usage.render_target and !depth,
            .depth_stencil = depth,
        },
    }, initial) catch return error.Failed;
    errdefer _ = com.release(texture);

    res.* = .{
        .texture = texture,
        .srv = null,
        .rtv = null,
        .dsv = null,
        .width = desc.width,
        .height = desc.height,
        .format = desc.format,
    };
    if (desc.usage.sampled and !depth) res.srv = self.raw.createShaderResourceView(asResource(texture)) catch return error.Failed;
    errdefer if (res.srv) |v| {
        _ = com.release(v);
    };
    if (desc.usage.render_target and !depth) res.rtv = self.raw.createRenderTargetView(asResource(texture)) catch return error.Failed;
    errdefer if (res.rtv) |v| {
        _ = com.release(v);
    };
    if (depth) res.dsv = self.raw.createDepthStencilView(asResource(texture)) catch return error.Failed;

    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (res.srv) |v| _ = com.release(v);
    if (res.rtv) |v| _ = com.release(v);
    if (res.dsv) |v| _ = com.release(v);
    _ = com.release(res.texture);
    self.gpa.destroy(res);
}

fn updateTexture(impl: backend.Impl, native: backend.Native, bytes: []const u8, row_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    self.context.vtable.UpdateSubresource(self.context, asResource(res.texture), 0, null, bytes.ptr, @intCast(row_pitch), 0);
}

fn readTexture(impl: backend.Impl, native: backend.Native, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const context = self.context;

    // The pipeline cannot read a staging resource and the CPU cannot read
    // anything else, so two textures and a copy. There is no shorter way.
    const staging = self.raw.createTexture2D(.{
        .width = res.width,
        .height = res.height,
        .format = dxgiFormat(res.format),
        .usage = .staging,
        .cpu_access = .{ .read = true },
    }, null) catch return error.Failed;
    defer _ = com.release(staging);

    context.vtable.CopyResource(context, asResource(staging), asResource(res.texture));

    var mapped: MappedSubresource = .{};
    context.vtable.Map(context, asResource(staging), 0, .read, 0, &mapped).check() catch return error.Failed;
    defer context.vtable.Unmap(context, asResource(staging), 0);
    const data = mapped.data orelse return error.Failed;

    const bpp = res.format.bytesPerPixel();
    const row = @as(usize, res.width) * 4;
    const pixels = try gpa.alloc(u8, row * res.height);
    errdefer gpa.free(pixels);

    for (0..res.height) |y| {
        const source = data[y * mapped.row_pitch ..][0 .. res.width * bpp];
        const destination = pixels[y * row ..][0..row];
        switch (res.format) {
            .rgba8_unorm, .rgba8_unorm_srgb => @memcpy(destination, source),
            .bgra8_unorm => for (0..res.width) |x| {
                destination[x * 4 + 0] = source[x * 4 + 2];
                destination[x * 4 + 1] = source[x * 4 + 1];
                destination[x * 4 + 2] = source[x * 4 + 0];
                destination[x * 4 + 3] = source[x * 4 + 3];
            },
            .r8_unorm => for (0..res.width) |x| {
                destination[x * 4 + 0] = source[x];
                destination[x * 4 + 1] = source[x];
                destination[x * 4 + 2] = source[x];
                destination[x * 4 + 3] = 255;
            },
            .rgba16_float, .rgba32_float => return error.Unsupported,
            .depth24_stencil8, .depth32_float => unreachable, // refused by Device
        }
    }
    return pixels;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);
    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    const filter: FilterMode = switch (desc.min_filter) {
        .nearest => switch (desc.mag_filter) {
            .nearest => .min_mag_mip_point,
            .linear => .min_point_mag_linear_mip_point,
        },
        .linear => switch (desc.mag_filter) {
            .nearest => .min_linear_mag_mip_point,
            .linear => .min_mag_linear_mip_point,
        },
    };
    res.* = .{
        .state = self.raw.createSamplerState(.{
            .filter = filter,
            .address_u = address(desc.wrap_u),
            .address_v = address(desc.wrap_v),
        }) catch return error.Failed,
    };
    return res;
}

fn address(wrap: types.Wrap) AddressMode {
    return switch (wrap) {
        .repeat => .wrap,
        .clamp_to_edge => .clamp,
        .mirror => .mirror,
    };
}

fn destroySampler(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SamplerRes, native);
    _ = com.release(res.state);
    self.gpa.destroy(res);
}

// -------------------------------------------------------------------------
// Shaders and pipelines
// -------------------------------------------------------------------------

fn compilerOf(self: *D3d, log: *Io.Writer) Error!d3d.Compiler {
    if (self.compiler) |compiler| return compiler;
    self.compiler = d3d.Compiler.load() catch {
        log.writeAll("fluxion-rhi: d3dcompiler_47.dll is not on this machine, so HLSL source cannot be compiled") catch {};
        return error.ShaderFailed;
    };
    return self.compiler.?;
}

fn compileStage(compiler: d3d.Compiler, source: []const u8, target: [:0]const u8, label: []const u8, log: *Io.Writer) Error!*d3d.ID3DBlob {
    _ = label;
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
        log.writeAll("fluxion-rhi: the Direct3D 11 backend needs `ShaderDesc.hlsl`, and none was given") catch {};
        return error.ShaderFailed;
    };
    const compiler = try compilerOf(self, log);

    const vs_blob = try compileStage(compiler, sources.vertex, "vs_5_0", desc.label, log);
    defer _ = com.release(vs_blob);
    const ps_blob = try compileStage(compiler, sources.fragment, "ps_5_0", desc.label, log);
    defer _ = com.release(ps_blob);

    const res = try self.gpa.create(ShaderRes);
    errdefer self.gpa.destroy(res);
    const vertex_code = try self.gpa.dupe(u8, vs_blob.bytes());
    errdefer self.gpa.free(vertex_code);

    const vertex = self.raw.createVertexShader(vertex_code) catch return error.ShaderFailed;
    errdefer _ = com.release(vertex);
    const pixel = self.raw.createPixelShader(ps_blob.bytes()) catch return error.ShaderFailed;

    res.* = .{ .vertex = vertex, .pixel = pixel, .vertex_code = vertex_code };
    return res;
}

fn destroyShader(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(ShaderRes, native);
    _ = com.release(res.pixel);
    _ = com.release(res.vertex);
    self.gpa.free(res.vertex_code);
    self.gpa.destroy(res);
}

fn vertexFormat(format: types.VertexFormat) Format {
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

fn createPipeline(impl: backend.Impl, desc: types.PipelineDesc, shader: backend.Native, log: *Io.Writer) Error!backend.Native {
    const self = cast(impl);
    const shader_res = as(ShaderRes, shader);

    if (desc.buffers.len > max_vertex_slots) {
        log.print("fluxion-rhi: the Direct3D 11 backend binds at most {d} vertex buffers", .{max_vertex_slots}) catch {};
        return error.PipelineFailed;
    }

    const res = try self.gpa.create(PipelineRes);
    errdefer self.gpa.destroy(res);

    // The semantic is `ATTR` with the location as its index: the one
    // convention that lets a GLSL `layout(location = n)` and an HLSL input
    // agree without a reflection step.
    var layout: ?*IInputLayout = null;
    if (desc.attributes.len > 0) {
        var elements: [32]InputElement = undefined;
        if (desc.attributes.len > elements.len) return error.PipelineFailed;
        for (desc.attributes, 0..) |attribute, i| {
            const step = desc.buffers[attribute.buffer].step;
            elements[i] = .{
                .semantic_name = "ATTR",
                .semantic_index = attribute.location,
                .format = vertexFormat(attribute.format),
                .input_slot = attribute.buffer,
                .aligned_byte_offset = attribute.offset,
                .input_slot_class = if (step == .instance) .per_instance else .per_vertex,
                .instance_step_rate = if (step == .instance) 1 else 0,
            };
        }
        layout = self.raw.createInputLayout(elements[0..desc.attributes.len], shader_res.vertex_code) catch {
            log.writeAll("the vertex layout does not match the shader's inputs (every attribute needs an `ATTRn` the shader reads)") catch {};
            return error.PipelineFailed;
        };
    }
    errdefer if (layout) |l| {
        _ = com.release(l);
    };

    var blend_desc: BlendDesc = .{};
    blend_desc.render_target[0] = .{
        .blend_enable = if (desc.blend.enabled) 1 else 0,
        .src_blend = blendFactor(desc.blend.src_rgb),
        .dest_blend = blendFactor(desc.blend.dst_rgb),
        .blend_op = blendOp(desc.blend.op_rgb),
        .src_blend_alpha = blendFactor(desc.blend.src_alpha),
        .dest_blend_alpha = blendFactor(desc.blend.dst_alpha),
        .blend_op_alpha = blendOp(desc.blend.op_alpha),
    };
    const blend = self.raw.createBlendState(blend_desc) catch return error.PipelineFailed;
    errdefer _ = com.release(blend);

    const rasterizer = self.raw.createRasterizerState(.{
        .cull_mode = switch (desc.cull) {
            .none => .none,
            .back => .back,
            .front => .front,
        },
        .front_counter_clockwise = if (desc.front_face == .ccw) 1 else 0,
    }) catch return error.PipelineFailed;
    errdefer _ = com.release(rasterizer);

    const depth = self.raw.createDepthStencilState(.{
        .depth_enable = if (desc.depth.test_enabled) 1 else 0,
        .depth_write_mask = if (desc.depth.write) .all else .zero,
        .depth_function = comparison(desc.depth.compare),
    }) catch return error.PipelineFailed;

    var strides: [max_vertex_slots]u32 = @splat(0);
    for (desc.buffers, 0..) |buffer, i| strides[i] = buffer.stride;

    res.* = .{
        .shader = shader_res,
        .layout = layout,
        .blend = blend,
        .rasterizer = rasterizer,
        .depth = depth,
        .topology = switch (desc.topology) {
            .triangles => .triangle_list,
            .triangle_strip => .triangle_strip,
            .lines => .line_list,
            .line_strip => .line_strip,
            .points => .point_list,
        },
        .strides = strides,
    };
    return res;
}

fn blendFactor(f: types.BlendFactor) Blend {
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

fn blendOp(op: types.BlendOp) BlendOp {
    return switch (op) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .rev_subtract,
        .min => .min,
        .max => .max,
    };
}

fn comparison(f: types.CompareFn) Comparison {
    return switch (f) {
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

fn destroyPipeline(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(PipelineRes, native);
    if (res.layout) |l| _ = com.release(l);
    com.releaseAll(.{ res.blend, res.rasterizer, res.depth });
    if (self.pipeline == res) self.pipeline = null;
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
        com.unknown(self.device.device),
        @ptrFromInt(desc.native_window),
        &chain_desc,
        null,
        null,
        &swap_chain,
    ).check() catch return error.Failed;
    errdefer _ = com.release(swap_chain.?);

    const res = try self.gpa.create(SurfaceRes);
    errdefer self.gpa.destroy(res);
    res.* = .{ .swap_chain = swap_chain.?, .rtv = undefined, .width = 0, .height = 0 };
    try attachBackBuffer(self, res);
    return res;
}

fn attachBackBuffer(self: *D3d, res: *SurfaceRes) Error!void {
    var raw: ?*anyopaque = null;
    res.swap_chain.vtable.GetBuffer(res.swap_chain, 0, com.iidOf(ITexture2D), &raw).check() catch return error.Failed;
    const back_buffer = com.received(ITexture2D, .s_ok, raw) catch return error.Failed;
    defer _ = com.release(back_buffer);

    res.rtv = self.raw.createRenderTargetView(asResource(back_buffer)) catch return error.Failed;
    const desc = describeTexture(back_buffer);
    res.width = desc.width;
    res.height = desc.height;
}

fn destroySurface(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    // A back buffer still bound to the output stage stays alive however many
    // times its view is released, and a flip-model swap chain then waits at
    // exit for buffers the compositor cannot have back.
    self.context.vtable.ClearState(self.context);
    self.context.vtable.Flush(self.context);
    _ = com.release(res.rtv);
    _ = com.release(res.swap_chain);
    self.gpa.destroy(res);
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    self.context.vtable.ClearState(self.context);
    self.context.vtable.Flush(self.context);
    _ = com.release(res.rtv);
    res.swap_chain.vtable.ResizeBuffers(res.swap_chain, 0, width, height, .unknown, 0).check() catch return error.Failed;
    try attachBackBuffer(self, res);
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

fn present(impl: backend.Impl, native: backend.Native, vsync: bool) Error!void {
    _ = impl;
    const res = as(SurfaceRes, native);
    res.swap_chain.vtable.Present(res.swap_chain, if (vsync) 1 else 0, 0).check() catch |err| switch (err) {
        error.DeviceRemoved, error.DeviceReset => return error.DeviceLost,
        else => return error.Failed,
    };
}

// -------------------------------------------------------------------------
// Submitting
// -------------------------------------------------------------------------

fn submit(impl: backend.Impl, device: *Device, list: []const commands.Command) Error!void {
    const self = cast(impl);
    const context = self.context;

    for (list) |command| {
        switch (command) {
            .begin_pass => |pass| try beginPass(self, device, pass),
            .end_pass => {
                // Unbind what the pass drew into and read from, so the next
                // pass can sample this one's target without a hazard.
                const no_targets = [_]?*IRenderTargetView{null};
                context.vtable.OMSetRenderTargets(context, 1, &no_targets, null);
                const no_views: [8]?*IShaderResourceView = @splat(null);
                context.vtable.PSSetShaderResources(context, 0, no_views.len, &no_views);
                self.pipeline = null;
            },
            .set_pipeline => |h| {
                const res = as(PipelineRes, device.pipelines.get(h).?.native);
                self.pipeline = res;
                self.bindings_dirty = true;
                context.vtable.IASetInputLayout(context, res.layout);
                context.vtable.VSSetShader(context, res.shader.vertex, null, 0);
                context.vtable.PSSetShader(context, res.shader.pixel, null, 0);
                context.vtable.OMSetBlendState(context, res.blend, null, 0xFFFFFFFF);
                context.vtable.RSSetState(context, res.rasterizer);
                context.vtable.OMSetDepthStencilState(context, res.depth, 0);
                context.vtable.IASetPrimitiveTopology(context, res.topology);
            },
            .set_viewport => |v| {
                context.vtable.RSSetViewports(context, 1, &[_]Viewport{.{
                    .left = v.x,
                    .top = v.y,
                    .width = v.width,
                    .height = v.height,
                    .min_depth = v.min_depth,
                    .max_depth = v.max_depth,
                }});
            },
            .set_scissor => |maybe| {
                const r: RectL = if (maybe) |r| .{
                    .left = r.x,
                    .top = r.y,
                    .right = r.x + @as(i32, @intCast(r.width)),
                    .bottom = r.y + @as(i32, @intCast(r.height)),
                } else .{
                    .left = 0,
                    .top = 0,
                    .right = @intCast(self.target_width),
                    .bottom = @intCast(self.target_height),
                };
                context.vtable.RSSetScissorRects(context, 1, &[_]RectL{r});
            },
            .set_vertex_buffer => |b| {
                if (b.slot >= max_vertex_slots) return error.Unsupported;
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                self.vertex_bindings[b.slot] = .{ .buffer = res.buffer, .offset = b.offset };
                self.bindings_dirty = true;
            },
            .set_index_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                context.vtable.IASetIndexBuffer(context, res.buffer, if (b.format == .u16) .r16_uint else .r32_uint, 0);
            },
            .set_uniform_buffer => |b| {
                const res = as(BufferRes, device.buffers.get(b.buffer).?.native);
                const buffers = [_]?*IBuffer{res.buffer};
                context.vtable.VSSetConstantBuffers(context, b.slot, 1, &buffers);
                context.vtable.PSSetConstantBuffers(context, b.slot, 1, &buffers);
            },
            .set_texture => |b| {
                const texture = as(TextureRes, device.textures.get(b.texture).?.native);
                const sampler = as(SamplerRes, device.samplers.get(b.sampler).?.native);
                const views = [_]?*IShaderResourceView{texture.srv};
                const samplers = [_]?*ISamplerState{sampler.state};
                context.vtable.PSSetShaderResources(context, b.slot, 1, &views);
                context.vtable.PSSetSamplers(context, b.slot, 1, &samplers);
                context.vtable.VSSetShaderResources(context, b.slot, 1, &views);
                context.vtable.VSSetSamplers(context, b.slot, 1, &samplers);
            },
            .draw => |d| {
                flushBindings(self);
                context.vtable.DrawInstanced(context, d.vertex_count, d.instance_count, d.first_vertex, 0);
            },
            .draw_indexed => |d| {
                flushBindings(self);
                context.vtable.DrawIndexedInstanced(context, d.index_count, d.instance_count, d.first_index, d.base_vertex, 0);
            },
        }
    }

    if (self.debug) {
        const reason = self.device.removedReason();
        if (reason.failed()) return error.DeviceLost;
    }
}

fn beginPass(self: *D3d, device: *Device, pass: types.RenderPassDesc) Error!void {
    const context = self.context;

    var rtv: *IRenderTargetView = undefined;
    switch (pass.color.target) {
        .surface => |h| {
            const res = as(SurfaceRes, device.surfaces.get(h).?.native);
            rtv = res.rtv;
            self.target_width = res.width;
            self.target_height = res.height;
        },
        .texture => |h| {
            const res = as(TextureRes, device.textures.get(h).?.native);
            rtv = res.rtv.?;
            self.target_width = res.width;
            self.target_height = res.height;
        },
    }
    var dsv: ?*IDepthStencilView = null;
    if (pass.depth) |depth| dsv = as(TextureRes, device.textures.get(depth.texture).?.native).dsv;

    // Whatever was sampled last pass may be this pass's target.
    const no_views: [8]?*IShaderResourceView = @splat(null);
    context.vtable.PSSetShaderResources(context, 0, no_views.len, &no_views);

    const targets = [_]?*IRenderTargetView{rtv};
    context.vtable.OMSetRenderTargets(context, 1, &targets, dsv);

    if (pass.color.load == .clear) context.vtable.ClearRenderTargetView(context, rtv, &pass.color.clear_color);
    if (pass.depth) |depth| if (depth.load == .clear) {
        context.vtable.ClearDepthStencilView(context, dsv.?, .{ .depth = true, .stencil = true }, depth.clear_depth, depth.clear_stencil);
    };

    context.vtable.RSSetViewports(context, 1, &[_]Viewport{.{
        .width = @floatFromInt(self.target_width),
        .height = @floatFromInt(self.target_height),
    }});
    context.vtable.RSSetScissorRects(context, 1, &[_]RectL{.{
        .left = 0,
        .top = 0,
        .right = @intCast(self.target_width),
        .bottom = @intCast(self.target_height),
    }});

    self.pipeline = null;
    self.vertex_bindings = @splat(.{});
    self.bindings_dirty = true;
}

/// Hand the bound vertex buffers over with the pipeline's strides, which is
/// why this waits for the draw: a stride belongs to a pipeline and a buffer
/// may be bound before one is set.
fn flushBindings(self: *D3d) void {
    if (!self.bindings_dirty) return;
    self.bindings_dirty = false;
    const pipeline = self.pipeline orelse return;

    var buffers: [max_vertex_slots]?*IBuffer = undefined;
    var offsets: [max_vertex_slots]u32 = undefined;
    for (0..max_vertex_slots) |i| {
        buffers[i] = self.vertex_bindings[i].buffer;
        offsets[i] = self.vertex_bindings[i].offset;
    }
    self.context.vtable.IASetVertexBuffers(self.context, 0, max_vertex_slots, &buffers, &pipeline.strides, &offsets);
}

// -------------------------------------------------------------------------
// Tests. WARP needs no window and no card, so the whole backend is checked
// here by drawing and reading back.
// -------------------------------------------------------------------------

const testing = std.testing;

test "the structs are shaped the way the runtime reads them" {
    try testing.expectEqual(@as(usize, 24), @sizeOf(BufferDesc));
    try testing.expectEqual(@as(usize, 24), @sizeOf(Viewport));
    try testing.expectEqual(@as(usize, 32), @sizeOf(RenderTargetBlendDesc));
    try testing.expectEqual(@as(usize, 264), @sizeOf(BlendDesc));
    try testing.expectEqual(@as(usize, 40), @sizeOf(RasterizerDesc));
    try testing.expectEqual(@as(usize, 52), @sizeOf(SamplerStateDesc));
    try testing.expectEqual(@as(usize, 52), @sizeOf(DepthStencilDesc));
    try testing.expectEqual(@as(usize, 32), @sizeOf(InputElement));
    try testing.expectEqual(@as(usize, 24), @sizeOf(Box));
}

fn warpDevice() !Device {
    return Device.init(testing.allocator, .{ .backend = .d3d11, .software = true }) catch |err| switch (err) {
        error.NoDevice, error.Unsupported => error.SkipZigTest,
        else => err,
    };
}

const flat_vs =
    \\struct In { float2 position : ATTR0; float4 colour : ATTR1; };
    \\struct Out { float4 position : SV_POSITION; float4 colour : COLOR0; };
    \\cbuffer Frame : register(b0) { float4 tint; };
    \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.colour = i.colour * tint; return o; }
;
const flat_ps =
    \\struct Out { float4 position : SV_POSITION; float4 colour : COLOR0; };
    \\float4 main(Out i) : SV_TARGET { return i.colour; }
;

const Vertex = extern struct { position: [2]f32, colour: [4]f32 };

test "a triangle through the whole backend, on WARP" {
    var device = try warpDevice();
    defer device.deinit();

    const target = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });
    const shader = device.createShader(.{ .hlsl = .{ .vertex = flat_vs, .fragment = flat_ps } }) catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0 },
            .{ .location = 1, .format = .float4, .offset = 8 },
        },
        .buffers = &.{.{ .stride = @sizeOf(Vertex) }},
    });

    // Red, pointing up, over the middle of a blue field.
    const vertices = [_]Vertex{
        .{ .position = .{ -0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.0, 0.6 }, .colour = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 0.6, -0.6 }, .colour = .{ 1, 0, 0, 1 } },
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
    const tint = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    try device.updateBuffer(tint, 0, std.mem.asBytes(&[4]f32{ 1, 1, 1, 1 }));

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.setUniformBuffer(0, tint);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);

    const at = struct {
        fn f(p: []const u8, x: usize, y: usize) [4]u8 {
            return p[(y * 64 + x) * 4 ..][0..4].*;
        }
    }.f;
    // Top-left origin: the point of the triangle is near the top, so a pixel
    // just under the top edge in the middle is red and the corners are blue.
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, at(pixels, 32, 20));
    try testing.expectEqual([4]u8{ 255, 0, 0, 255 }, at(pixels, 32, 40));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 2, 2));
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 61, 61));
    // And the base is at the bottom, so the top corners are blue while a
    // pixel low in the middle is red.
    try testing.expectEqual([4]u8{ 0, 0, 255, 255 }, at(pixels, 2, 61));
}

test "a uniform buffer update reaches the shader, and a partial one keeps the rest" {
    var device = try warpDevice();
    defer device.deinit();

    const target = try device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .render_target = true } });
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = flat_vs, .fragment = flat_ps } });
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0 },
            .{ .location = 1, .format = .float4, .offset = 8 },
        },
        .buffers = &.{.{ .stride = @sizeOf(Vertex) }},
        .topology = .triangle_strip,
    });
    // A quad over everything, white.
    const vertices = [_]Vertex{
        .{ .position = .{ -1, -1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ -1, 1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, -1 }, .colour = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, 1 }, .colour = .{ 1, 1, 1, 1 } },
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
    const tint = try device.createBuffer(.{ .kind = .uniform, .size = 16 });
    try device.updateBuffer(tint, 0, std.mem.asBytes(&[4]f32{ 0, 1, 0, 1 }));
    // Only the red channel, four bytes into the float4: the green must survive.
    try device.updateBuffer(tint, 0, std.mem.asBytes(&[1]f32{1}));

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.setUniformBuffer(0, tint);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, testing.allocator);
    defer testing.allocator.free(pixels);
    try testing.expectEqual([4]u8{ 255, 255, 0, 255 }, pixels[(2 * 16 + 2) * 4 ..][0..4].*);
}

test "a shader that does not compile says why" {
    var device = try warpDevice();
    defer device.deinit();
    try testing.expectError(error.ShaderFailed, device.createShader(.{ .hlsl = .{ .vertex = "this is not HLSL", .fragment = flat_ps } }));
    try testing.expect(device.diagnostics().len > 10);
    // And a shader in the wrong language is refused with a reason, not a crash.
    try testing.expectError(error.ShaderFailed, device.createShader(.{ .glsl = .{ .vertex = "", .fragment = "" } }));
    try testing.expect(std.mem.indexOf(u8, device.diagnostics(), "hlsl") != null);
}
