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
//!
//! **A format is a row in a table.** What Direct3D calls each `types.Format` -
//! and how a texel of it is unpacked to RGBA8 - is `native_table`, with `null`
//! for what this API cannot do, and the compiler checks that every format has
//! a row. `caps` is not written down anywhere: it is what `CheckFormatSupport`
//! and `CheckMultisampleQualityLevels` said about each row when the device
//! opened, and `Device` refuses what it did not say before this file is asked.
//!
//! **A texture is a resource and its views.** A volume is a `Texture3D`, and a
//! cube and an array are `Texture2D`s with layers; the view a shader reads
//! through has the texture's own shape, and a view a pass draws into names one
//! level of one layer and is made the first time a pass asks. A depth texture
//! is made typeless, so that it can be the depth buffer of one pass and the
//! shadow map of the next.
//!
//! **Levels are made by the hardware where it can.** `GenerateMips` needs a
//! texture created to be one, so every colour chain the format allows is, and
//! the render target bind that costs is this file's business and not the
//! caller's. **A multisampled pass resolves in `end_pass`**, once its targets
//! are unbound, into a texture or into the swap chain's back buffer.

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

/// `DXGI_FORMAT`, as far as this backend needs it. The numbers are
/// `dxgiformat.h`'s and a test at the bottom of this file pins them: a wrong
/// one is not a compile error, it is a texture in another format.
const Format = enum(u32) {
    unknown = 0,
    r32g32b32a32_float = 2,
    r32g32b32_float = 6,
    r16g16b16a16_float = 10,
    r32g32_float = 16,
    // A depth format has three names, because a texture that is drawn into as
    // depth and read as a shader resource is made in one and seen through the
    // other two: the resource is typeless, the depth-stencil view says how
    // it is written and the shader resource view says how it is read.
    r32g8x24_typeless = 19,
    d32_float_s8x24_uint = 20,
    r32_float_x8x24_typeless = 21,
    r10g10b10a2_unorm = 24,
    r11g11b10_float = 26,
    r8g8b8a8_unorm = 28,
    r8g8b8a8_unorm_srgb = 29,
    r8g8b8a8_uint = 30,
    r16g16_float = 34,
    r32_typeless = 39,
    d32_float = 40,
    r32_float = 41,
    r32_uint = 42,
    r32_sint = 43,
    r24g8_typeless = 44,
    d24_unorm_s8_uint = 45,
    r24_unorm_x8_typeless = 46,
    r8g8_unorm = 49,
    r16_typeless = 53,
    r16_float = 54,
    d16_unorm = 55,
    r16_unorm = 56,
    r16_uint = 57,
    r8_unorm = 61,
    bc1_typeless = 70,
    bc1_unorm = 71,
    bc1_unorm_srgb = 72,
    bc3_typeless = 76,
    bc3_unorm = 77,
    bc3_unorm_srgb = 78,
    bc4_typeless = 79,
    bc4_unorm = 80,
    bc5_typeless = 82,
    bc5_unorm = 83,
    b8g8r8a8_unorm = 87,
    b8g8r8a8_unorm_srgb = 91,
    bc6h_typeless = 94,
    bc6h_uf16 = 95,
    bc7_typeless = 97,
    bc7_unorm = 98,
    bc7_unorm_srgb = 99,
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

/// `D3D11_FILTER`. It is a bit field, and the names are only the ones a
/// caller of this file reads: the rest are made by `samplerFilter` from the
/// bits below, so the enum stays open.
const FilterMode = enum(u32) {
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

/// What `D3D11_ENCODE_BASIC_FILTER` and `D3D11_ENCODE_ANISOTROPIC_FILTER` put
/// together: a bit for each stage that is linear, another that says the
/// lookup is a comparison, and all three linear stages plus the one between
/// them for anisotropy.
const filter_bits = struct {
    const mip_linear: u32 = 0x01;
    const mag_linear: u32 = 0x04;
    const min_linear: u32 = 0x10;
    const comparison: u32 = 0x80;
};

/// `D3D11_RESOURCE_MISC_FLAG`, the bits a texture uses.
const TextureMisc = packed struct(u32) {
    /// The texture can be filled by `GenerateMips`, which needs it to be
    /// both a shader resource and a render target.
    generate_mips: bool = false,
    shared: bool = false,
    texture_cube: bool = false,
    _reserved: u29 = 0,
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
    misc: TextureMisc = .{},
};

const Texture3DDesc = extern struct {
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32 = 1,
    format: Format,
    usage: Usage = .default,
    bind: BindFlags = .{},
    cpu_access: CpuAccess = .{},
    misc: TextureMisc = .{},
};

/// `D3D11_SRV_DIMENSION`, the ones a texture can be seen as.
const SrvDimension = enum(u32) {
    unknown = 0,
    texture2d = 4,
    texture2d_array = 5,
    texture2d_ms = 6,
    texture3d = 8,
    texture_cube = 9,
};

/// `D3D11_RTV_DIMENSION`.
const RtvDimension = enum(u32) {
    unknown = 0,
    texture2d = 4,
    texture2d_array = 5,
    texture2d_ms = 6,
    texture3d = 8,
};

/// `D3D11_DSV_DIMENSION`. There is no volume: a depth buffer is never 3D.
const DsvDimension = enum(u32) {
    unknown = 0,
    texture2d = 3,
    texture2d_array = 4,
    texture2d_ms = 5,
};

// The three view descriptions are a format, a dimension and a union of one
// small struct per dimension in the headers. Every member of those unions is a
// prefix of the same four (or three) integers, and the dimension says how many
// of them count, so each is declared here as the integers with the meaning of
// the widest member, and a dimension that has fewer ignores the rest.

/// `D3D11_SHADER_RESOURCE_VIEW_DESC`: `Texture2D`, `Texture3D` and
/// `TextureCube` read the first two fields, `Texture2DArray` all four.
const ShaderResourceViewDesc = extern struct {
    format: Format,
    dimension: SrvDimension,
    most_detailed_mip: u32 = 0,
    mip_levels: u32 = 1,
    first_array_slice: u32 = 0,
    array_size: u32 = 1,
};

/// `D3D11_RENDER_TARGET_VIEW_DESC`: `Texture2D` reads `mip_slice`;
/// `Texture2DArray` and `Texture3D` (whose `FirstWSlice` and `WSize` sit in
/// the same places) read all three.
const RenderTargetViewDesc = extern struct {
    format: Format,
    dimension: RtvDimension,
    mip_slice: u32 = 0,
    first_slice: u32 = 0,
    slice_count: u32 = 1,
};

/// `D3D11_DEPTH_STENCIL_VIEW_DESC`.
const DepthStencilViewDesc = extern struct {
    format: Format,
    dimension: DsvDimension,
    flags: u32 = 0,
    mip_slice: u32 = 0,
    first_slice: u32 = 0,
    slice_count: u32 = 1,
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

/// `D3D11_COLOR_WRITE_ENABLE_ALL`.
const color_write_all: u8 = 0x0F;

const RenderTargetBlendDesc = extern struct {
    blend_enable: Bool = 0,
    src_blend: Blend = .one,
    dest_blend: Blend = .zero,
    blend_op: BlendOp = .add,
    src_blend_alpha: Blend = .one,
    dest_blend_alpha: Blend = .zero,
    blend_op_alpha: BlendOp = .add,
    render_target_write_mask: u8 = color_write_all,
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
const ITexture3D = DeviceChild("{037E866E-F56D-4357-A8AF-9DABBE6E250E}");
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
        CopySubresourceRegion: *const fn (*IDeviceContext, *IResource, u32, u32, u32, u32, *IResource, u32, ?*const Box) callconv(.winapi) void,
        CopyResource: *const fn (*IDeviceContext, *IResource, *IResource) callconv(.winapi) void,
        UpdateSubresource: *const fn (*IDeviceContext, *IResource, u32, ?*const Box, *const anyopaque, u32, u32) callconv(.winapi) void,
        CopyStructureCount: *const anyopaque,
        ClearRenderTargetView: *const fn (*IDeviceContext, *IRenderTargetView, *const [4]f32) callconv(.winapi) void,
        ClearUnorderedAccessViewUint: *const anyopaque,
        ClearUnorderedAccessViewFloat: *const anyopaque,
        ClearDepthStencilView: *const fn (*IDeviceContext, *IDepthStencilView, ClearFlags, f32, u8) callconv(.winapi) void,
        GenerateMips: *const fn (*IDeviceContext, *IShaderResourceView) callconv(.winapi) void,
        SetResourceMinLOD: *const anyopaque,
        GetResourceMinLOD: *const anyopaque,
        ResolveSubresource: *const fn (*IDeviceContext, *IResource, u32, *IResource, u32, Format) callconv(.winapi) void,
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

    fn createTexture2D(self: Raw, desc: Texture2DDesc) d3d.Error!*ITexture2D {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const Texture2DDesc, ?*const SubresourceData, *?*ITexture2D) callconv(.winapi) Hresult, self.device.vtable.CreateTexture2D);
        var texture: ?*ITexture2D = null;
        return com.received(ITexture2D, create(self.device, &desc, null, &texture), texture);
    }

    fn createTexture3D(self: Raw, desc: Texture3DDesc) d3d.Error!*ITexture3D {
        const create = slot(*const fn (*d3d11.ID3D11Device, *const Texture3DDesc, ?*const SubresourceData, *?*ITexture3D) callconv(.winapi) Hresult, self.device.vtable.CreateTexture3D);
        var texture: ?*ITexture3D = null;
        return com.received(ITexture3D, create(self.device, &desc, null, &texture), texture);
    }

    /// With no description the view is of the whole resource, in the
    /// resource's own format: what a swap chain's back buffer wants.
    fn createShaderResourceView(self: Raw, resource: *IResource, desc: ?*const ShaderResourceViewDesc) d3d.Error!*IShaderResourceView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const ShaderResourceViewDesc, *?*IShaderResourceView) callconv(.winapi) Hresult, self.device.vtable.CreateShaderResourceView);
        var view: ?*IShaderResourceView = null;
        return com.received(IShaderResourceView, create(self.device, resource, desc, &view), view);
    }

    fn createRenderTargetView(self: Raw, resource: *IResource, desc: ?*const RenderTargetViewDesc) d3d.Error!*IRenderTargetView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const RenderTargetViewDesc, *?*IRenderTargetView) callconv(.winapi) Hresult, self.device.vtable.CreateRenderTargetView);
        var view: ?*IRenderTargetView = null;
        return com.received(IRenderTargetView, create(self.device, resource, desc, &view), view);
    }

    fn createDepthStencilView(self: Raw, resource: *IResource, desc: ?*const DepthStencilViewDesc) d3d.Error!*IDepthStencilView {
        const create = slot(*const fn (*d3d11.ID3D11Device, *IResource, ?*const DepthStencilViewDesc, *?*IDepthStencilView) callconv(.winapi) Hresult, self.device.vtable.CreateDepthStencilView);
        var view: ?*IDepthStencilView = null;
        return com.received(IDepthStencilView, create(self.device, resource, desc, &view), view);
    }

    /// What the device can do with a format, as the runtime says it. A format
    /// the runtime does not know at this feature level is `E_FAIL`, which is
    /// an honest "nothing".
    fn checkFormatSupport(self: Raw, format: Format) d3d11.FormatSupport {
        var bits: d3d11.FormatSupport = .{};
        self.device.vtable.CheckFormatSupport(self.device, @intFromEnum(format), &bits).check() catch return .{};
        return bits;
    }

    /// How many quality levels a multisampled texture of this format and
    /// sample count has; zero is "not at all".
    fn checkMultisampleQualityLevels(self: Raw, format: Format, samples: u32) u32 {
        const check = slot(*const fn (*d3d11.ID3D11Device, Format, u32, *u32) callconv(.winapi) Hresult, self.device.vtable.CheckMultisampleQualityLevels);
        var levels: u32 = 0;
        check(self.device, format, samples, &levels).check() catch return 0;
        return levels;
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
    /// What the runtime said about every format, asked once when the device
    /// opened. `caps` is computed from it and `createTexture` consults it.
    probes: std.EnumArray(types.Format, Probe),

    // Per-submit state.
    pipeline: ?*PipelineRes = null,
    target_width: u32 = 0,
    target_height: u32 = 0,
    vertex_bindings: [max_vertex_slots]VertexBinding = @splat(.{}),
    bindings_dirty: bool = false,
    /// What `end_pass` resolves: one entry per attachment of the pass that
    /// asked for it.
    resolves: [fl11.max_color_attachments]Resolve = undefined,
    resolve_count: u32 = 0,
};

/// One multisampled attachment and where its samples go. Both sides are the
/// first subresource: `RenderTarget` names a texture or a surface, never a
/// level or a layer of one.
const Resolve = struct {
    source: *IResource,
    destination: *IResource,
    /// The format the samples are averaged in: the attachment's own, seen as a
    /// colour, which is what the runtime insists on for a typeless resource.
    format: Format,
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
    /// An `ID3D11Texture2D` - which is also what a cube and an array are - or
    /// an `ID3D11Texture3D`. `dimension` says which.
    resource: *IResource,
    dimension: types.Dimension,
    width: u32,
    height: u32,
    /// The depth of a volume, the layers of an array, six for a cube, one
    /// otherwise: as many `layer` values as level zero has.
    depth_or_layers: u32,
    mip_levels: u32,
    samples: u32,
    format: types.Format,
    native: Native,
    /// The whole texture as a shader sees it, in the kind its dimension is:
    /// there for a texture that can be sampled or that `GenerateMips` fills.
    srv: ?*IShaderResourceView,
    /// A view of one subresource for a pass to draw into, made the first
    /// time a pass names it. Colour and depth alike: a texture is one or the
    /// other.
    targets: std.ArrayList(TargetView) = .empty,
};

const TargetView = struct {
    mip: u32,
    layer: u32,
    /// An `IRenderTargetView` or an `IDepthStencilView`, which the format
    /// of the texture decides. Both are device children, and that is all
    /// a release needs.
    view: *IRenderTargetView,
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
    /// Buffer zero, which is the one a flip-model swap chain always draws
    /// into. Kept for `ResolveSubresource`, which wants the resource and not a
    /// view of it.
    back_buffer: *ITexture2D,
    rtv: *IRenderTargetView,
    width: u32,
    height: u32,
};

/// What a swap chain is made in, and so the one format a multisampled texture
/// can be resolved into the surface from.
const surface_format: types.Format = .rgba8_unorm;

pub fn open(gpa: Allocator, desc: types.DeviceDesc) Error!backend.Opened {
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

    const raw: Raw = .{ .device = device.device };
    var probes: std.EnumArray(types.Format, Probe) = .initFill(.{});
    for (std.enums.values(types.Format)) |format| probes.set(format, probeFormat(raw, format));

    const self = try gpa.create(D3d);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .library = library,
        .dxgi_library = dxgi_library,
        .factory = factory,
        .device = device,
        .raw = raw,
        .context = @ptrCast(device.context),
        .debug = desc.debug,
        .probes = probes,
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
    .caps = caps,
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
// Formats
// -------------------------------------------------------------------------

/// How a texel becomes RGBA8 in `readTexture`: one kernel for each way a texel
/// can be packed, whatever the format is called. What a format has in each
/// channel and how many channels it has is `types.Format.info`'s to say.
const Decode = enum { unorm8, bgra8, float16, float32, rgb10a2, rg11b10_float };

/// What Direct3D calls a `types.Format`.
const Native = struct {
    /// What the resource is made in. Typeless for depth, which is written as
    /// one type and read as another.
    resource: Format,
    /// What a shader resource view says the texels are: for depth, the colour
    /// format that reads its bits.
    srv: Format,
    /// What a render target view or a depth-stencil view writes.
    target: Format,
    /// How `readTexture` unpacks it, or null for what is never read as
    /// colour: depth, and what is compressed.
    decode: ?Decode = null,
};

/// A format that is the same thing however it is looked at.
fn plain(format: Format, decode: Decode) Native {
    return .{ .resource = format, .srv = format, .target = format, .decode = decode };
}

/// A block-compressed format: only ever a shader resource, so there is one view
/// of it and it is the resource's own.
fn blocks(format: Format) Native {
    return .{ .resource = format, .srv = format, .target = format };
}

/// A depth format, made typeless so that it can be both the depth attachment
/// of one pass and the shadow map the next one reads.
fn deep(resource: Format, srv: Format, target: Format) Native {
    return .{ .resource = resource, .srv = srv, .target = target };
}

const NativeRow = struct { format: types.Format, native: ?Native };

/// One row per `types.Format`, in any order, and `null` for what this API
/// cannot do: ETC2 and ASTC are OpenGL ES's and the phones', and no Direct3D
/// 11 device has them. A format without a row does not compile, and neither
/// does one with two.
const native_rows = [_]NativeRow{
    .{ .format = .r8_unorm, .native = plain(.r8_unorm, .unorm8) },
    .{ .format = .rg8_unorm, .native = plain(.r8g8_unorm, .unorm8) },
    .{ .format = .rgba8_unorm, .native = plain(.r8g8b8a8_unorm, .unorm8) },
    .{ .format = .rgba8_unorm_srgb, .native = plain(.r8g8b8a8_unorm_srgb, .unorm8) },
    .{ .format = .bgra8_unorm, .native = plain(.b8g8r8a8_unorm, .bgra8) },
    .{ .format = .bgra8_unorm_srgb, .native = plain(.b8g8r8a8_unorm_srgb, .bgra8) },
    .{ .format = .r16_float, .native = plain(.r16_float, .float16) },
    .{ .format = .rg16_float, .native = plain(.r16g16_float, .float16) },
    .{ .format = .rgba16_float, .native = plain(.r16g16b16a16_float, .float16) },
    .{ .format = .r32_float, .native = plain(.r32_float, .float32) },
    .{ .format = .rg32_float, .native = plain(.r32g32_float, .float32) },
    .{ .format = .rgba32_float, .native = plain(.r32g32b32a32_float, .float32) },
    .{ .format = .rgb10a2_unorm, .native = plain(.r10g10b10a2_unorm, .rgb10a2) },
    .{ .format = .rg11b10_float, .native = plain(.r11g11b10_float, .rg11b10_float) },

    .{ .format = .depth16_unorm, .native = deep(.r16_typeless, .r16_unorm, .d16_unorm) },
    .{ .format = .depth24_stencil8, .native = deep(.r24g8_typeless, .r24_unorm_x8_typeless, .d24_unorm_s8_uint) },
    .{ .format = .depth32_float, .native = deep(.r32_typeless, .r32_float, .d32_float) },
    .{ .format = .depth32_float_stencil8, .native = deep(.r32g8x24_typeless, .r32_float_x8x24_typeless, .d32_float_s8x24_uint) },

    .{ .format = .bc1_rgba_unorm, .native = blocks(.bc1_unorm) },
    .{ .format = .bc1_rgba_unorm_srgb, .native = blocks(.bc1_unorm_srgb) },
    .{ .format = .bc3_rgba_unorm, .native = blocks(.bc3_unorm) },
    .{ .format = .bc3_rgba_unorm_srgb, .native = blocks(.bc3_unorm_srgb) },
    .{ .format = .bc4_r_unorm, .native = blocks(.bc4_unorm) },
    .{ .format = .bc5_rg_unorm, .native = blocks(.bc5_unorm) },
    .{ .format = .bc6h_rgb_ufloat, .native = blocks(.bc6h_uf16) },
    .{ .format = .bc7_rgba_unorm, .native = blocks(.bc7_unorm) },
    .{ .format = .bc7_rgba_unorm_srgb, .native = blocks(.bc7_unorm_srgb) },

    .{ .format = .etc2_rgb8_unorm, .native = null },
    .{ .format = .etc2_rgb8_unorm_srgb, .native = null },
    .{ .format = .etc2_rgba8_unorm, .native = null },
    .{ .format = .etc2_rgba8_unorm_srgb, .native = null },
    .{ .format = .astc_4x4_unorm, .native = null },
    .{ .format = .astc_4x4_unorm_srgb, .native = null },
    .{ .format = .astc_6x6_unorm, .native = null },
    .{ .format = .astc_6x6_unorm_srgb, .native = null },
    .{ .format = .astc_8x8_unorm, .native = null },
    .{ .format = .astc_8x8_unorm_srgb, .native = null },
};

const native_table: std.EnumArray(types.Format, ?Native) = blk: {
    var table: std.EnumArray(types.Format, ?Native) = .initUndefined();
    var seen: std.EnumSet(types.Format) = .initEmpty();
    for (native_rows) |row| {
        if (seen.contains(row.format)) @compileError("native_table: two rows for " ++ @tagName(row.format));
        seen.insert(row.format);
        table.set(row.format, row.native);
    }
    for (std.enums.values(types.Format)) |format| {
        if (!seen.contains(format)) @compileError("native_table: no row for " ++ @tagName(format));
    }
    break :blk table;
};

/// The limits Direct3D 11 promises at feature level 11.0, which are the ones
/// `open` asks for and 11.1 does not change. The specification states them by
/// feature level rather than letting a device be asked: each is the `D3D11_REQ_*`
/// or `D3D11_MAX_*` constant in `d3d11.h` named beside it.
const fl11 = struct {
    /// `D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION`
    const max_texture_2d = 16384;
    /// `D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION`
    const max_texture_3d = 2048;
    /// `D3D11_REQ_TEXTURECUBE_DIMENSION`
    const max_texture_cube = 16384;
    /// `D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION`
    const max_texture_layers = 2048;
    /// `D3D11_MAX_MAXANISOTROPY`
    const max_anisotropy = 16;
    /// `D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT`
    const max_color_attachments = 8;
    /// `D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT`, the largest count worth asking a
    /// device about.
    const max_samples = 32;
};

/// What the runtime said about one format, in its own words, asked once: the
/// view a shader reads through and the view a pass writes through are asked
/// separately because for depth they are different formats.
const Probe = struct {
    /// `CheckFormatSupport` of `Native.srv`.
    srv: d3d11.FormatSupport = .{},
    /// `CheckFormatSupport` of `Native.target`.
    target: d3d11.FormatSupport = .{},
    /// A bit for each power of two above one that `CheckMultisampleQualityLevels`
    /// gives a level for, in `Native.target`.
    multisample: u8 = 0,
};

fn probeFormat(raw: Raw, format: types.Format) Probe {
    const native = native_table.get(format) orelse return .{};
    var probe: Probe = .{
        .srv = raw.checkFormatSupport(native.srv),
        .target = raw.checkFormatSupport(native.target),
    };
    var samples: u32 = 2;
    while (samples <= fl11.max_samples) : (samples *= 2) {
        if (raw.checkMultisampleQualityLevels(native.target, samples) > 0) probe.multisample |= @as(u8, 1) << @intCast(std.math.log2_int(u32, samples));
    }
    return probe;
}

/// What a `Probe` means for one `FormatSupport`. A claim is only ever made
/// from a bit the runtime set, and never for a format the table has no row for.
fn deriveSupport(format: types.Format, native: ?Native, probe: Probe) types.FormatSupport {
    if (native == null) return .{};
    const depth = format.isDepth();

    // A shader `Sample` needs the runtime's `SHADER_SAMPLE`, which is the
    // one bit that says the format can be read through a sampler; a format
    // that can only be `Load`ed is not offered.
    const sampled = probe.srv.texture2d and probe.srv.shader_sample;
    const render_target = probe.target.texture2d and (if (depth) probe.target.depth_stencil else probe.target.render_target);

    var support: types.FormatSupport = .{
        .sampled = sampled,
        .filterable = sampled,
        .render_target = render_target,
        .blendable = render_target and !depth and probe.target.blendable,
        // `GenerateMips` fills the levels of a texture that is a shader
        // resource and a render target both, so all three bits have to be there.
        .generate_mips = sampled and render_target and !depth and !format.isCompressed() and probe.target.mip_autogen,
        .sample_counts = if (sampled or render_target) 0b1 else 0,
    };
    // A multisampled colour target is only worth having if it can be resolved.
    if (render_target and (depth or (probe.target.multisample_render_target and probe.target.multisample_resolve))) {
        support.sample_counts |= probe.multisample;
    }
    // Which shapes: the runtime's own bits for a resource of that dimension,
    // asked of the views a shader makes and, for what is drawn into, of the
    // ones a pass makes. It is what leaves a volume of depth out.
    if (support.sampled or support.render_target) for (std.enums.values(types.Dimension)) |dimension| {
        const need: u32 = @bitCast(dimension_support.get(dimension));
        const readable = @as(u32, @bitCast(probe.srv)) & need == need;
        const drawable = !render_target or @as(u32, @bitCast(probe.target)) & need == need;
        if (!readable or !drawable) support.dimensions.remove(dimension);
    };
    return support;
}

/// What a format has to be able to do for a texture of this shape to be made
/// in it.
const dimension_support = std.EnumArray(types.Dimension, d3d11.FormatSupport).init(.{
    .d2 = .{ .texture2d = true },
    .d2_array = .{ .texture2d = true },
    .cube = .{ .texture2d = true, .texturecube = true },
    .d3 = .{ .texture3d = true },
});

/// Whether the resource and the views a pass and a shader make of it can all
/// be this shape. It is what keeps a volume of depth out: the depth-stencil
/// format has no `TEXTURE3D`, and there is no such resource to make.
fn supportsDimension(probe: Probe, dimension: types.Dimension) bool {
    const need: u32 = @bitCast(dimension_support.get(dimension));
    const srv: u32 = @bitCast(probe.srv);
    const target: u32 = @bitCast(probe.target);
    return srv & need == need and target & need == need;
}

fn caps(impl: backend.Impl) types.Caps {
    const self = cast(impl);
    var answer: types.Caps = .{
        .limits = .{
            .max_texture_2d = fl11.max_texture_2d,
            .max_texture_3d = fl11.max_texture_3d,
            .max_texture_cube = fl11.max_texture_cube,
            .max_texture_layers = fl11.max_texture_layers,
            .max_anisotropy = fl11.max_anisotropy,
            .max_color_attachments = fl11.max_color_attachments,
        },
        // Border addressing and a bias on the level are part of the sampler
        // state at every Direct3D 11 feature level.
        .features = .{ .sampler_border = true, .sampler_lod_bias = true },
    };
    for (std.enums.values(types.Format)) |format| {
        answer.formats.set(format, deriveSupport(format, native_table.get(format), self.probes.get(format)));
    }
    return answer;
}

// -------------------------------------------------------------------------
// Reading pixels back
// -------------------------------------------------------------------------

/// Reads the first `channels` values of a texel, in the order they are stored,
/// each as eight bits.
const Kernel = *const fn (texel: []const u8, channels: u8) [4]u8;

const kernels = std.EnumArray(Decode, Kernel).init(.{
    .unorm8 = decodeUnorm8,
    .bgra8 = decodeBgra8,
    .float16 = decodeFloat16,
    .float32 = decodeFloat32,
    .rgb10a2 = decodeRgb10a2,
    .rg11b10_float = decodeRg11b10,
});

/// Which stored value goes to red, green, blue and alpha, for a format that has
/// this many channels: one is grey, two are red and green, and a format that
/// has no alpha is opaque.
const Lane = enum { c0, c1, c2, c3, zero, one };
const lanes = [_][4]Lane{
    .{ .zero, .zero, .zero, .one }, // no channels: never asked
    .{ .c0, .c0, .c0, .one },
    .{ .c0, .c1, .zero, .one },
    .{ .c0, .c1, .c2, .one },
    .{ .c0, .c1, .c2, .c3 },
};

fn pick(lane: Lane, values: [4]u8) u8 {
    return switch (lane) {
        .zero => 0,
        .one => 255,
        .c0, .c1, .c2, .c3 => values[@intFromEnum(lane)],
    };
}

fn unormFromFloat(value: f32) u8 {
    // NaN is what `@max` drops, so it reads as zero.
    return @intFromFloat(@round(@min(@max(value, 0), 1) * 255));
}

fn decodeUnorm8(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| values[c] = texel[c];
    return values;
}

fn decodeBgra8(texel: []const u8, channels: u8) [4]u8 {
    var values = decodeUnorm8(texel, channels);
    std.mem.swap(u8, &values[0], &values[2]);
    return values;
}

fn decodeFloat16(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| {
        const half: f16 = @bitCast(std.mem.readInt(u16, texel[c * 2 ..][0..2], .little));
        values[c] = unormFromFloat(half);
    }
    return values;
}

fn decodeFloat32(texel: []const u8, channels: u8) [4]u8 {
    var values: [4]u8 = @splat(0);
    for (0..channels) |c| {
        const single: f32 = @bitCast(std.mem.readInt(u32, texel[c * 4 ..][0..4], .little));
        values[c] = unormFromFloat(single);
    }
    return values;
}

fn decodeRgb10a2(texel: []const u8, channels: u8) [4]u8 {
    _ = channels;
    const packed_texel = std.mem.readInt(u32, texel[0..4], .little);
    const ten_bit_max = (1 << 10) - 1;
    const two_bit_max = (1 << 2) - 1;
    var values: [4]u8 = undefined;
    inline for (0..3) |c| {
        const raw: u32 = (packed_texel >> (c * 10)) & ten_bit_max;
        values[c] = @intCast((raw * 255 + ten_bit_max / 2) / ten_bit_max);
    }
    const alpha: u32 = packed_texel >> 30;
    values[3] = @intCast((alpha * 255 + two_bit_max / 2) / two_bit_max);
    return values;
}

/// The unsigned small floats of `R11G11B10_FLOAT`: five bits of exponent, biased
/// by 15, and what is left is mantissa, with the special values IEEE gives them.
fn smallFloat(bits: u32, comptime mantissa_bits: u5) f32 {
    const exponent_bias = 15;
    const exponent_all_ones = 31;
    const exponent = bits >> mantissa_bits;
    const mantissa: f32 = @floatFromInt(bits & ((1 << mantissa_bits) - 1));
    const scale: f32 = @floatFromInt(@as(u32, 1) << mantissa_bits);
    if (exponent == 0) return mantissa / scale * @exp2(@as(f32, 1 - exponent_bias));
    if (exponent == exponent_all_ones) return if (mantissa == 0) std.math.inf(f32) else std.math.nan(f32);
    return (1 + mantissa / scale) * @exp2(@as(f32, @floatFromInt(@as(i32, @intCast(exponent)) - exponent_bias)));
}

fn decodeRg11b10(texel: []const u8, channels: u8) [4]u8 {
    _ = channels;
    const packed_texel = std.mem.readInt(u32, texel[0..4], .little);
    const red_and_green_bits = 11;
    const red_mantissa = 6;
    const blue_mantissa = 5;
    const field = (1 << red_and_green_bits) - 1;
    return .{
        unormFromFloat(smallFloat(packed_texel & field, red_mantissa)),
        unormFromFloat(smallFloat((packed_texel >> red_and_green_bits) & field, red_mantissa)),
        unormFromFloat(smallFloat(packed_texel >> (2 * red_and_green_bits), blue_mantissa)),
        0,
    };
}

// -------------------------------------------------------------------------
// Textures
// -------------------------------------------------------------------------

/// How each shape of texture is seen by a shader and by a pass. A cube is a
/// `TextureCube` to a shader, and a pass draws into one face of it as it does
/// into one layer of an array. A multisampled texture is the same whatever its
/// shape, and cannot be seen by a shader at all.
const ViewDimensions = struct { srv: SrvDimension, rtv: RtvDimension, dsv: DsvDimension };

const view_dimensions = std.EnumArray(types.Dimension, ViewDimensions).init(.{
    .d2 = .{ .srv = .texture2d, .rtv = .texture2d, .dsv = .texture2d },
    .d2_array = .{ .srv = .texture2d_array, .rtv = .texture2d_array, .dsv = .texture2d_array },
    .cube = .{ .srv = .texture_cube, .rtv = .texture2d_array, .dsv = .texture2d_array },
    .d3 = .{ .srv = .texture3d, .rtv = .texture3d, .dsv = .unknown },
});

const multisampled_dimensions: ViewDimensions = .{ .srv = .unknown, .rtv = .texture2d_ms, .dsv = .texture2d_ms };

fn viewDimensions(res: *const TextureRes) ViewDimensions {
    return if (res.samples > 1) multisampled_dimensions else view_dimensions.get(res.dimension);
}

/// `D3D11CalcSubresource`. A volume is one array element whatever its depth,
/// so a slice of it is not part of the index.
fn subresourceIndex(res: *const TextureRes, mip: u32, layer: u32) u32 {
    return mip + if (res.dimension == .d3) 0 else layer * res.mip_levels;
}

fn createTexture(impl: backend.Impl, desc: types.TextureDesc) Error!backend.Native {
    const self = cast(impl);

    // `Device` has asked `caps` about the format; the shape is the one thing
    // it cannot ask, and a format that has no volume or cube is said here.
    const native = native_table.get(desc.format) orelse return error.Unsupported;
    const probe = self.probes.get(desc.format);
    if (!supportsDimension(probe, desc.dimension)) return error.Unsupported;
    // Direct3D 11 makes a block-compressed texture only in a whole number of
    // blocks at level zero - the levels below it may shrink past one - which
    // OpenGL and Vulkan do not ask for, so it is this device that cannot.
    const row = desc.format.info();
    if (desc.format.isCompressed() and (desc.width % row.block_width != 0 or desc.height % row.block_height != 0)) return error.Unsupported;
    const support = deriveSupport(desc.format, native, probe);

    const res = try self.gpa.create(TextureRes);
    errdefer self.gpa.destroy(res);

    const depth = desc.format.isDepth();
    const multisampled = desc.samples > 1;
    // Levels that the hardware can fill are made so that it can, whatever the
    // caller said the texture was for: `generateMips` is a command, and this is
    // what it needs. It costs a render target bind on a texture nobody draws
    // into, and the caller never sees it.
    const autogen = desc.mip_levels > 1 and support.generate_mips;
    const bind: BindFlags = .{
        .shader_resource = !multisampled and (desc.usage.sampled or autogen),
        .render_target = !depth and (desc.usage.render_target or autogen),
        .depth_stencil = depth and desc.usage.render_target,
    };

    const resource: *IResource = switch (desc.dimension) {
        .d3 => asResource(self.raw.createTexture3D(.{
            .width = desc.width,
            .height = desc.height,
            .depth = desc.depth_or_layers,
            .mip_levels = desc.mip_levels,
            .format = native.resource,
            .bind = bind,
            .misc = .{ .generate_mips = autogen },
        }) catch return error.Failed),
        .d2, .d2_array, .cube => asResource(self.raw.createTexture2D(.{
            .width = desc.width,
            .height = desc.height,
            .mip_levels = desc.mip_levels,
            .array_size = desc.layers(),
            .format = native.resource,
            .sample = .{ .count = desc.samples },
            .bind = bind,
            .misc = .{ .generate_mips = autogen, .texture_cube = desc.dimension == .cube },
        }) catch return error.Failed),
    };
    errdefer _ = com.release(resource);

    res.* = .{
        .resource = resource,
        .dimension = desc.dimension,
        .width = desc.width,
        .height = desc.height,
        .depth_or_layers = switch (desc.dimension) {
            .d2 => 1,
            .cube => 6,
            .d3, .d2_array => desc.depth_or_layers,
        },
        .mip_levels = desc.mip_levels,
        .samples = desc.samples,
        .format = desc.format,
        .native = native,
        .srv = null,
    };
    if (bind.shader_resource) {
        const dimensions = viewDimensions(res);
        res.srv = self.raw.createShaderResourceView(resource, &.{
            .format = native.srv,
            .dimension = dimensions.srv,
            .mip_levels = desc.mip_levels,
            .array_size = res.depth_or_layers,
        }) catch return error.Failed;
    }
    errdefer if (res.srv) |view| {
        _ = com.release(view);
    };

    // The texels: level zero of everything, layer after layer. A resource made
    // with no initial data and then filled is what a texture that will have
    // a chain generated wants, and it is one path for all of them.
    if (desc.data) |data| {
        const pitch = desc.effectiveRowPitch();
        try writeTexture(impl, res, .{
            .width = desc.width,
            .height = desc.height,
            .depth = res.depth_or_layers,
        }, data, pitch, pitch * desc.format.rowCount(desc.height));
    }
    return res;
}

fn destroyTexture(impl: backend.Impl, native: backend.Native) void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    if (res.srv) |v| _ = com.release(v);
    for (res.targets.items) |target| _ = com.release(target.view);
    res.targets.deinit(self.gpa);
    _ = com.release(res.resource);
    self.gpa.destroy(res);
}

/// A view of one level of one layer, face or slice for a pass to draw into. A
/// depth texture gets a depth-stencil view and anything else a render target
/// view, and they are kept: a texture drawn into every frame does not make one
/// every frame.
fn targetView(self: *D3d, res: *TextureRes, mip: u32, layer: u32) Error!*IRenderTargetView {
    for (res.targets.items) |target| {
        if (target.mip == mip and target.layer == layer) return target.view;
    }
    const dimensions = viewDimensions(res);
    const view: *IRenderTargetView = if (res.format.isDepth()) depth: {
        const view = self.raw.createDepthStencilView(res.resource, &.{
            .format = res.native.target,
            .dimension = dimensions.dsv,
            .mip_slice = mip,
            .first_slice = layer,
        }) catch return error.Failed;
        break :depth @ptrCast(view);
    } else self.raw.createRenderTargetView(res.resource, &.{
        .format = res.native.target,
        .dimension = dimensions.rtv,
        .mip_slice = mip,
        .first_slice = layer,
    }) catch return error.Failed;
    errdefer _ = com.release(view);
    try res.targets.append(self.gpa, .{ .mip = mip, .layer = layer, .view = view });
    return view;
}

/// `size` rounded up to a whole number of blocks; a block is one texel wide
/// for a format that is not compressed.
fn roundUp(size: u32, block: u8) u32 {
    return (std.math.divCeil(u32, size, block) catch unreachable) * block;
}

fn writeTexture(impl: backend.Impl, native: backend.Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const row = res.format.info();
    const level_width = types.mipExtent(res.width, region.mip);
    const level_height = types.mipExtent(res.height, region.mip);
    const level_slices = if (res.dimension == .d3) types.mipExtent(res.depth_or_layers, region.mip) else res.depth_or_layers;

    // A box is in texels, and a compressed one is in whole blocks: `Device`
    // has seen to the start, and the end is rounded up because the last block of
    // a level that is not a multiple of one hangs over its edge and is written
    // whole - the runtime counts the size of a level in blocks.
    const right = roundUp(region.x + region.width, row.block_width);
    const bottom = roundUp(region.y + region.height, row.block_height);
    // The whole of a subresource is what a null box says, and needs no
    // arithmetic about that overhang.
    const whole_level = region.x == 0 and region.y == 0 and right >= level_width and bottom >= level_height;

    if (res.dimension == .d3) {
        const box: Box = .{ .left = region.x, .top = region.y, .front = region.z, .right = right, .bottom = bottom, .back = region.z + region.depth };
        const whole = whole_level and region.z == 0 and region.depth == level_slices;
        self.context.vtable.UpdateSubresource(self.context, res.resource, subresourceIndex(res, region.mip, 0), if (whole) null else &box, bytes.ptr, @intCast(row_pitch), @intCast(slice_pitch));
        return;
    }
    // Layers and faces are subresources of their own, one call each.
    for (0..region.depth) |i| {
        const box: Box = .{ .left = region.x, .top = region.y, .right = right, .bottom = bottom };
        self.context.vtable.UpdateSubresource(self.context, res.resource, subresourceIndex(res, region.mip, region.z + @as(u32, @intCast(i))), if (whole_level) null else &box, bytes[i * slice_pitch ..].ptr, @intCast(row_pitch), 0);
    }
}

fn readTexture(impl: backend.Impl, native: backend.Native, sub: types.Subresource, gpa: Allocator) Error![]u8 {
    const self = cast(impl);
    const res = as(TextureRes, native);
    const context = self.context;

    const decode = res.native.decode orelse return error.Unsupported;
    const kernel = kernels.get(decode);
    const row = res.format.info();
    const width = types.mipExtent(res.width, sub.mip);
    const height = types.mipExtent(res.height, sub.mip);

    // The pipeline cannot read a staging resource and the CPU cannot read
    // anything else, so two textures and a copy. There is no shorter way. The
    // staging one is the size of the image asked for and nothing else: a
    // volume's is a volume one slice deep, because a copy cannot change what
    // kind of resource it is between.
    const staging: *IResource = if (res.dimension == .d3)
        asResource(self.raw.createTexture3D(.{
            .width = width,
            .height = height,
            .depth = 1,
            .format = res.native.resource,
            .usage = .staging,
            .cpu_access = .{ .read = true },
        }) catch return error.Failed)
    else
        asResource(self.raw.createTexture2D(.{
            .width = width,
            .height = height,
            .format = res.native.resource,
            .usage = .staging,
            .cpu_access = .{ .read = true },
        }) catch return error.Failed);
    defer _ = com.release(staging);

    if (res.dimension == .d3) {
        const slice: Box = .{ .left = 0, .top = 0, .front = sub.layer, .right = width, .bottom = height, .back = sub.layer + 1 };
        context.vtable.CopySubresourceRegion(context, staging, 0, 0, 0, 0, res.resource, subresourceIndex(res, sub.mip, 0), &slice);
    } else {
        context.vtable.CopySubresourceRegion(context, staging, 0, 0, 0, 0, res.resource, subresourceIndex(res, sub.mip, sub.layer), null);
    }

    var mapped: MappedSubresource = .{};
    context.vtable.Map(context, staging, 0, .read, 0, &mapped).check() catch return error.Failed;
    defer context.vtable.Unmap(context, staging, 0);
    const data = mapped.data orelse return error.Failed;

    const texel: usize = row.block_bytes;
    const out_row = @as(usize, width) * 4;
    const pixels = try gpa.alloc(u8, out_row * height);
    errdefer gpa.free(pixels);

    const layout = lanes[row.channels];
    for (0..height) |y| {
        const source = data[y * mapped.row_pitch ..][0 .. width * texel];
        const destination = pixels[y * out_row ..][0..out_row];
        for (0..width) |x| {
            const values = kernel(source[x * texel ..][0..texel], row.channels);
            for (layout, 0..) |lane, i| destination[x * 4 + i] = pick(lane, values);
        }
    }
    return pixels;
}

// -------------------------------------------------------------------------
// Samplers
// -------------------------------------------------------------------------

const min_filter_bits = std.EnumArray(types.Filter, u32).init(.{ .nearest = 0, .linear = filter_bits.min_linear });
const mag_filter_bits = std.EnumArray(types.Filter, u32).init(.{ .nearest = 0, .linear = filter_bits.mag_linear });
const mip_filter_bits = std.EnumArray(types.MipFilter, u32).init(.{ .none = 0, .nearest = 0, .linear = filter_bits.mip_linear });

/// `D3D11_ENCODE_BASIC_FILTER` and its comparison and anisotropic relatives.
/// Anisotropy is asked for by a number and given only to a sampler whose
/// magnification and minification are both linear: it is a way of sampling
/// with a linear filter along a stretched footprint, and a caller who asked for
/// `nearest` gets `nearest`.
fn samplerFilter(desc: types.SamplerDesc) FilterMode {
    const comparison_bit: u32 = if (desc.compare != null) filter_bits.comparison else 0;
    const linear = desc.min_filter == .linear and desc.mag_filter == .linear;
    if (desc.max_anisotropy > 1 and linear) return @enumFromInt(@intFromEnum(FilterMode.anisotropic) | comparison_bit);
    return @enumFromInt(min_filter_bits.get(desc.min_filter) | mag_filter_bits.get(desc.mag_filter) | mip_filter_bits.get(desc.mip_filter) | comparison_bit);
}

const address_modes = std.EnumArray(types.Wrap, AddressMode).init(.{
    .repeat = .wrap,
    .clamp_to_edge = .clamp,
    .mirror = .mirror,
    .border = .border,
});

const border_colors = std.EnumArray(types.BorderColor, [4]f32).init(.{
    .transparent_black = .{ 0, 0, 0, 0 },
    .opaque_black = .{ 0, 0, 0, 1 },
    .opaque_white = .{ 1, 1, 1, 1 },
});

fn createSampler(impl: backend.Impl, desc: types.SamplerDesc) Error!backend.Native {
    const self = cast(impl);
    const res = try self.gpa.create(SamplerRes);
    errdefer self.gpa.destroy(res);

    // "Level zero only" is a range of levels with one level in it: D3D11 has
    // no mip filter that is off, only a top and a bottom to clamp the level to.
    const level_zero_only = desc.mip_filter == .none;
    res.* = .{
        .state = self.raw.createSamplerState(.{
            .filter = samplerFilter(desc),
            .address_u = address_modes.get(desc.wrap_u),
            .address_v = address_modes.get(desc.wrap_v),
            .address_w = address_modes.get(desc.wrap_w),
            .mip_lod_bias = desc.lod_bias,
            .max_anisotropy = desc.max_anisotropy,
            .comparison = if (desc.compare) |compare| comparison(compare) else .never,
            .border_color = border_colors.get(desc.border),
            .min_lod = if (level_zero_only) 0 else desc.lod_min,
            .max_lod = if (level_zero_only) 0 else desc.lod_max,
        }) catch return error.Failed,
    };
    return res;
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
        // A pipeline for a pass with no colour writes none, whatever its shader outputs.
        .render_target_write_mask = if (desc.color_format == null) 0 else color_write_all,
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
        .multisample_enable = if (desc.samples > 1) 1 else 0,
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
    res.* = .{ .swap_chain = swap_chain.?, .back_buffer = undefined, .rtv = undefined, .width = 0, .height = 0 };
    try attachBackBuffer(self, res);
    return res;
}

fn attachBackBuffer(self: *D3d, res: *SurfaceRes) Error!void {
    var raw: ?*anyopaque = null;
    res.swap_chain.vtable.GetBuffer(res.swap_chain, 0, com.iidOf(ITexture2D), &raw).check() catch return error.Failed;
    const back_buffer = com.received(ITexture2D, .s_ok, raw) catch return error.Failed;
    errdefer _ = com.release(back_buffer);

    res.rtv = self.raw.createRenderTargetView(asResource(back_buffer), null) catch return error.Failed;
    res.back_buffer = back_buffer;
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
    _ = com.release(res.back_buffer);
    _ = com.release(res.swap_chain);
    self.gpa.destroy(res);
}

fn resizeSurface(impl: backend.Impl, native: backend.Native, width: u32, height: u32) Error!void {
    const self = cast(impl);
    const res = as(SurfaceRes, native);
    self.context.vtable.ClearState(self.context);
    self.context.vtable.Flush(self.context);
    _ = com.release(res.rtv);
    _ = com.release(res.back_buffer);
    res.swap_chain.vtable.ResizeBuffers(res.swap_chain, 0, width, height, .unknown, 0).check() catch return error.Failed;
    try attachBackBuffer(self, res);
}

fn surfaceSize(impl: backend.Impl, native: backend.Native) [2]u32 {
    _ = impl;
    const res = as(SurfaceRes, native);
    return .{ res.width, res.height };
}

/// A sync interval of one waits for the refresh; of nought, shows the frame
/// at once - which, in the flip model the swap chain is made with, the
/// compositor still shows whole: `disabled` and `mailbox` alike.
fn present(impl: backend.Impl, native: backend.Native, mode: types.PresentMode) Error!void {
    _ = impl;
    const res = as(SurfaceRes, native);
    res.swap_chain.vtable.Present(res.swap_chain, if (mode.waits()) 1 else 0, 0).check() catch |err| switch (err) {
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
                // pass can sample this one's target without a hazard - and so
                // that what is resolved is not still bound as a target.
                context.vtable.OMSetRenderTargets(context, 0, null, null);
                unbindTextures(self);
                for (self.resolves[0..self.resolve_count]) |resolve| {
                    context.vtable.ResolveSubresource(context, resolve.destination, 0, resolve.source, 0, resolve.format);
                }
                self.resolve_count = 0;
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
            .generate_mips => |h| {
                const res = as(TextureRes, device.textures.get(h).?.native);
                // Only ever asked of a texture `createTexture` made fillable.
                context.vtable.GenerateMips(context, res.srv orelse return error.InvalidArgument);
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

/// Nothing sampled stays bound between passes, so that this pass's target
/// may be the last pass's texture. Only the slots a frame uses are cleared,
/// and they are on both stages that `set_texture` binds.
const unbound_slots = 8;

fn unbindTextures(self: *D3d) void {
    const none: [unbound_slots]?*IShaderResourceView = @splat(null);
    self.context.vtable.PSSetShaderResources(self.context, 0, none.len, &none);
    self.context.vtable.VSSetShaderResources(self.context, 0, none.len, &none);
}

fn beginPass(self: *D3d, device: *Device, pass: types.RenderPassDesc) Error!void {
    const context = self.context;

    // Whatever was sampled last pass may be this pass's target.
    unbindTextures(self);

    // The colour attachments, `color` first and `extra_colors` after it, in
    // the slots of the output merger they will be bound to. `Device` has
    // counted them against `max_color_attachments`.
    var attachments: [fl11.max_color_attachments]types.ColorAttachment = undefined;
    var count: u32 = 0;
    if (pass.color) |color| {
        attachments[count] = color;
        count += 1;
    }
    for (pass.extra_colors) |extra| {
        attachments[count] = extra;
        count += 1;
    }

    var views: [fl11.max_color_attachments]?*IRenderTargetView = @splat(null);
    var clears: [fl11.max_color_attachments]?types.Color = @splat(null);
    var extent: ?[2]u32 = null;
    self.resolve_count = 0;

    for (attachments[0..count], 0..) |attachment, i| {
        var size: [2]u32 = undefined;
        var source: ?*TextureRes = null;
        switch (attachment.target) {
            .surface => |h| {
                const res = as(SurfaceRes, device.surfaces.get(h).?.native);
                views[i] = res.rtv;
                size = .{ res.width, res.height };
            },
            .texture => |h| {
                const res = as(TextureRes, device.textures.get(h).?.native);
                views[i] = try targetView(self, res, attachment.mip_level, attachment.layer);
                size = .{ types.mipExtent(res.width, attachment.mip_level), types.mipExtent(res.height, attachment.mip_level) };
                source = res;
            },
        }
        extent = extent orelse size;
        if (attachment.load == .clear) clears[i] = attachment.clear_color;

        if (attachment.resolve) |target| {
            // `Device` only lets a multisampled texture have one, so the
            // source is a texture, and its first subresource is all of it.
            const from = source orelse return error.InvalidArgument;
            const destination: *IResource = switch (target) {
                .texture => |h| as(TextureRes, device.textures.get(h).?.native).resource,
                .surface => |h| blk: {
                    // A swap chain is made in one format and one size, and
                    // `ResolveSubresource` will not convert between them.
                    const surface = as(SurfaceRes, device.surfaces.get(h).?.native);
                    if (from.format != surface_format or surface.width != size[0] or surface.height != size[1]) return error.InvalidArgument;
                    break :blk asResource(surface.back_buffer);
                },
            };
            self.resolves[self.resolve_count] = .{ .source = from.resource, .destination = destination, .format = from.native.target };
            self.resolve_count += 1;
        }
    }

    var dsv: ?*IDepthStencilView = null;
    var depth_clear: ?types.DepthAttachment = null;
    var depth_has_stencil = false;
    if (pass.depth) |depth| {
        const res = as(TextureRes, device.textures.get(depth.texture).?.native);
        dsv = @ptrCast(try targetView(self, res, depth.mip_level, depth.layer));
        extent = extent orelse .{ types.mipExtent(res.width, depth.mip_level), types.mipExtent(res.height, depth.mip_level) };
        if (depth.load == .clear) depth_clear = depth;
        depth_has_stencil = res.format.hasStencil();
    }

    // With no colour at all, a depth-only pass, there are no views to bind and
    // the count says so.
    context.vtable.OMSetRenderTargets(context, count, if (count == 0) null else &views, dsv);
    for (clears[0..count], views[0..count]) |clear, view| {
        if (clear) |color| context.vtable.ClearRenderTargetView(context, view.?, &color);
    }
    if (depth_clear) |depth| {
        context.vtable.ClearDepthStencilView(context, dsv.?, .{ .depth = true, .stencil = depth_has_stencil }, depth.clear_depth, depth.clear_stencil);
    }

    self.target_width = extent.?[0];
    self.target_height = extent.?[1];
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

// -------------------------------------------------------------------------
// The tables, before any device is asked
// -------------------------------------------------------------------------

test "the DXGI numbers are the header's" {
    // From `dxgiformat.h`. A wrong one is a silent bug: a texture in a format
    // nobody asked for, or a call the runtime refuses with no explanation.
    const known = [_]struct { Format, u32 }{
        .{ .unknown, 0 },
        .{ .r32g32b32a32_float, 2 },
        .{ .r32g32b32_float, 6 },
        .{ .r16g16b16a16_float, 10 },
        .{ .r32g32_float, 16 },
        .{ .r32g8x24_typeless, 19 },
        .{ .d32_float_s8x24_uint, 20 },
        .{ .r32_float_x8x24_typeless, 21 },
        .{ .r10g10b10a2_unorm, 24 },
        .{ .r11g11b10_float, 26 },
        .{ .r8g8b8a8_unorm, 28 },
        .{ .r8g8b8a8_unorm_srgb, 29 },
        .{ .r8g8b8a8_uint, 30 },
        .{ .r16g16_float, 34 },
        .{ .r32_typeless, 39 },
        .{ .d32_float, 40 },
        .{ .r32_float, 41 },
        .{ .r32_uint, 42 },
        .{ .r32_sint, 43 },
        .{ .r24g8_typeless, 44 },
        .{ .d24_unorm_s8_uint, 45 },
        .{ .r24_unorm_x8_typeless, 46 },
        .{ .r8g8_unorm, 49 },
        .{ .r16_typeless, 53 },
        .{ .r16_float, 54 },
        .{ .d16_unorm, 55 },
        .{ .r16_unorm, 56 },
        .{ .r16_uint, 57 },
        .{ .r8_unorm, 61 },
        .{ .bc1_typeless, 70 },
        .{ .bc1_unorm, 71 },
        .{ .bc1_unorm_srgb, 72 },
        .{ .bc3_typeless, 76 },
        .{ .bc3_unorm, 77 },
        .{ .bc3_unorm_srgb, 78 },
        .{ .bc4_typeless, 79 },
        .{ .bc4_unorm, 80 },
        .{ .bc5_typeless, 82 },
        .{ .bc5_unorm, 83 },
        .{ .b8g8r8a8_unorm, 87 },
        .{ .b8g8r8a8_unorm_srgb, 91 },
        .{ .bc6h_typeless, 94 },
        .{ .bc6h_uf16, 95 },
        .{ .bc7_typeless, 97 },
        .{ .bc7_unorm, 98 },
        .{ .bc7_unorm_srgb, 99 },
    };
    for (known) |entry| try testing.expectEqual(entry[1], @intFromEnum(entry[0]));
}

test "the texture and view descriptions are shaped the way the runtime reads them" {
    try testing.expectEqual(@as(usize, 44), @sizeOf(Texture2DDesc));
    try testing.expectEqual(@as(usize, 36), @sizeOf(Texture3DDesc));
    try testing.expectEqual(@as(usize, 24), @sizeOf(ShaderResourceViewDesc));
    try testing.expectEqual(@as(usize, 20), @sizeOf(RenderTargetViewDesc));
    try testing.expectEqual(@as(usize, 24), @sizeOf(DepthStencilViewDesc));
    // `D3D11_RESOURCE_MISC_GENERATE_MIPS`, `_SHARED` and `_TEXTURECUBE`.
    try testing.expectEqual(@as(u32, 0x1), @as(u32, @bitCast(TextureMisc{ .generate_mips = true })));
    try testing.expectEqual(@as(u32, 0x2), @as(u32, @bitCast(TextureMisc{ .shared = true })));
    try testing.expectEqual(@as(u32, 0x4), @as(u32, @bitCast(TextureMisc{ .texture_cube = true })));
    // The view dimensions that are the header's.
    try testing.expectEqual(@as(u32, 4), @intFromEnum(SrvDimension.texture2d));
    try testing.expectEqual(@as(u32, 5), @intFromEnum(SrvDimension.texture2d_array));
    try testing.expectEqual(@as(u32, 6), @intFromEnum(SrvDimension.texture2d_ms));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(SrvDimension.texture3d));
    try testing.expectEqual(@as(u32, 9), @intFromEnum(SrvDimension.texture_cube));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(RtvDimension.texture2d));
    try testing.expectEqual(@as(u32, 5), @intFromEnum(RtvDimension.texture2d_array));
    try testing.expectEqual(@as(u32, 6), @intFromEnum(RtvDimension.texture2d_ms));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(RtvDimension.texture3d));
    try testing.expectEqual(@as(u32, 3), @intFromEnum(DsvDimension.texture2d));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(DsvDimension.texture2d_array));
    try testing.expectEqual(@as(u32, 5), @intFromEnum(DsvDimension.texture2d_ms));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(AddressMode.border));
}

test "a filter is what D3D11_ENCODE_BASIC_FILTER says it is" {
    const Case = struct { types.SamplerDesc, u32 };
    const cases = [_]Case{
        // D3D11_FILTER_MIN_MAG_MIP_POINT and the seven that follow it.
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .none }, 0x00 },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest }, 0x00 },
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .linear }, 0x01 },
        .{ .{ .min_filter = .nearest, .mag_filter = .linear, .mip_filter = .nearest }, 0x04 },
        .{ .{ .min_filter = .nearest, .mag_filter = .linear, .mip_filter = .linear }, 0x05 },
        .{ .{ .min_filter = .linear, .mag_filter = .nearest, .mip_filter = .nearest }, 0x10 },
        .{ .{ .min_filter = .linear, .mag_filter = .nearest, .mip_filter = .linear }, 0x11 },
        .{ .{ .min_filter = .linear, .mag_filter = .linear, .mip_filter = .nearest }, 0x14 },
        .{ .{ .min_filter = .linear, .mag_filter = .linear, .mip_filter = .linear }, 0x15 },
        // D3D11_FILTER_ANISOTROPIC, and what asking for it with a point filter gets.
        .{ .{ .max_anisotropy = 8, .mip_filter = .linear }, 0x55 },
        .{ .{ .max_anisotropy = 8, .min_filter = .nearest }, 0x04 },
        // The comparison variants: 0x80 and the same seven, and D3D11_FILTER_COMPARISON_ANISOTROPIC.
        .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .compare = .less }, 0x80 },
        .{ .{ .min_filter = .linear, .mag_filter = .linear, .mip_filter = .linear, .compare = .less_equal }, 0x95 },
        .{ .{ .min_filter = .linear, .mag_filter = .linear, .mip_filter = .nearest, .compare = .less_equal }, 0x94 },
        .{ .{ .max_anisotropy = 4, .compare = .greater }, 0xd5 },
    };
    for (cases) |case| try testing.expectEqual(case[1], @intFromEnum(samplerFilter(case[0])));
}

test "every format has a native row or a null, and the rows agree with the format" {
    // A probe that says yes to everything, to see what a row lets through.
    const everything: Probe = .{ .srv = @bitCast(~@as(u32, 0)), .target = @bitCast(~@as(u32, 0)), .multisample = 0xFF };

    for (std.enums.values(types.Format)) |format| {
        const support = deriveSupport(format, native_table.get(format), everything);
        const row = native_table.get(format) orelse {
            // No row: nothing may be claimed, whatever the runtime said.
            try testing.expectEqual(types.FormatSupport{}, support);
            try testing.expect(format.isCompressed());
            continue;
        };
        const row_info = format.info();
        if (row_info.depth) {
            // Written as one type and read as another, so the resource is typeless
            // and the two views are different formats. Never read back as colour.
            try testing.expect(row.resource != row.target and row.resource != row.srv and row.srv != row.target);
            try testing.expectEqual(@as(?Decode, null), row.decode);
            try testing.expect(!support.blendable and !support.generate_mips);
        } else if (format.isCompressed()) {
            try testing.expect(row.resource == row.srv and row.srv == row.target);
            try testing.expectEqual(@as(?Decode, null), row.decode);
            // Compressed formats are for sampling: the Device refuses a target, and the caps never offer one.
            try testing.expect(!support.generate_mips);
        } else {
            try testing.expect(row.resource == row.srv and row.srv == row.target);
            const decode = row.decode orelse return error.TestUnexpectedResult;
            // The kernel that unpacks it reads as many bytes and channels as the format's row says.
            const per_channel: usize = switch (decode) {
                .unorm8 => 1,
                .float16 => 2,
                .float32 => 4,
                .bgra8, .rgb10a2, .rg11b10_float => 0,
            };
            if (per_channel != 0) {
                try testing.expectEqual(@as(usize, row_info.channels) * per_channel, row_info.block_bytes);
            } else {
                try testing.expectEqual(@as(usize, 4), row_info.block_bytes);
            }
        }
        // Something the table has a row for can be claimed, when the runtime says so.
        try testing.expect(support.sampled);
        try testing.expect(support.supportsSamples(1));
    }
}

test "nothing is claimed that the runtime did not say" {
    // A format the runtime knows nothing about, and one it knows nothing of
    // for drawing: caps follows the probe and not the table alone.
    const nothing: Probe = .{};
    for (std.enums.values(types.Format)) |format| {
        try testing.expectEqual(types.FormatSupport{}, deriveSupport(format, native_table.get(format), nothing));
    }
    const read_only: Probe = .{ .srv = .{ .texture2d = true, .shader_sample = true }, .multisample = 0xFF };
    const support = deriveSupport(.rgba8_unorm, native_table.get(.rgba8_unorm), read_only);
    try testing.expect(support.sampled and support.filterable);
    try testing.expect(!support.render_target and !support.generate_mips and !support.blendable);
    // More than one sample is for a render target, and a resolve, and only then.
    try testing.expectEqual(@as(u8, 0b1), support.sample_counts);
    // A volume needs the runtime's TEXTURE3D on both the views a texture is made with.
    try testing.expect(!supportsDimension(read_only, .d3));
    try testing.expect(supportsDimension(.{ .srv = .{ .texture2d = true }, .target = .{ .texture2d = true } }, .d2_array));
}

// -------------------------------------------------------------------------
// The device, on WARP
// -------------------------------------------------------------------------

const palette = [_][4]u8{
    .{ 255, 0, 0, 255 },
    .{ 0, 255, 0, 255 },
    .{ 0, 0, 255, 255 },
    .{ 255, 255, 0, 255 },
    .{ 0, 255, 255, 255 },
    .{ 255, 0, 255, 255 },
    .{ 255, 255, 255, 255 },
    .{ 128, 64, 32, 255 },
};

fn solidTexels(comptime count: usize, colour: [4]u8) [count * 4]u8 {
    var bytes: [count * 4]u8 = undefined;
    for (0..count) |i| bytes[i * 4 ..][0..4].* = colour;
    return bytes;
}

fn expectSolid(pixels: []const u8, colour: [4]u8) !void {
    try testing.expect(pixels.len % 4 == 0);
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) try testing.expectEqualSlices(u8, &colour, pixels[i..][0..4]);
}

fn expectNear(expected: [4]u8, actual: [4]u8, tolerance: u8) !void {
    for (expected, actual) |e, a| {
        const distance = if (e > a) e - a else a - e;
        if (distance > tolerance) {
            std.debug.print("expected {any}, found {any}\n", .{ expected, actual });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

fn pixelAt(pixels: []const u8, width: usize, x: usize, y: usize) [4]u8 {
    return pixels[(y * width + x) * 4 ..][0..4].*;
}

fn readLevel(device: *Device, texture: types.Texture, sub: types.Subresource) ![]u8 {
    return device.readSubresource(texture, sub, testing.allocator);
}

test "caps on WARP: limits, features, and what each format can be used for" {
    var device = try warpDevice();
    defer device.deinit();
    const caps_of = device.caps();

    // Feature level 11.0's own numbers.
    try testing.expectEqual(@as(u32, 16384), caps_of.limits.max_texture_2d);
    try testing.expectEqual(@as(u32, 2048), caps_of.limits.max_texture_3d);
    try testing.expectEqual(@as(u32, 16384), caps_of.limits.max_texture_cube);
    try testing.expectEqual(@as(u32, 2048), caps_of.limits.max_texture_layers);
    try testing.expectEqual(@as(u32, 16), caps_of.limits.max_anisotropy);
    try testing.expectEqual(@as(u32, 8), caps_of.limits.max_color_attachments);
    try testing.expect(caps_of.features.sampler_border and caps_of.features.sampler_lod_bias);

    const rgba8 = caps_of.formatSupport(.rgba8_unorm);
    try testing.expect(rgba8.sampled and rgba8.filterable and rgba8.render_target and rgba8.blendable and rgba8.generate_mips);
    try testing.expect(rgba8.supportsSamples(1) and rgba8.supportsSamples(4));

    // Depth is drawn into and read as a shadow map, and neither blends nor has levels made.
    const depth = caps_of.formatSupport(.depth32_float);
    try testing.expect(depth.sampled and depth.render_target and !depth.blendable and !depth.generate_mips);
    try testing.expect(caps_of.formatSupport(.depth24_stencil8).render_target);
    try testing.expect(caps_of.formatSupport(.depth16_unorm).sampled);

    // BC is Direct3D 11's to guarantee: sampled, never drawn into, one sample.
    for ([_]types.Format{ .bc1_rgba_unorm, .bc1_rgba_unorm_srgb, .bc3_rgba_unorm, .bc4_r_unorm, .bc5_rg_unorm, .bc6h_rgb_ufloat, .bc7_rgba_unorm, .bc7_rgba_unorm_srgb }) |format| {
        const bc = caps_of.formatSupport(format);
        try testing.expect(bc.sampled and bc.filterable);
        try testing.expect(!bc.render_target and !bc.generate_mips);
        try testing.expectEqual(@as(u8, 0b1), bc.sample_counts);
    }
    // ETC2 and ASTC are not on any Direct3D 11 device.
    for ([_]types.Format{ .etc2_rgb8_unorm, .etc2_rgba8_unorm_srgb, .astc_4x4_unorm, .astc_8x8_unorm_srgb }) |format| {
        try testing.expectEqual(types.FormatSupport{}, caps_of.formatSupport(format));
    }

    // Nothing is claimed that the table has no row for.
    for (std.enums.values(types.Format)) |format| {
        const support = caps_of.formatSupport(format);
        if (native_table.get(format) == null) try testing.expectEqual(types.FormatSupport{}, support);
        if (support.render_target) try testing.expect(!format.isCompressed());
        if (support.generate_mips) try testing.expect(support.sampled and support.render_target);
    }
}

test "whatever caps claims can be made" {
    var device = try warpDevice();
    defer device.deinit();
    const caps_of = device.caps();

    for (std.enums.values(types.Format)) |format| {
        const support = caps_of.formatSupport(format);
        if (!support.sampled and !support.render_target) continue;
        // Every shape a colour format can take, sampled or drawn into; a depth
        // format has no volume, and says so as Unsupported and not as a crash.
        for (std.enums.values(types.Dimension)) |dimension| {
            const usage: types.TextureUsage = if (support.sampled) .{} else .{ .sampled = false, .render_target = true };
            const made = device.createTexture(.{ .dimension = dimension, .width = 8, .height = 8, .depth_or_layers = if (dimension == .d2) 1 else 4, .format = format, .usage = usage });
            if (dimension == .d3 and format.isDepth()) {
                try testing.expectError(error.Unsupported, made);
            } else {
                device.destroyTexture(try made);
            }
            if (support.render_target and support.sampled and !(dimension == .d3 and format.isDepth())) {
                device.destroyTexture(try device.createTexture(.{ .dimension = dimension, .width = 8, .height = 8, .depth_or_layers = if (dimension == .d2) 1 else 4, .format = format, .usage = .{ .sampled = true, .render_target = true } }));
            }
        }
        if (support.render_target) {
            var samples: u32 = 2;
            while (samples <= fl11.max_samples) : (samples *= 2) {
                if (!support.supportsSamples(samples)) continue;
                device.destroyTexture(try device.createTexture(.{ .width = 16, .height = 16, .samples = samples, .format = format, .usage = .{ .sampled = false, .render_target = true } }));
            }
        }
        if (support.generate_mips) {
            device.destroyTexture(try device.createTexture(.{ .width = 16, .height = 16, .mip_levels = 0, .format = format }));
        }
    }
}

test "each mip level of a texture is written and read on its own" {
    var device = try warpDevice();
    defer device.deinit();

    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 0 });
    try testing.expectEqual(@as(u32, 4), (try device.textureInfo(texture)).mip_levels);

    var buffer: [8 * 8 * 4]u8 = undefined;
    for (0..4) |mip| {
        const size: usize = types.mipExtent(8, @intCast(mip));
        const level = buffer[0 .. size * size * 4];
        for (0..size * size) |i| level[i * 4 ..][0..4].* = palette[mip];
        try device.writeTexture(texture, .{ .mip = @intCast(mip) }, level, 0, 0);
    }
    for (0..4) |mip| {
        const size: usize = types.mipExtent(8, @intCast(mip));
        const pixels = try readLevel(&device, texture, .{ .mip = @intCast(mip) });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(size * size * 4, pixels.len);
        try expectSolid(pixels, palette[mip]);
    }

    // A box in the middle of level zero, and the rest of it keeps its colour.
    const patch = solidTexels(9, palette[3]);
    try device.writeTexture(texture, .{ .x = 2, .y = 3, .width = 3, .height = 3 }, &patch, 0, 0);
    {
        const pixels = try readLevel(&device, texture, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[3], pixelAt(pixels, 8, 2, 3));
        try testing.expectEqual(palette[3], pixelAt(pixels, 8, 4, 5));
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 1, 3));
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 5, 4));
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 3, 6));
    }

    // A row pitch wider than a row: the bytes between are not texels.
    var padded: [16 * 2]u8 = @splat(0xEE);
    padded[0..8].* = palette[1] ++ palette[1];
    padded[16..24].* = palette[2] ++ palette[2];
    try device.writeTexture(texture, .{ .mip = 1, .x = 1, .y = 1, .width = 2, .height = 2 }, &padded, 16, 0);
    {
        const pixels = try readLevel(&device, texture, .{ .mip = 1 });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[1], pixelAt(pixels, 4, 1, 1));
        try testing.expectEqual(palette[1], pixelAt(pixels, 4, 2, 1));
        try testing.expectEqual(palette[2], pixelAt(pixels, 4, 1, 2));
        try testing.expectEqual(palette[2], pixelAt(pixels, 4, 2, 2));
        try testing.expectEqual(palette[1], pixelAt(pixels, 4, 0, 0));
    }
}

test "generateMips fills the chain from level zero, in list order" {
    var device = try warpDevice();
    defer device.deinit();

    // A 4x4 checkerboard of black and white: every level below is grey.
    const white = palette[6];
    const black = [4]u8{ 0, 0, 0, 255 };
    var checker: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| checker[i * 4 ..][0..4].* = if ((i % 4 + i / 4) % 2 == 0) white else black;

    // A texture that says only that it is sampled: the chain is made fillable anyway.
    const texture = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0, .data = &checker });
    try testing.expectEqual(@as(u32, 3), (try device.textureInfo(texture)).mip_levels);

    const cmd = device.begin();
    try cmd.generateMips(texture);
    try device.submit();

    {
        // Level zero is what was written.
        const pixels = try readLevel(&device, texture, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqualSlices(u8, &checker, pixels);
    }
    {
        const pixels = try readLevel(&device, texture, .{ .mip = 1 });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, 2 * 2 * 4), pixels.len);
        for (0..4) |i| try expectNear(.{ 127, 127, 127, 255 }, pixels[i * 4 ..][0..4].*, 2);
    }
    {
        // The one texel that is left is the average of all sixteen.
        const pixels = try readLevel(&device, texture, .{ .mip = 2 });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, 4), pixels.len);
        try expectNear(.{ 127, 127, 127, 255 }, pixels[0..4].*, 2);
    }
}

test "a cube is six faces, written and read by layer" {
    var device = try warpDevice();
    defer device.deinit();

    // Six faces, one colour each, written a face at a time.
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4 });
    try testing.expectEqual(@as(u32, 6), (try device.textureInfo(cube)).depth_or_layers);
    const face = solidTexels(16, .{ 0, 0, 0, 255 });
    for (0..6) |i| {
        var bytes = face;
        for (0..16) |t| bytes[t * 4 ..][0..4].* = palette[i];
        try device.writeTexture(cube, .{ .z = @intCast(i), .depth = 1 }, &bytes, 0, 0);
    }
    for (0..6) |i| {
        const pixels = try readLevel(&device, cube, .{ .layer = @intCast(i) });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[i]);
    }

    // And all six at once, from the bytes the texture was made from.
    var all: [16 * 6 * 4]u8 = undefined;
    for (0..6) |i| for (0..16) |t| {
        all[(i * 16 + t) * 4 ..][0..4].* = palette[5 - i];
    };
    const made = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &all });
    for (0..6) |i| {
        const pixels = try readLevel(&device, made, .{ .layer = @intCast(i) });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[5 - i]);
    }
}

test "a volume is slices, written and read at level zero and one" {
    var device = try warpDevice();
    defer device.deinit();

    // Four slices of 4x4, from bytes, with a second level of two slices of 2x2.
    var initial: [4 * 4 * 4 * 4]u8 = undefined;
    for (0..4) |z| for (0..16) |t| {
        initial[(z * 16 + t) * 4 ..][0..4].* = palette[z];
    };
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4, .mip_levels = 2, .data = &initial });
    for (0..4) |z| {
        const pixels = try readLevel(&device, volume, .{ .layer = @intCast(z) });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, 4 * 4 * 4), pixels.len);
        try expectSolid(pixels, palette[z]);
    }

    // Level one, a slice at a time.
    for (0..2) |z| {
        const slice = solidTexels(4, palette[4 + z]);
        try device.writeTexture(volume, .{ .mip = 1, .z = @intCast(z), .depth = 1 }, &slice, 0, 0);
    }
    for (0..2) |z| {
        const pixels = try readLevel(&device, volume, .{ .mip = 1, .layer = @intCast(z) });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, 2 * 2 * 4), pixels.len);
        try expectSolid(pixels, palette[4 + z]);
    }

    // The whole of level one in a single write, slices further apart than they are big.
    var spaced: [32 * 2]u8 = @splat(0xEE);
    spaced[0..16].* = solidTexels(4, palette[2]);
    spaced[32..48].* = solidTexels(4, palette[3]);
    try device.writeTexture(volume, .{ .mip = 1 }, &spaced, 0, 32);
    for (0..2) |z| {
        const pixels = try readLevel(&device, volume, .{ .mip = 1, .layer = @intCast(z) });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[2 + z]);
    }

    // A box inside one slice of level zero leaves the neighbours be.
    const patch = solidTexels(4, palette[6]);
    try device.writeTexture(volume, .{ .x = 1, .y = 1, .z = 2, .width = 2, .height = 2, .depth = 1 }, &patch, 0, 0);
    {
        const pixels = try readLevel(&device, volume, .{ .layer = 2 });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[6], pixelAt(pixels, 4, 1, 1));
        try testing.expectEqual(palette[6], pixelAt(pixels, 4, 2, 2));
        try testing.expectEqual(palette[2], pixelAt(pixels, 4, 0, 0));
        try testing.expectEqual(palette[2], pixelAt(pixels, 4, 3, 3));
    }
    {
        const pixels = try readLevel(&device, volume, .{ .layer = 3 });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[3]);
    }
}

// The flat shaders again, with a depth to draw at: what the passes below draw with.
const flat3_vs =
    \\struct In { float3 position : ATTR0; float4 colour : ATTR1; };
    \\struct Out { float4 position : SV_POSITION; float4 colour : COLOR0; };
    \\Out main(In i) { Out o; o.position = float4(i.position, 1); o.colour = i.colour; return o; }
;

const Vertex3 = extern struct { position: [3]f32, colour: [4]f32 };

/// A rectangle as a triangle strip, at one depth, in one colour.
fn quad3(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, colour: [4]f32) [4]Vertex3 {
    return .{
        .{ .position = .{ x0, y0, z }, .colour = colour },
        .{ .position = .{ x0, y1, z }, .colour = colour },
        .{ .position = .{ x1, y0, z }, .colour = colour },
        .{ .position = .{ x1, y1, z }, .colour = colour },
    };
}

fn vertexBuffer(device: *Device, vertices: anytype) !types.Buffer {
    return device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });
}

const FlatOptions = struct {
    color_format: ?types.Format = .rgba8_unorm,
    extra_color_formats: []const types.Format = &.{},
    depth_format: ?types.Format = null,
    depth: types.DepthState = .none,
    samples: u32 = 1,
    topology: types.Topology = .triangle_strip,
};

fn flat3Pipeline(device: *Device, shader: types.Shader, options: FlatOptions) !types.Pipeline {
    return device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float3, .offset = 0 },
            .{ .location = 1, .format = .float4, .offset = 12 },
        },
        .buffers = &.{.{ .stride = @sizeOf(Vertex3) }},
        .topology = options.topology,
        .depth = options.depth,
        .color_format = options.color_format,
        .extra_color_formats = options.extra_color_formats,
        .depth_format = options.depth_format,
        .samples = options.samples,
    });
}

fn flat3Shader(device: *Device) !types.Shader {
    return device.createShader(.{ .hlsl = .{ .vertex = flat3_vs, .fragment = flat_ps } }) catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };
}

/// One pass drawing one buffer of four vertices with one pipeline.
fn drawStrip(device: *Device, pass: types.RenderPassDesc, pipeline: types.Pipeline, buffer: types.Buffer) !void {
    const cmd = device.begin();
    try cmd.beginPass(pass);
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.draw(.{ .vertex_count = 4 });
    try cmd.endPass();
    device.submit() catch |err| {
        std.debug.print("{s}\n", .{device.diagnostics()});
        return err;
    };
}

test "a pass draws into one layer and one level of a texture, and nowhere else" {
    var device = try warpDevice();
    defer device.deinit();

    const shader = try flat3Shader(&device);
    const pipeline = try flat3Pipeline(&device, shader, .{});

    // Three layers of 16x16 with a second level: a place to draw into that is neither the first layer nor the first level.
    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 16, .height = 16, .depth_or_layers = 3, .mip_levels = 2, .usage = .{ .render_target = true } });
    const underneath = solidTexels(16 * 16, palette[3]);
    for (0..3) |layer| try device.writeTexture(array, .{ .z = @intCast(layer), .depth = 1 }, &underneath, 0, 0);

    // Red over layer one, level zero.
    const red = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0, .{ 1, 0, 0, 1 }));
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = array }, .layer = 1, .load = .load } }, pipeline, red);
    // Green over the left half of layer two's level one: half of eight is four columns, not eight.
    const green = try vertexBuffer(&device, quad3(-1, -1, 0, 1, 0, .{ 0, 1, 0, 1 }));
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = array }, .layer = 2, .mip_level = 1, .clear_color = .{ 0, 0, 1, 1 } } }, pipeline, green);

    {
        const pixels = try readLevel(&device, array, .{ .layer = 1 });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[0]);
    }
    {
        // The neighbours of what was drawn into are what was there.
        const pixels = try readLevel(&device, array, .{ .layer = 0 });
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[3]);
        const last = try readLevel(&device, array, .{ .layer = 2 });
        defer testing.allocator.free(last);
        try expectSolid(last, palette[3]);
    }
    {
        // Level one of layer two is 8x8, cleared blue with its left half green. A viewport of the whole 16
        // would have put green over all of it.
        const pixels = try readLevel(&device, array, .{ .layer = 2, .mip = 1 });
        defer testing.allocator.free(pixels);
        try testing.expectEqual(@as(usize, 8 * 8 * 4), pixels.len);
        try testing.expectEqual(palette[1], pixelAt(pixels, 8, 1, 4));
        try testing.expectEqual(palette[1], pixelAt(pixels, 8, 3, 0));
        try testing.expectEqual(palette[2], pixelAt(pixels, 8, 5, 4));
        try testing.expectEqual(palette[2], pixelAt(pixels, 8, 7, 7));
    }

    // A face of a cube is a layer too, and a slice of a volume is drawn into by the same word.
    const cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .usage = .{ .render_target = true } });
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = cube }, .layer = 4 } }, pipeline, red);
    const volume = try device.createTexture(.{ .dimension = .d3, .width = 8, .height = 8, .depth_or_layers = 4, .usage = .{ .render_target = true } });
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = volume }, .layer = 3 } }, pipeline, green);
    {
        const face = try readLevel(&device, cube, .{ .layer = 4 });
        defer testing.allocator.free(face);
        try expectSolid(face, palette[0]);
        const slice = try readLevel(&device, volume, .{ .layer = 3 });
        defer testing.allocator.free(slice);
        // The green went to the left half of the fourth slice: not the first, and not all of it.
        try testing.expectEqual(palette[1], pixelAt(slice, 8, 1, 4));
        try testing.expectEqual([4]u8{ 0, 0, 0, 255 }, pixelAt(slice, 8, 6, 4));
        const first = try readLevel(&device, volume, .{ .layer = 0 });
        defer testing.allocator.free(first);
        try expectSolid(first, .{ 0, 0, 0, 0 });
    }
}

test "a multisampled target is resolved into a texture at the end of the pass" {
    var device = try warpDevice();
    defer device.deinit();
    if (!device.caps().formatSupport(.rgba8_unorm).supportsSamples(4)) return error.SkipZigTest;

    const shader = try flat3Shader(&device);
    const pipeline = try flat3Pipeline(&device, shader, .{ .topology = .triangles, .samples = 4 });
    const msaa = try device.createTexture(.{ .width = 32, .height = 32, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const resolved = try device.createTexture(.{ .width = 32, .height = 32, .usage = .{ .render_target = true } });

    // A red triangle over blue, with slanted sides, so that some pixels are only partly covered.
    const red: [4]f32 = .{ 1, 0, 0, 1 };
    const triangle = [_]Vertex3{
        .{ .position = .{ -0.8, -0.8, 0 }, .colour = red },
        .{ .position = .{ 0, 0.8, 0 }, .colour = red },
        .{ .position = .{ 0.8, -0.8, 0 }, .colour = red },
    };
    const buffer = try vertexBuffer(&device, triangle);
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .clear_color = .{ 0, 0, 1, 1 }, .resolve = .{ .texture = resolved } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try readLevel(&device, resolved, .{});
    defer testing.allocator.free(pixels);
    // Inside is red, outside is blue, and the edge is neither: samples averaged.
    try testing.expectEqual(palette[0], pixelAt(pixels, 32, 16, 24));
    try testing.expectEqual(palette[2], pixelAt(pixels, 32, 1, 1));
    try testing.expectEqual(palette[2], pixelAt(pixels, 32, 30, 3));
    var blended: usize = 0;
    for (0..32 * 32) |i| {
        const p = pixels[i * 4 ..][0..4];
        if (p[0] > 0 and p[0] < 255 and p[2] > 0 and p[2] < 255) blended += 1;
    }
    try testing.expect(blended > 10);

    // The multisampled texture itself cannot be read, and Device says so before the backend is asked.
    try testing.expectError(error.InvalidArgument, device.readTexture(msaa, testing.allocator));
}

test "a depth-only pass, and then a colour pass that tests against it" {
    var device = try warpDevice();
    defer device.deinit();

    const shader = try flat3Shader(&device);
    // No colour: nothing is written but depth, and a shader that outputs one is not an error.
    const depth_only = try flat3Pipeline(&device, shader, .{ .color_format = null, .depth_format = .depth32_float, .depth = .standard });
    const coloured = try flat3Pipeline(&device, shader, .{ .depth_format = .depth32_float, .depth = .standard });

    const depth = try device.createTexture(.{ .width = 16, .height = 16, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } });
    const target = try device.createTexture(.{ .width = 16, .height = 16, .usage = .{ .render_target = true } });

    // The left half of the depth buffer is written near, and nothing else.
    const near_left = try vertexBuffer(&device, quad3(-1, -1, 0, 1, 0.25, .{ 1, 1, 1, 1 }));
    try drawStrip(&device, .{ .depth = .{ .texture = depth } }, depth_only, near_left);

    // A red plane behind that, over everything, with the depth kept: only the right half of it is seen.
    const far = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0.75, .{ 1, 0, 0, 1 }));
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } }, .depth = .{ .texture = depth, .load = .load } }, coloured, far);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[2], pixelAt(pixels, 16, 3, 8));
        try testing.expectEqual(palette[2], pixelAt(pixels, 16, 7, 1));
        try testing.expectEqual(palette[0], pixelAt(pixels, 16, 9, 8));
        try testing.expectEqual(palette[0], pixelAt(pixels, 16, 14, 14));
    }

    // And a green plane in front of both, still testing against what the passes left, is seen everywhere.
    const nearer = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0.1, .{ 0, 1, 0, 1 }));
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .load = .load }, .depth = .{ .texture = depth, .load = .load } }, coloured, nearer);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[1], pixelAt(pixels, 16, 3, 8));
        try testing.expectEqual(palette[1], pixelAt(pixels, 16, 12, 8));
    }

    // With the depth cleared instead of kept, the far plane is seen everywhere.
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } }, .depth = .{ .texture = depth } }, coloured, far);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[0], pixelAt(pixels, 16, 3, 8));
    }
}

test "a pass with several colour attachments writes each of them" {
    var device = try warpDevice();
    defer device.deinit();

    const two_outputs =
        \\struct Out { float4 position : SV_POSITION; float4 colour : COLOR0; };
        \\struct Both { float4 first : SV_TARGET0; float4 second : SV_TARGET1; };
        \\Both main(Out i) { Both b; b.first = i.colour; b.second = float4(1.0 - i.colour.rgb, 1); return b; }
    ;
    const shader = try device.createShader(.{ .hlsl = .{ .vertex = flat3_vs, .fragment = two_outputs } });
    const pipeline = try flat3Pipeline(&device, shader, .{ .extra_color_formats = &.{.rgba16_float} });

    const first = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });
    const second = try device.createTexture(.{ .width = 8, .height = 8, .format = .rgba16_float, .usage = .{ .render_target = true } });
    const quad = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0, .{ 1, 0, 0, 1 }));
    try drawStrip(&device, .{
        .color = .{ .target = .{ .texture = first } },
        .extra_colors = &.{.{ .target = .{ .texture = second } }},
    }, pipeline, quad);

    const pixels = try readLevel(&device, first, .{});
    defer testing.allocator.free(pixels);
    try expectSolid(pixels, palette[0]);
    // The second output is one minus the first, and it is a float texture read as RGBA8.
    const other = try readLevel(&device, second, .{});
    defer testing.allocator.free(other);
    try expectSolid(other, palette[4]);
}

// A textured quad: what the sampling tests below draw with.
const tex_vs =
    \\struct In { float2 position : ATTR0; float2 uv : ATTR1; };
    \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
    \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.uv = i.uv; return o; }
;
const tex_ps_header =
    \\cbuffer Params : register(b0) { float4 param; };
    \\SamplerState s : register(s0);
    \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
;
const tex2d_ps = tex_ps_header ++
    \\Texture2D t : register(t0);
    \\float4 main(Out i) : SV_TARGET { return t.Sample(s, i.uv); }
;
const tex_array_ps = tex_ps_header ++
    \\Texture2DArray t : register(t0);
    \\float4 main(Out i) : SV_TARGET { return t.Sample(s, float3(i.uv, param.w)); }
;
const tex_cube_ps = tex_ps_header ++
    \\TextureCube t : register(t0);
    \\float4 main(Out i) : SV_TARGET { return t.Sample(s, param.xyz); }
;
const tex3d_ps = tex_ps_header ++
    \\Texture3D t : register(t0);
    \\float4 main(Out i) : SV_TARGET { return t.Sample(s, float3(i.uv, param.w)); }
;

const TexVertex = extern struct { position: [2]f32, uv: [2]f32 };

/// A quad over the whole target with the texture coordinates from zero to `uv_extent`.
fn texQuad(uv_extent: f32) [4]TexVertex {
    return .{
        .{ .position = .{ -1, -1 }, .uv = .{ 0, uv_extent } },
        .{ .position = .{ -1, 1 }, .uv = .{ 0, 0 } },
        .{ .position = .{ 1, -1 }, .uv = .{ uv_extent, uv_extent } },
        .{ .position = .{ 1, 1 }, .uv = .{ uv_extent, 0 } },
    };
}

const Sampling = struct {
    shader: types.Shader,
    pipeline: types.Pipeline,
    params: types.Buffer,
    quad: types.Buffer,
    target: types.Texture,

    fn init(device: *Device, fragment: [:0]const u8, size: u32, uv_extent: f32) !Sampling {
        const shader = device.createShader(.{ .hlsl = .{ .vertex = tex_vs, .fragment = fragment } }) catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };
        return .{
            .shader = shader,
            .pipeline = try device.createPipeline(.{
                .shader = shader,
                .attributes = &.{
                    .{ .location = 0, .format = .float2, .offset = 0 },
                    .{ .location = 1, .format = .float2, .offset = 8 },
                },
                .buffers = &.{.{ .stride = @sizeOf(TexVertex) }},
                .topology = .triangle_strip,
            }),
            .params = try device.createBuffer(.{ .kind = .uniform, .size = 16 }),
            .quad = try vertexBuffer(device, texQuad(uv_extent)),
            .target = try device.createTexture(.{ .width = size, .height = size, .usage = .{ .render_target = true } }),
        };
    }

    /// Draw `texture` through `sampler` over the target, with the shader's `param` set, and read the target.
    fn draw(self: Sampling, device: *Device, texture: types.Texture, sampler: types.Sampler, param: [4]f32) ![]u8 {
        try device.updateBuffer(self.params, 0, std.mem.asBytes(&param));
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = self.target }, .clear_color = .{ 0.5, 0.5, 0.5, 1 } } });
        try cmd.setPipeline(self.pipeline);
        try cmd.setVertexBuffer(0, self.quad, 0);
        try cmd.setUniformBuffer(0, self.params);
        try cmd.setTexture(0, texture, sampler);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        device.submit() catch |err| {
            std.debug.print("{s}\n", .{device.diagnostics()});
            return err;
        };
        return device.readTexture(self.target, testing.allocator);
    }
};

test "a texture is bound as the kind it is: array, cube and volume" {
    var device = try warpDevice();
    defer device.deinit();
    const nearest = try device.createSampler(.nearest);

    // An array of three layers, one colour each.
    {
        const sampling = try Sampling.init(&device, tex_array_ps, 4, 1);
        var layers: [4 * 4 * 4 * 3]u8 = undefined;
        for (0..3) |z| for (0..16) |t| {
            layers[(z * 16 + t) * 4 ..][0..4].* = palette[z];
        };
        const array = try device.createTexture(.{ .dimension = .d2_array, .width = 4, .height = 4, .depth_or_layers = 3, .data = &layers });
        for (0..3) |z| {
            const pixels = try sampling.draw(&device, array, nearest, .{ 0, 0, 0, @floatFromInt(z) });
            defer testing.allocator.free(pixels);
            try expectSolid(pixels, palette[z]);
        }
    }
    // A cube, looked at along a direction: +X is face zero, -Y is face three, and -Z is face five.
    {
        const sampling = try Sampling.init(&device, tex_cube_ps, 4, 1);
        var faces: [4 * 4 * 4 * 6]u8 = undefined;
        for (0..6) |f| for (0..16) |t| {
            faces[(f * 16 + t) * 4 ..][0..4].* = palette[f];
        };
        const cube = try device.createTexture(.{ .dimension = .cube, .width = 4, .height = 4, .data = &faces });
        const directions = [_]struct { [4]f32, usize }{
            .{ .{ 1, 0, 0, 0 }, 0 },
            .{ .{ 0, -1, 0, 0 }, 3 },
            .{ .{ 0, 0, -1, 0 }, 5 },
        };
        for (directions) |look| {
            const pixels = try sampling.draw(&device, cube, nearest, look[0]);
            defer testing.allocator.free(pixels);
            try expectSolid(pixels, palette[look[1]]);
        }
    }
    // A volume, looked into at a depth: the sample lands in one slice of four.
    {
        const sampling = try Sampling.init(&device, tex3d_ps, 4, 1);
        var slices: [4 * 4 * 4 * 4]u8 = undefined;
        for (0..4) |z| for (0..16) |t| {
            slices[(z * 16 + t) * 4 ..][0..4].* = palette[z];
        };
        const volume = try device.createTexture(.{ .dimension = .d3, .width = 4, .height = 4, .depth_or_layers = 4, .data = &slices });
        for ([_]u32{ 0, 2, 3 }) |z| {
            const pixels = try sampling.draw(&device, volume, nearest, .{ 0, 0, 0, (@as(f32, @floatFromInt(z)) + 0.5) / 4 });
            defer testing.allocator.free(pixels);
            try expectSolid(pixels, palette[z]);
        }
    }
}

test "a compressed texture is written in blocks and sampled" {
    var device = try warpDevice();
    defer device.deinit();
    if (!device.caps().formatSupport(.bc1_rgba_unorm).sampled) return error.SkipZigTest;

    // BC1 blocks of one colour: two equal RGB565 endpoints and indices of zero.
    const block = struct {
        fn of(comptime rgb565: u16) [8]u8 {
            const endpoint = std.mem.toBytes(std.mem.nativeToLittle(u16, rgb565));
            return endpoint ++ endpoint ++ [4]u8{ 0, 0, 0, 0 };
        }
    }.of;
    const red = block(0xF800);
    const green = block(0x07E0);
    const blue = block(0x001F);
    const white = block(0xFFFF);
    const all_blocks = red ++ green ++ blue ++ blue;

    // 8x8 is four blocks, with a chain of three: 8, 4 and 2 (one block again).
    const texture = try device.createTexture(.{ .width = 8, .height = 8, .mip_levels = 3, .format = .bc1_rgba_unorm, .data = &all_blocks });
    // One block of it, the bottom right one, is written on its own and becomes white.
    try device.writeTexture(texture, .{ .x = 4, .y = 4, .width = 4, .height = 4 }, &white, 0, 0);
    // The whole of the smallest level - a 2x2 image in a 4x4 block - and the one above it.
    try device.writeTexture(texture, .{ .mip = 2 }, &green, 0, 0);
    try device.writeTexture(texture, .{ .mip = 1 }, &red, 0, 0);

    const sampling = try Sampling.init(&device, tex2d_ps, 8, 1);
    const level_zero = try device.createSampler(.nearest);
    {
        const pixels = try sampling.draw(&device, texture, level_zero, .{ 0, 0, 0, 0 });
        defer testing.allocator.free(pixels);
        try expectNear(palette[0], pixelAt(pixels, 8, 1, 1), 8);
        try expectNear(palette[1], pixelAt(pixels, 8, 6, 1), 8);
        try expectNear(palette[2], pixelAt(pixels, 8, 1, 6), 8);
        try expectNear(palette[6], pixelAt(pixels, 8, 6, 6), 8);
    }
    // Each of the two levels below, alone: the clamp to one level is the sampler's range.
    const only_one = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 2, .lod_max = 2 });
    {
        const pixels = try sampling.draw(&device, texture, only_one, .{ 0, 0, 0, 0 });
        defer testing.allocator.free(pixels);
        try expectNear(palette[1], pixelAt(pixels, 8, 3, 3), 8);
    }
    const level_one = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 1, .lod_max = 1 });
    {
        const pixels = try sampling.draw(&device, texture, level_one, .{ 0, 0, 0, 0 });
        defer testing.allocator.free(pixels);
        try expectNear(palette[0], pixelAt(pixels, 8, 3, 3), 8);
    }

    // Level zero is a whole number of blocks on this API, and the levels below it need not be: 20 by 20 goes to
    // 10, 5, 2 and 1, and a box that reaches the edge of the 5x5 level is a block that hangs over it.
    try testing.expectError(error.Unsupported, device.createTexture(.{ .width = 6, .height = 6, .format = .bc1_rgba_unorm }));
    const chain = try device.createTexture(.{ .width = 20, .height = 20, .mip_levels = 0, .format = .bc1_rgba_unorm });
    try device.writeTexture(chain, .{ .mip = 2, .x = 4, .width = 1, .height = 4 }, &red, 0, 0);
    try device.writeTexture(chain, .{ .mip = 2, .x = 4, .y = 4, .width = 1, .height = 1 }, &red, 0, 0);
    try device.writeTexture(chain, .{ .mip = 2 }, &(red ++ green ++ blue ++ white), 0, 0);
}

test "a sampler picks its level, its border and its comparison" {
    var device = try warpDevice();
    defer device.deinit();

    // Four texels wide, three levels, each one colour: which one a sampler reads shows in the pixel.
    const chain = try device.createTexture(.{ .width = 4, .height = 4, .mip_levels = 0 });
    for (0..3) |mip| {
        const size: usize = types.mipExtent(4, @intCast(mip));
        var level: [4 * 4 * 4]u8 = undefined;
        for (0..size * size) |i| level[i * 4 ..][0..4].* = palette[mip];
        try device.writeTexture(chain, .{ .mip = @intCast(mip) }, level[0 .. size * size * 4], 0, 0);
    }
    {
        // A quad that maps one texel to one pixel is level zero; a bias of one makes it level one; without a mip filter, whatever the bias, it is level zero.
        const sampling = try Sampling.init(&device, tex2d_ps, 4, 1);
        const cases = [_]struct { types.SamplerDesc, [4]u8 }{
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest }, palette[0] },
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_bias = 1 }, palette[1] },
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_bias = 1, .lod_max = 0 }, palette[0] },
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .lod_min = 2 }, palette[2] },
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .none, .lod_bias = 1 }, palette[0] },
            // Anisotropy asked for with a nearest filter changes nothing; with a linear one it is clamped to what the device has, and accepted.
            .{ .{ .min_filter = .nearest, .mag_filter = .nearest, .mip_filter = .nearest, .max_anisotropy = 16 }, palette[0] },
        };
        for (cases) |case| {
            const sampler = try device.createSampler(case[0]);
            const pixels = try sampling.draw(&device, chain, sampler, .{ 0, 0, 0, 0 });
            defer testing.allocator.free(pixels);
            try expectSolid(pixels, case[1]);
        }
        _ = try device.createSampler(.{ .max_anisotropy = 16, .mip_filter = .linear });
        _ = try device.createSampler(.{ .wrap_u = .repeat, .wrap_v = .mirror, .wrap_w = .repeat });
    }
    {
        // Beyond the edge is the border colour: the texture is red, and the coordinates run to two.
        const sampling = try Sampling.init(&device, tex2d_ps, 8, 2);
        const red = try device.createTexture(.{ .width = 2, .height = 2, .data = &solidTexels(4, palette[0]) });
        const borders = [_]struct { types.BorderColor, [4]u8 }{
            .{ .transparent_black, .{ 0, 0, 0, 0 } },
            .{ .opaque_black, .{ 0, 0, 0, 255 } },
            .{ .opaque_white, .{ 255, 255, 255, 255 } },
        };
        for (borders) |border| {
            const sampler = try device.createSampler(.{ .min_filter = .nearest, .mag_filter = .nearest, .wrap_u = .border, .wrap_v = .border, .border = border[0] });
            const pixels = try sampling.draw(&device, red, sampler, .{ 0, 0, 0, 0 });
            defer testing.allocator.free(pixels);
            try testing.expectEqual(palette[0], pixelAt(pixels, 8, 1, 1));
            try testing.expectEqual(border[1], pixelAt(pixels, 8, 6, 1));
            try testing.expectEqual(border[1], pixelAt(pixels, 8, 1, 6));
            try testing.expectEqual(border[1], pixelAt(pixels, 8, 6, 6));
        }
    }
    {
        // A shadow map: a depth texture drawn at 0.5 and read back through a comparison sampler.
        const shader = try flat3Shader(&device);
        const depth_only = try flat3Pipeline(&device, shader, .{ .color_format = null, .depth_format = .depth32_float, .depth = .standard });
        const shadow_map = try device.createTexture(.{ .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .sampled = true, .render_target = true } });
        const plane = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0.5, .{ 1, 1, 1, 1 }));
        try drawStrip(&device, .{ .depth = .{ .texture = shadow_map } }, depth_only, plane);

        const shadow_ps =
            \\cbuffer Params : register(b0) { float4 param; };
            \\Texture2D<float> t : register(t0);
            \\SamplerComparisonState c : register(s0);
            \\struct Out { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
            \\float4 main(Out i) : SV_TARGET { float v = t.SampleCmp(c, i.uv, param.x); return float4(v, v, v, 1); }
        ;
        const sampling = try Sampling.init(&device, shadow_ps, 8, 1);
        const compare = try device.createSampler(.{ .compare = .less_equal });
        // The reference goes in through the shader's constant buffer.
        const lit = try sampling.draw(&device, shadow_map, compare, .{ 0.3, 0, 0, 0 });
        defer testing.allocator.free(lit);
        try expectSolid(lit, palette[6]);
        const shaded = try sampling.draw(&device, shadow_map, compare, .{ 0.7, 0, 0, 0 });
        defer testing.allocator.free(shaded);
        try expectSolid(shaded, .{ 0, 0, 0, 255 });
    }
}

test "every uncompressed format is read back as RGBA8" {
    var device = try warpDevice();
    defer device.deinit();

    const half = struct {
        fn bytes(comptime values: []const f32) [values.len * 2]u8 {
            var out: [values.len * 2]u8 = undefined;
            for (values, 0..) |v, i| out[i * 2 ..][0..2].* = std.mem.toBytes(std.mem.nativeToLittle(u16, @bitCast(@as(f16, @floatCast(v)))));
            return out;
        }
    }.bytes;
    const single = struct {
        fn bytes(comptime values: []const f32) [values.len * 4]u8 {
            var out: [values.len * 4]u8 = undefined;
            for (values, 0..) |v, i| out[i * 4 ..][0..4].* = std.mem.toBytes(std.mem.nativeToLittle(u32, @bitCast(v)));
            return out;
        }
    }.bytes;
    const word = struct {
        fn bytes(value: u32) [4]u8 {
            return std.mem.toBytes(std.mem.nativeToLittle(u32, value));
        }
    }.bytes;

    // One texel of each, and what it is as RGBA8: a grey for one channel, red and green for two, blue and alpha where there are none.
    const Case = struct { types.Format, []const u8, [4]u8 };
    const cases = [_]Case{
        .{ .r8_unorm, &.{0x80}, .{ 128, 128, 128, 255 } },
        .{ .rg8_unorm, &.{ 10, 20 }, .{ 10, 20, 0, 255 } },
        .{ .rgba8_unorm, &.{ 1, 2, 3, 4 }, .{ 1, 2, 3, 4 } },
        .{ .rgba8_unorm_srgb, &.{ 1, 2, 3, 4 }, .{ 1, 2, 3, 4 } },
        .{ .bgra8_unorm, &.{ 1, 2, 3, 4 }, .{ 3, 2, 1, 4 } },
        .{ .bgra8_unorm_srgb, &.{ 1, 2, 3, 4 }, .{ 3, 2, 1, 4 } },
        .{ .r16_float, &half(&.{0.5}), .{ 128, 128, 128, 255 } },
        .{ .rg16_float, &half(&.{ 1, 0.25 }), .{ 255, 64, 0, 255 } },
        .{ .rgba16_float, &half(&.{ 1, 0, 0.5, 2 }), .{ 255, 0, 128, 255 } },
        .{ .r32_float, &single(&.{0.5}), .{ 128, 128, 128, 255 } },
        .{ .rg32_float, &single(&.{ 0, 1 }), .{ 0, 255, 0, 255 } },
        .{ .rgba32_float, &single(&.{ 1, 0.5, -1, 0.25 }), .{ 255, 128, 0, 64 } },
        // Ten bits a colour, two of alpha: 1023, 0, 512 and 3.
        .{ .rgb10a2_unorm, &word(1023 | (512 << 20) | (3 << 30)), .{ 255, 0, 128, 255 } },
        // Small floats: red 1.0 is exponent 15, blue 0.5 is exponent 14.
        .{ .rg11b10_float, &word((15 << 6) | (14 << 27)), .{ 255, 0, 128, 255 } },
    };
    for (cases) |case| {
        const support = device.caps().formatSupport(case[0]);
        if (!support.sampled) continue;
        const texture = try device.createTexture(.{ .width = 1, .height = 1, .format = case[0], .data = case[1] });
        const pixels = try readLevel(&device, texture, .{});
        defer testing.allocator.free(pixels);
        try expectNear(case[2], pixels[0..4].*, 1);
    }
}

test "a depth attachment can be a face of a cube, and a level of an array" {
    var device = try warpDevice();
    defer device.deinit();

    const shader = try flat3Shader(&device);
    const depth_only = try flat3Pipeline(&device, shader, .{ .color_format = null, .depth_format = .depth32_float, .depth = .standard });
    const coloured = try flat3Pipeline(&device, shader, .{ .depth_format = .depth32_float, .depth = .standard });
    const coloured_with_stencil = try flat3Pipeline(&device, shader, .{ .depth_format = .depth24_stencil8, .depth = .standard });

    const cube = try device.createTexture(.{ .dimension = .cube, .width = 8, .height = 8, .format = .depth32_float, .usage = .{ .sampled = false, .render_target = true } });
    const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });

    // Face five is written near on its left half; face one is cleared and left alone.
    const near_left = try vertexBuffer(&device, quad3(-1, -1, 0, 1, 0.25, .{ 1, 1, 1, 1 }));
    try drawStrip(&device, .{ .depth = .{ .texture = cube, .layer = 5 } }, depth_only, near_left);

    const far = try vertexBuffer(&device, quad3(-1, -1, 1, 1, 0.75, .{ 1, 0, 0, 1 }));
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } }, .depth = .{ .texture = cube, .layer = 5, .load = .load } }, coloured, far);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[2], pixelAt(pixels, 8, 1, 4));
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 6, 4));
    }
    // Face one was cleared and nothing was written to it, so nothing is in the way there.
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } }, .depth = .{ .texture = cube, .layer = 1 } }, coloured, far);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 1, 4));
    }

    // A level of an array of depth and stencil: the second level of a 16x16 chain is 8x8, the size of the colour it goes with.
    const array = try device.createTexture(.{ .dimension = .d2_array, .width = 16, .height = 16, .depth_or_layers = 2, .mip_levels = 2, .format = .depth24_stencil8, .usage = .{ .sampled = false, .render_target = true } });
    try drawStrip(&device, .{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 1, 1 } }, .depth = .{ .texture = array, .layer = 1, .mip_level = 1 } }, coloured_with_stencil, far);
    {
        const pixels = try readLevel(&device, target, .{});
        defer testing.allocator.free(pixels);
        try testing.expectEqual(palette[0], pixelAt(pixels, 8, 4, 4));
    }
}

const user32 = struct {
    extern "user32" fn CreateWindowExA(u32, [*:0]const u8, [*:0]const u8, u32, i32, i32, i32, i32, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.winapi) ?*anyopaque;
    extern "user32" fn DestroyWindow(?*anyopaque) callconv(.winapi) c_int;
    /// `WS_POPUP`: no frame, and with no `WS_VISIBLE` never shown.
    const ws_popup: u32 = 0x80000000;
};

/// What is in a surface's back buffer, as RGBA8. Nothing in the library reads a
/// surface - a window is for looking at - so the test goes to the backend.
fn readBackBuffer(device: *Device, surface: types.Surface) ![]u8 {
    const self = cast(device.impl);
    const res = as(SurfaceRes, device.surfaces.get(surface).?.native);
    const staging = self.raw.createTexture2D(.{
        .width = res.width,
        .height = res.height,
        .format = native_table.get(surface_format).?.resource,
        .usage = .staging,
        .cpu_access = .{ .read = true },
    }) catch return error.Failed;
    defer _ = com.release(staging);
    self.context.vtable.CopyResource(self.context, asResource(staging), asResource(res.back_buffer));

    var mapped: MappedSubresource = .{};
    try self.context.vtable.Map(self.context, asResource(staging), 0, .read, 0, &mapped).check();
    defer self.context.vtable.Unmap(self.context, asResource(staging), 0);
    const data = mapped.data orelse return error.Failed;
    const row = @as(usize, res.width) * 4;
    const pixels = try testing.allocator.alloc(u8, row * res.height);
    for (0..res.height) |y| @memcpy(pixels[y * row ..][0..row], data[y * mapped.row_pitch ..][0..row]);
    return pixels;
}

test "a pass draws into a surface, and a multisampled one resolves into it" {
    // A window that is never shown, for the swap chain to belong to. A machine that will not make one - no desktop
    // to make it on - skips, like a machine with no device.
    const window = user32.CreateWindowExA(0, "STATIC", "fluxion-rhi", user32.ws_popup, 0, 0, 32, 32, null, null, null, null) orelse return error.SkipZigTest;
    defer _ = user32.DestroyWindow(window);

    var device = try warpDevice();
    defer device.deinit();
    if (!device.caps().formatSupport(surface_format).supportsSamples(4)) return error.SkipZigTest;
    const surface = device.createSurface(.{ .native_window = @intFromPtr(window), .width = 32, .height = 32, .present_mode = .disabled }) catch return error.SkipZigTest;

    // A pass that clears it is green all over.
    {
        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface }, .clear_color = .{ 0, 1, 0, 1 } } });
        try cmd.endPass();
        try device.submit();
        const pixels = try readBackBuffer(&device, surface);
        defer testing.allocator.free(pixels);
        try expectSolid(pixels, palette[1]);
    }

    // The red triangle of the multisampled test, over blue, resolved into it.
    const shader = try flat3Shader(&device);
    const pipeline = try flat3Pipeline(&device, shader, .{ .topology = .triangles, .samples = 4 });
    const msaa = try device.createTexture(.{ .width = 32, .height = 32, .samples = 4, .usage = .{ .sampled = false, .render_target = true } });
    const red: [4]f32 = .{ 1, 0, 0, 1 };
    const triangle = [_]Vertex3{
        .{ .position = .{ -0.8, -0.8, 0 }, .colour = red },
        .{ .position = .{ 0, 0.8, 0 }, .colour = red },
        .{ .position = .{ 0.8, -0.8, 0 }, .colour = red },
    };
    const buffer = try vertexBuffer(&device, triangle);
    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = msaa }, .clear_color = .{ 0, 0, 1, 1 }, .resolve = .{ .surface = surface } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try readBackBuffer(&device, surface);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(palette[0], pixelAt(pixels, 32, 16, 24));
    try testing.expectEqual(palette[2], pixelAt(pixels, 32, 1, 1));
    var blended: usize = 0;
    for (0..32 * 32) |i| {
        const p = pixels[i * 4 ..][0..4];
        if (p[0] > 0 and p[0] < 255 and p[2] > 0 and p[2] < 255) blended += 1;
    }
    try testing.expect(blended > 10);
}
