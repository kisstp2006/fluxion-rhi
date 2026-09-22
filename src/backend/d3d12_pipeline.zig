// SPDX-License-Identifier: BSD-2-Clause

//! Root signatures and pipeline states: `ID3D12RootSignature`,
//! `ID3D12PipelineState`, `D3D12_ROOT_PARAMETER` and the rest of what
//! `D3D12SerializeRootSignature` and `CreateGraphicsPipelineState` take.
//!
//! `fluxion-d3d`'s own `d3d12.RootSignatureDesc` already has the right shape
//! for the empty signature it tests with; `serializeGraphicsRootSignature`
//! below builds one with real parameters in it without changing that type -
//! `parameters` is `?*const anyopaque` there and stays that way, cast into on
//! the way in, exactly as a `void*` in the C header is.

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

const ID3D12Device = d3d12.ID3D12Device;
const ID3D12DeviceChild = d3d12.ID3D12DeviceChild;
const ID3D12Pageable = d3d12.ID3D12Pageable;
const Format = resource.Format;
const SampleDesc = resource.SampleDesc;

const Bool = c_int;

// -------------------------------------------------------------------------
// Root signature parameters
// -------------------------------------------------------------------------

/// `D3D12_DESCRIPTOR_RANGE_TYPE`.
pub const DescriptorRangeType = enum(u32) { srv = 0, uav = 1, cbv = 2, sampler = 3 };

/// `D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND`: this range starts right after the
/// one before it in the table, which is what every range in this backend's
/// one root signature wants.
pub const descriptor_range_offset_append: u32 = 0xFFFFFFFF;

/// `D3D12_DESCRIPTOR_RANGE`.
pub const DescriptorRange = extern struct {
    range_type: DescriptorRangeType,
    num_descriptors: u32,
    base_shader_register: u32,
    register_space: u32 = 0,
    offset_in_descriptors_from_table_start: u32 = descriptor_range_offset_append,
};

/// `D3D12_ROOT_DESCRIPTOR_TABLE`.
pub const RootDescriptorTable = extern struct {
    num_descriptor_ranges: u32,
    descriptor_ranges: [*]const DescriptorRange,
};

/// `D3D12_ROOT_CONSTANTS`. Declared for layout completeness - one of the
/// three shapes `RootParameterUnion` must be exactly as wide as - and never
/// populated: this backend passes uniforms through constant buffers, not
/// inline root constants.
pub const RootConstants = extern struct {
    shader_register: u32,
    register_space: u32 = 0,
    num_32bit_values: u32,
};

/// `D3D12_ROOT_DESCRIPTOR`.
pub const RootDescriptor = extern struct {
    shader_register: u32,
    register_space: u32 = 0,
};

/// `D3D12_ROOT_PARAMETER_TYPE`.
pub const RootParameterType = enum(u32) {
    descriptor_table = 0,
    constants_32bit = 1,
    cbv = 2,
    srv = 3,
    uav = 4,
};

/// `D3D12_SHADER_VISIBILITY`.
pub const ShaderVisibility = enum(u32) {
    all = 0,
    vertex = 1,
    hull = 2,
    domain = 3,
    geometry = 4,
    pixel = 5,
};

/// `D3D12_ROOT_PARAMETER`. `ParameterType` leads, then an anonymous union
/// whose widest member is `DescriptorTable` (holds a pointer, needs 8-byte
/// alignment, so the union is 16 bytes), then `ShaderVisibility` - modelled
/// as a real `extern union` of the three variants, the same pattern
/// `d3d12_command.zig`'s `ResourceBarrier` uses.
pub const RootParameter = extern struct {
    parameter_type: RootParameterType,
    u: extern union {
        descriptor_table: RootDescriptorTable,
        constants: RootConstants,
        descriptor: RootDescriptor,
    },
    shader_visibility: ShaderVisibility = .all,

    pub fn table(ranges: []const DescriptorRange, visibility: ShaderVisibility) RootParameter {
        return .{
            .parameter_type = .descriptor_table,
            .u = .{ .descriptor_table = .{ .num_descriptor_ranges = @intCast(ranges.len), .descriptor_ranges = ranges.ptr } },
            .shader_visibility = visibility,
        };
    }

    pub fn cbv(shader_register: u32, visibility: ShaderVisibility) RootParameter {
        return .{
            .parameter_type = .cbv,
            .u = .{ .descriptor = .{ .shader_register = shader_register } },
            .shader_visibility = visibility,
        };
    }
};

/// Serialise a root signature with real parameters. `d3d12.RootSignatureDesc.parameters`
/// is `?*const anyopaque`, exactly the shape a `const void*` field in the
/// header has, so a typed array goes in with one cast and the struct that
/// already has a passing test in `fluxion-d3d` is untouched.
pub fn serializeGraphicsRootSignature(lib: d3d.D3d12, params: []const RootParameter, flags: u32) Error!*d3d12.ID3DBlob {
    const serialize = lib.entries.D3D12SerializeRootSignature orelse return error.Unsupported;
    const desc: d3d12.RootSignatureDesc = .{
        .parameter_count = @intCast(params.len),
        .parameters = if (params.len == 0) null else @ptrCast(params.ptr),
        .flags = flags,
    };
    var blob: ?*d3d12.ID3DBlob = null;
    var errors: ?*d3d12.ID3DBlob = null;
    const result = serialize(&desc, .v1_0, &blob, &errors);
    defer if (errors) |e| {
        _ = com.release(e);
    };
    return com.received(d3d12.ID3DBlob, result, blob);
}

pub fn createRootSignature(device: *ID3D12Device, node_mask: u32, blob: []const u8) Error!*ID3D12RootSignature {
    const create = resource.slot(*const fn (*ID3D12Device, u32, [*]const u8, usize, *const Guid, *?*anyopaque) callconv(.winapi) Hresult, device.vtable.CreateRootSignature);
    var raw: ?*anyopaque = null;
    const result = create(device, node_mask, blob.ptr, blob.len, com.iidOf(ID3D12RootSignature), &raw);
    return com.received(ID3D12RootSignature, result, raw);
}

// -------------------------------------------------------------------------
// Pipeline state
// -------------------------------------------------------------------------

/// `D3D12_SHADER_BYTECODE`.
pub const ShaderBytecode = extern struct {
    bytecode: ?[*]const u8 = null,
    length: usize = 0,

    pub fn of(bytes: []const u8) ShaderBytecode {
        return .{ .bytecode = bytes.ptr, .length = bytes.len };
    }
};

/// `D3D12_STREAM_OUTPUT_DESC`. Every field at its zero/null default: this
/// backend never uses stream output, and the struct exists only so
/// `GraphicsPipelineStateDesc` has the right byte width at the right offset.
pub const StreamOutputDesc = extern struct {
    so_declaration: ?*const anyopaque = null,
    num_entries: u32 = 0,
    buffer_strides: ?*const anyopaque = null,
    num_strides: u32 = 0,
    rasterized_stream: u32 = 0,
};

/// `D3D12_BLEND`. Values shared with Direct3D 11's `D3D11_BLEND` (see
/// `d3d11.zig`).
pub const Blend = enum(u32) {
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

pub const BlendOp = enum(u32) { add = 1, subtract = 2, rev_subtract = 3, min = 4, max = 5 };

/// `D3D12_LOGIC_OP`. Only `clear` (zero) is ever used, as the
/// required-but-inert value while `LogicOpEnable` is false.
pub const LogicOp = enum(u32) { clear = 0 };

/// `D3D12_COLOR_WRITE_ENABLE_ALL`.
pub const color_write_all: u8 = 0x0F;

/// `D3D12_RENDER_TARGET_BLEND_DESC`.
pub const RenderTargetBlendDesc = extern struct {
    blend_enable: Bool = 0,
    logic_op_enable: Bool = 0,
    src_blend: Blend = .one,
    dest_blend: Blend = .zero,
    blend_op: BlendOp = .add,
    src_blend_alpha: Blend = .one,
    dest_blend_alpha: Blend = .zero,
    blend_op_alpha: BlendOp = .add,
    logic_op: LogicOp = .clear,
    render_target_write_mask: u8 = color_write_all,
};

/// `D3D12_BLEND_DESC`.
pub const BlendDesc = extern struct {
    alpha_to_coverage_enable: Bool = 0,
    independent_blend_enable: Bool = 0,
    render_target: [8]RenderTargetBlendDesc = @splat(.{}),
};

/// `D3D12_FILL_MODE`.
pub const FillMode = enum(u32) { wireframe = 2, solid = 3 };

/// `D3D12_CULL_MODE`.
pub const CullMode = enum(u32) { none = 1, front = 2, back = 3 };

/// `D3D12_CONSERVATIVE_RASTERIZATION_MODE`.
pub const ConservativeRasterizationMode = enum(u32) { off = 0, on = 1 };

/// `D3D12_RASTERIZER_DESC`.
pub const RasterizerDesc = extern struct {
    fill_mode: FillMode = .solid,
    cull_mode: CullMode = .none,
    front_counter_clockwise: Bool = 1,
    depth_bias: i32 = 0,
    depth_bias_clamp: f32 = 0,
    slope_scaled_depth_bias: f32 = 0,
    depth_clip_enable: Bool = 1,
    multisample_enable: Bool = 0,
    antialiased_line_enable: Bool = 0,
    forced_sample_count: u32 = 0,
    conservative_raster: ConservativeRasterizationMode = .off,
};

/// `D3D12_DEPTH_WRITE_MASK`.
pub const DepthWriteMask = enum(u32) { zero = 0, all = 1 };

/// `D3D12_COMPARISON_FUNC`, re-declared here rather than imported from
/// `d3d12_resource.zig`: a pipeline state and a sampler both read the same
/// eight values, and neither module needs the other for anything else.
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

pub const StencilOp = enum(u32) { keep = 1, zero = 2, replace = 3, incr_sat = 4, decr_sat = 5, invert = 6, incr = 7, decr = 8 };

/// `D3D12_DEPTH_STENCILOP_DESC`.
pub const DepthStencilOpDesc = extern struct {
    stencil_fail_op: StencilOp = .keep,
    stencil_depth_fail_op: StencilOp = .keep,
    stencil_pass_op: StencilOp = .keep,
    stencil_func: ComparisonFunc = .always,
};

/// `D3D12_DEPTH_STENCIL_DESC`. Always `depth_enable = false` in this MVP -
/// there is no depth attachment in scope - but declared in full because a
/// pipeline state description always carries one.
pub const DepthStencilDesc = extern struct {
    depth_enable: Bool = 0,
    depth_write_mask: DepthWriteMask = .all,
    depth_func: ComparisonFunc = .less,
    stencil_enable: Bool = 0,
    stencil_read_mask: u8 = 0xFF,
    stencil_write_mask: u8 = 0xFF,
    front_face: DepthStencilOpDesc = .{},
    back_face: DepthStencilOpDesc = .{},
};

/// `D3D12_INPUT_CLASSIFICATION`.
pub const InputClassification = enum(u32) { per_vertex_data = 0, per_instance_data = 1 };

/// `D3D12_INPUT_ELEMENT_DESC`.
pub const InputElementDesc = extern struct {
    semantic_name: [*:0]const u8,
    semantic_index: u32 = 0,
    format: Format,
    input_slot: u32 = 0,
    aligned_byte_offset: u32 = 0,
    input_slot_class: InputClassification = .per_vertex_data,
    instance_data_step_rate: u32 = 0,
};

/// `D3D12_INPUT_LAYOUT_DESC`.
pub const InputLayoutDesc = extern struct {
    elements: ?[*]const InputElementDesc = null,
    count: u32 = 0,
};

/// `D3D12_INDEX_BUFFER_STRIP_CUT_VALUE`.
pub const IndexBufferStripCutValue = enum(u32) { disabled = 0, cut_0xffff = 1, cut_0xffffffff = 2 };

/// `D3D12_PRIMITIVE_TOPOLOGY_TYPE`: the coarse family a pipeline state is
/// built for. Not to be confused with `d3d12_command.zig`'s finer
/// `PrimitiveTopology`, which `IASetPrimitiveTopology` takes - a pipeline
/// built with `.triangle` accepts a list or a strip at draw time.
pub const PrimitiveTopologyType = enum(u32) { undefined = 0, point = 1, line = 2, triangle = 3, patch = 4 };

/// `D3D12_CACHED_PIPELINE_STATE`. Always empty: this backend never seeds a
/// pipeline from a cache blob.
pub const CachedPipelineState = extern struct {
    cached_blob: ?*const anyopaque = null,
    cached_blob_size_in_bytes: usize = 0,
};

/// `D3D12_PIPELINE_STATE_FLAGS`.
pub const PipelineStateFlags = packed struct(u32) {
    tool_debug: bool = false,
    _reserved: u31 = 0,
};

/// `D3D12_GRAPHICS_PIPELINE_STATE_DESC`. Field order is the header's exactly -
/// `NodeMask` comes after `SampleDesc`, not before it, which is easy to get
/// backwards by analogy with `D3D12_COMMAND_QUEUE_DESC`.
pub const GraphicsPipelineStateDesc = extern struct {
    root_signature: ?*ID3D12RootSignature = null,
    vs: ShaderBytecode = .{},
    ps: ShaderBytecode = .{},
    ds: ShaderBytecode = .{},
    hs: ShaderBytecode = .{},
    gs: ShaderBytecode = .{},
    stream_output: StreamOutputDesc = .{},
    blend_state: BlendDesc = .{},
    sample_mask: u32 = 0xFFFFFFFF,
    rasterizer_state: RasterizerDesc = .{},
    depth_stencil_state: DepthStencilDesc = .{},
    input_layout: InputLayoutDesc = .{},
    ib_strip_cut_value: IndexBufferStripCutValue = .disabled,
    primitive_topology_type: PrimitiveTopologyType = .triangle,
    num_render_targets: u32 = 0,
    rtv_formats: [8]Format = @splat(.unknown),
    dsv_format: Format = .unknown,
    sample: SampleDesc = .{},
    node_mask: u32 = 0,
    cached_pso: CachedPipelineState = .{},
    flags: PipelineStateFlags = .{},
};

// -------------------------------------------------------------------------
// The interfaces
// -------------------------------------------------------------------------

/// `ID3D12RootSignature`. Adds nothing of its own over `ID3D12DeviceChild`.
pub const ID3D12RootSignature = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{C54A6B66-72DF-4EE8-8BE5-A946A1429214}");

    pub const VTable = extern struct {
        base: ID3D12DeviceChild.VTable,
    };
};

/// `ID3D12PipelineState`. `GetCachedBlob` is left opaque; this backend never
/// caches a PSO to disk.
pub const ID3D12PipelineState = extern struct {
    vtable: *const VTable,

    pub const iid = Guid.parseComptime("{765A30F3-F624-4C6F-A828-ACE948622445}");

    pub const VTable = extern struct {
        base: ID3D12Pageable.VTable,
        GetCachedBlob: *const anyopaque,
    };
};

pub fn createGraphicsPipelineState(device: *ID3D12Device, desc: *const GraphicsPipelineStateDesc) Error!*ID3D12PipelineState {
    const create = resource.slot(*const fn (*ID3D12Device, *const GraphicsPipelineStateDesc, *const Guid, *?*anyopaque) callconv(.winapi) Hresult, device.vtable.CreateGraphicsPipelineState);
    var raw: ?*anyopaque = null;
    const result = create(device, desc, com.iidOf(ID3D12PipelineState), &raw);
    return com.received(ID3D12PipelineState, result, raw);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "the descriptions the runtime reads are shaped as it expects" {
    try testing.expectEqual(@as(usize, 20), @sizeOf(DescriptorRange));
    try testing.expectEqual(@as(usize, 32), @sizeOf(RootParameter));
    try testing.expectEqual(@as(usize, 40), @sizeOf(RenderTargetBlendDesc));
    try testing.expectEqual(@as(usize, 328), @sizeOf(BlendDesc));
    try testing.expectEqual(@as(usize, 44), @sizeOf(RasterizerDesc));
    try testing.expectEqual(@as(usize, 52), @sizeOf(DepthStencilDesc));
    try testing.expectEqual(@as(usize, 32), @sizeOf(InputElementDesc));
    try testing.expectEqual(@as(u32, 5768), resource.shader_4_component_mapping_identity);
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

test "a root signature with real parameters, serialised and created" {
    var lib = try loadOrSkip();
    defer lib.unload();
    var dxgi_lib = d3d.Dxgi.load() catch return error.SkipZigTest;
    defer dxgi_lib.unload();
    const device = try warpDeviceOrSkip(lib, &dxgi_lib);
    defer _ = com.release(device);

    const ranges = [_]DescriptorRange{
        // A different register (`t0`) from the root CBV below (`b0`): two root
        // parameters binding the same register and space is an invalid
        // signature, and this is exercising two real parameters, not one.
        .{ .range_type = .srv, .num_descriptors = 4, .base_shader_register = 0 },
    };
    const params = [_]RootParameter{
        RootParameter.cbv(0, .all),
        RootParameter.table(&ranges, .pixel),
    };

    var blob = try serializeGraphicsRootSignature(lib, &params, d3d12.root_signature_allow_input_assembler_input_layout);
    defer _ = com.release(blob);

    const root_signature = try createRootSignature(device, 0, blob.bytes());
    defer _ = com.release(root_signature);
}
