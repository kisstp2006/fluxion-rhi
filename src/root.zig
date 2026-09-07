// SPDX-License-Identifier: BSD-2-Clause

//! Fluxion RHI - one way to draw, on whichever API the machine has.
//!
//! A render hardware interface: buffers, textures, samplers, shaders,
//! pipelines and passes, described once as plain values and executed by a
//! backend chosen when the device is made. Two backends today - OpenGL 3.3
//! and Direct3D 11 - and a third, `none`, that accepts everything and draws
//! nothing, for the machines that have no GPU and the tests that need none.
//!
//!   `Device`     one backend, the resources on it, and the frame being recorded
//!   `types`      everything a program says to a device, backend-free
//!   `commands`   a frame written down before it is drawn
//!   `backend`    what a backend has to answer to - the seam a new one is written against
//!
//! ```zig
//! const rhi = @import("fluxion_rhi");
//!
//! var device = try rhi.Device.init(gpa, .{ .gl = window.hooks() });
//! defer device.deinit();
//!
//! const surface = try device.createSurface(.{});
//! const shader = try device.createShader(.{ .glsl = .{ .vertex = vs, .fragment = fs } });
//! const pipeline = try device.createPipeline(.{
//!     .shader = shader,
//!     .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
//!     .buffers = &.{.{ .stride = 8 }},
//!     .blend = .alpha,
//! });
//!
//! const cmd = device.begin();
//! try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface } } });
//! try cmd.setPipeline(pipeline);
//! try cmd.setVertexBuffer(0, quad, 0);
//! try cmd.draw(.{ .vertex_count = 4 });
//! try cmd.endPass();
//! try device.submit();
//! try device.present(surface);
//! ```
//!
//! **What it is for, first.** 2D: sprites, text, UI, tilemaps - instanced
//! quads, textures, blending, scissor rectangles, an orthographic matrix. The
//! shape is not 2D, though: depth states, depth attachments, cull modes and
//! `Clip` are all here, so a 3D renderer is more of the same calls and not a
//! different library.
//!
//! **Where the seams are.** A window comes from outside - `fluxion-platform`,
//! or anything that can hand over an `HWND` or a `getProcAddress`. Shaders
//! come as source in the backend's own language; the contract between the
//! two languages is in `ShaderDesc`, and a cross-compiler is a thing that
//! produces both without this library changing. A third backend is a file
//! under `backend/` and an arm in `Device.init`.
//!
//! Nothing here allocates except through the allocator handed to
//! `Device.init`, and nothing here opens a window or writes a file.

const std = @import("std");

pub const types = @import("types.zig");
pub const commands = @import("commands.zig");
pub const backend = @import("backend.zig");
pub const resources = @import("resources.zig");

pub const Device = @import("Device.zig");
pub const CommandList = commands.CommandList;
pub const Command = commands.Command;

pub const Backend = types.Backend;
pub const Select = types.Select;
pub const Error = types.Error;
pub const Info = types.Info;

pub const Buffer = types.Buffer;
pub const Texture = types.Texture;
pub const Sampler = types.Sampler;
pub const Shader = types.Shader;
pub const Pipeline = types.Pipeline;
pub const Surface = types.Surface;

pub const Color = types.Color;
pub const Extent = types.Extent;
pub const Format = types.Format;
pub const VertexFormat = types.VertexFormat;
pub const IndexFormat = types.IndexFormat;
pub const Topology = types.Topology;
pub const BufferKind = types.BufferKind;
pub const BufferDesc = types.BufferDesc;
pub const TextureUsage = types.TextureUsage;
pub const TextureDesc = types.TextureDesc;
pub const Filter = types.Filter;
pub const Wrap = types.Wrap;
pub const SamplerDesc = types.SamplerDesc;
pub const ShaderStages = types.ShaderStages;
pub const ShaderDesc = types.ShaderDesc;
pub const VertexStep = types.VertexStep;
pub const VertexBufferLayout = types.VertexBufferLayout;
pub const VertexAttribute = types.VertexAttribute;
pub const BlendFactor = types.BlendFactor;
pub const BlendOp = types.BlendOp;
pub const BlendState = types.BlendState;
pub const CompareFn = types.CompareFn;
pub const DepthState = types.DepthState;
pub const CullMode = types.CullMode;
pub const FrontFace = types.FrontFace;
pub const PipelineDesc = types.PipelineDesc;
pub const Viewport = types.Viewport;
pub const Rect = types.Rect;
pub const LoadOp = types.LoadOp;
pub const RenderTarget = types.RenderTarget;
pub const ColorAttachment = types.ColorAttachment;
pub const DepthAttachment = types.DepthAttachment;
pub const RenderPassDesc = types.RenderPassDesc;
pub const GlHooks = types.GlHooks;
pub const GlProc = types.GlProc;
pub const DeviceDesc = types.DeviceDesc;
pub const SurfaceDesc = types.SurfaceDesc;

/// Which backends this build could open. See `Device.available`.
pub const available = Device.available;

test {
    _ = types;
    _ = commands;
    _ = backend;
    _ = resources;
    _ = Device;
    _ = @import("backend/none.zig");
    _ = @import("backend/gl.zig");
    if (@import("builtin").os.tag == .windows) _ = @import("backend/d3d11.zig");
}
