// SPDX-License-Identifier: BSD-2-Clause

//! What a backend has to answer to.
//!
//! A backend is a vtable and an opaque pointer, chosen when the device is made
//! and never looked at again by anything outside `Device`. Adding one - Vulkan,
//! Metal, Direct3D 12 - is a new file under `backend/` that fills this table,
//! and one more arm in `Device.init`. Nothing a program calls changes.
//!
//! Every `native` here is whatever the backend allocated for a resource;
//! `Device` stores it behind a handle and hands it back on every call. The
//! backend resolves handles it finds inside a command list through the
//! `Device` it is given, which is the one place the two meet.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const types = @import("types.zig");
const commands = @import("commands.zig");
const Device = @import("Device.zig");

pub const Impl = *anyopaque;
pub const Native = *anyopaque;
pub const Error = types.Error;

pub const Vtable = struct {
    deinit: *const fn (Impl) void,
    info: *const fn (Impl) types.Info,

    createBuffer: *const fn (Impl, types.BufferDesc) Error!Native,
    destroyBuffer: *const fn (Impl, Native) void,
    updateBuffer: *const fn (Impl, Native, offset: usize, bytes: []const u8) Error!void,

    createTexture: *const fn (Impl, types.TextureDesc) Error!Native,
    destroyTexture: *const fn (Impl, Native) void,
    updateTexture: *const fn (Impl, Native, bytes: []const u8, row_pitch: usize) Error!void,
    /// RGBA, eight bits a channel, top row first, tightly packed. The caller
    /// owns the bytes.
    readTexture: *const fn (Impl, Native, gpa: Allocator) Error![]u8,

    createSampler: *const fn (Impl, types.SamplerDesc) Error!Native,
    destroySampler: *const fn (Impl, Native) void,

    /// Compile and link. Complaints go to `log`, whether or not it failed.
    createShader: *const fn (Impl, types.ShaderDesc, log: *Io.Writer) Error!Native,
    destroyShader: *const fn (Impl, Native) void,

    createPipeline: *const fn (Impl, types.PipelineDesc, shader: Native, log: *Io.Writer) Error!Native,
    destroyPipeline: *const fn (Impl, Native) void,

    createSurface: *const fn (Impl, types.SurfaceDesc) Error!Native,
    destroySurface: *const fn (Impl, Native) void,
    resizeSurface: *const fn (Impl, Native, width: u32, height: u32) Error!void,
    surfaceSize: *const fn (Impl, Native) [2]u32,
    present: *const fn (Impl, Native, vsync: bool) Error!void,

    /// Execute a list that `Device` has already validated: every handle in it
    /// is alive, every draw is inside a pass, every pass is closed.
    submit: *const fn (Impl, *Device, []const commands.Command) Error!void,
};

/// A backend's constructor: what `Device.init` calls.
pub const Open = *const fn (gpa: Allocator, desc: types.DeviceDesc) Error!struct { Impl, *const Vtable };
