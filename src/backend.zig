// SPDX-License-Identifier: BSD-2-Clause

//! What a backend has to answer to.
//!
//! A backend is a vtable and an opaque pointer, chosen when the device is made
//! and never looked at again by anything outside `Device`. Adding one - Vulkan,
//! Metal, Direct3D 12 - is a new file that fills this table and an `Opener` that
//! says how to open it. `Device.initWith` takes any opener, so a backend does not
//! have to live in this library, and nothing has to be added to `Device`: the
//! built-in ones are listed by `Device.opener`, and a program, or a registry it
//! keeps, can hold openers of its own. Nothing a program calls changes.
//!
//! Every `native` here is whatever the backend allocated for a resource;
//! `Device` stores it behind a handle and hands it back on every call. The
//! backend resolves handles it finds inside a command list through the
//! `Device` it is given, which is the one place the two meet.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const math = @import("fluxion_math");

const types = @import("types.zig");
const commands = @import("commands.zig");
const Device = @import("Device.zig");

pub const Impl = *anyopaque;
pub const Native = *anyopaque;
pub const Error = types.Error;

pub const Vtable = struct {
    deinit: *const fn (Impl) void,
    info: *const fn (Impl) types.Info,
    /// What this device can do, asked once when it opens. `Device` keeps the
    /// answer and checks every request against it, so a backend is only
    /// asked for what it said it could do.
    caps: *const fn (Impl) types.Caps,

    createBuffer: *const fn (Impl, types.BufferDesc) Error!Native,
    destroyBuffer: *const fn (Impl, Native) void,
    updateBuffer: *const fn (Impl, Native, offset: usize, bytes: []const u8) Error!void,

    /// `desc` has been checked against `caps`, and its `mip_levels` is a
    /// count, never zero. `desc.data` is level zero of everything.
    createTexture: *const fn (Impl, types.TextureDesc) Error!Native,
    destroyTexture: *const fn (Impl, Native) void,
    /// Replace a box of texels. The region has no zero sizes left in it and
    /// lies inside the texture; the bytes hold at least what it needs at the
    /// pitches given, which are in bytes and never zero. `slice_pitch` is
    /// from one slice or layer to the next, and is meaningful only when the
    /// region's depth is more than one. Rows of a compressed format are
    /// rows of blocks.
    writeTexture: *const fn (Impl, Native, region: types.TextureRegion, bytes: []const u8, row_pitch: usize, slice_pitch: usize) Error!void,
    /// One mip level of one layer, face or slice, as RGBA, eight bits a
    /// channel, top row first, tightly packed: `width * height * 4` bytes of
    /// that level's size. The caller owns the bytes. Only asked of a
    /// colour format that is not compressed.
    ///
    /// **"Top row first" is the picture, not the memory.** An image that was
    /// written from memory reads back in the order it was written, first
    /// row first. An image that a pass drew into reads back with the top of
    /// what was drawn first - which on OpenGL and WebGL, where the framebuffer
    /// is bottom-up, means turning over exactly the images a pass has drawn
    /// into and no others. A backend whose storage is already top-down turns
    /// nothing over.
    ///
    /// **Writing is whole-image.** Writing any box into an image that was
    /// drawn into makes the whole image a written one from then on, so what
    /// was drawn around the box reads back the other way up on a backend
    /// that turns drawn images over. Redraw the image, or write all of it.
    readTexture: *const fn (Impl, Native, sub: types.Subresource, gpa: Allocator) Error![]u8,

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

/// What opening a backend gives back: its state, and its table.
pub const Opened = struct { Impl, *const Vtable };

/// A live Vulkan `VkInstance` and its `vkGetInstanceProcAddr`, both as the
/// caller would pass them to `fluxion-platform`'s `Window.createVulkanSurface`
/// - whose `instance` parameter is this same `usize` - to make a
/// `VkSurfaceKHR` for `SurfaceDesc.vulkan_surface`. See
/// `Device.vulkanInstanceHandles`.
pub const VulkanInstanceHandles = struct {
    instance: usize,
    get_instance_proc_addr: *const anyopaque,
};

/// A backend's constructor: what `Device.init` and `Device.initWith` call.
pub const Open = *const fn (gpa: Allocator, desc: types.DeviceDesc) Error!Opened;

/// A way to open a device on one backend.
///
/// It is plain data, so a program can keep them in a list, or a registry can
/// hold one per name. `Device.opener` has the ones this build brings; a backend
/// written elsewhere - Metal, say - makes its own.
pub const Opener = struct {
    /// What it is called: "gl", "d3d11", "metal". Borrowed by every device it
    /// opens, so it has to live as long as they do.
    name: []const u8,
    /// Which of the built-in backends this is, or `.other`.
    tag: types.Backend = .other,
    /// The clip space a projection for its devices is built for.
    clip: math.Clip,
    open: Open,
};
