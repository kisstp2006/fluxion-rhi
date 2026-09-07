// SPDX-License-Identifier: BSD-2-Clause

//! A frame, written down before it is drawn.
//!
//! Nothing here touches a driver. A `CommandList` is an array of tagged
//! unions, and `Device.submit` hands it to the backend, which walks it once.
//! On OpenGL and Direct3D 11 that walk *is* the drawing; on Vulkan or
//! Direct3D 12 it would be the recording of a real command buffer, with the
//! same list going in - which is why the list exists at all, rather than each
//! call going straight through.
//!
//! Recording cannot fail except for memory. Whether the commands make sense -
//! a draw outside a pass, a handle that is dead - is checked once, at submit,
//! where the answer names the command.

const std = @import("std");
const Allocator = std.mem.Allocator;

const types = @import("types.zig");

pub const Command = union(enum) {
    begin_pass: types.RenderPassDesc,
    end_pass,
    set_pipeline: types.Pipeline,
    set_viewport: types.Viewport,
    /// Null is "the whole attachment", which is also the state a pass begins in.
    set_scissor: ?types.Rect,
    set_vertex_buffer: VertexBinding,
    set_index_buffer: IndexBinding,
    set_uniform_buffer: UniformBinding,
    set_texture: TextureBinding,
    draw: Draw,
    draw_indexed: DrawIndexed,

    pub const VertexBinding = struct {
        slot: u32,
        buffer: types.Buffer,
        /// Bytes into the buffer where vertex zero is.
        offset: u32 = 0,
    };

    pub const IndexBinding = struct {
        buffer: types.Buffer,
        format: types.IndexFormat,
    };

    pub const UniformBinding = struct {
        slot: u32,
        buffer: types.Buffer,
    };

    pub const TextureBinding = struct {
        slot: u32,
        texture: types.Texture,
        sampler: types.Sampler,
    };

    pub const Draw = struct {
        vertex_count: u32,
        instance_count: u32 = 1,
        first_vertex: u32 = 0,
    };

    pub const DrawIndexed = struct {
        index_count: u32,
        instance_count: u32 = 1,
        first_index: u32 = 0,
        /// Added to every index before it reads the vertex buffer. On OpenGL
        /// 3.3 this and instancing cannot be combined; see `Device`.
        base_vertex: i32 = 0,
    };
};

pub const CommandList = struct {
    gpa: Allocator,
    items: std.ArrayListUnmanaged(Command) = .empty,

    pub fn init(gpa: Allocator) CommandList {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *CommandList) void {
        self.items.deinit(self.gpa);
        self.* = undefined;
    }

    /// Forget everything recorded, keeping the memory.
    pub fn reset(self: *CommandList) void {
        self.items.clearRetainingCapacity();
    }

    pub fn commands(self: *const CommandList) []const Command {
        return self.items.items;
    }

    fn push(self: *CommandList, command: Command) Allocator.Error!void {
        try self.items.append(self.gpa, command);
    }

    /// Point everything that follows at an attachment, and clear or keep it.
    pub fn beginPass(self: *CommandList, desc: types.RenderPassDesc) Allocator.Error!void {
        try self.push(.{ .begin_pass = desc });
    }

    pub fn endPass(self: *CommandList) Allocator.Error!void {
        try self.push(.end_pass);
    }

    pub fn setPipeline(self: *CommandList, pipeline: types.Pipeline) Allocator.Error!void {
        try self.push(.{ .set_pipeline = pipeline });
    }

    pub fn setViewport(self: *CommandList, viewport: types.Viewport) Allocator.Error!void {
        try self.push(.{ .set_viewport = viewport });
    }

    pub fn setScissor(self: *CommandList, rect: ?types.Rect) Allocator.Error!void {
        try self.push(.{ .set_scissor = rect });
    }

    pub fn setVertexBuffer(self: *CommandList, slot: u32, buffer: types.Buffer, offset: u32) Allocator.Error!void {
        try self.push(.{ .set_vertex_buffer = .{ .slot = slot, .buffer = buffer, .offset = offset } });
    }

    pub fn setIndexBuffer(self: *CommandList, buffer: types.Buffer, format: types.IndexFormat) Allocator.Error!void {
        try self.push(.{ .set_index_buffer = .{ .buffer = buffer, .format = format } });
    }

    pub fn setUniformBuffer(self: *CommandList, slot: u32, buffer: types.Buffer) Allocator.Error!void {
        try self.push(.{ .set_uniform_buffer = .{ .slot = slot, .buffer = buffer } });
    }

    pub fn setTexture(self: *CommandList, slot: u32, texture: types.Texture, sampler: types.Sampler) Allocator.Error!void {
        try self.push(.{ .set_texture = .{ .slot = slot, .texture = texture, .sampler = sampler } });
    }

    pub fn draw(self: *CommandList, args: Command.Draw) Allocator.Error!void {
        try self.push(.{ .draw = args });
    }

    pub fn drawIndexed(self: *CommandList, args: Command.DrawIndexed) Allocator.Error!void {
        try self.push(.{ .draw_indexed = args });
    }
};

test "a list is the commands in the order they were recorded" {
    var list: CommandList = .init(std.testing.allocator);
    defer list.deinit();

    try list.beginPass(.{ .color = .{ .target = .{ .surface = .none } } });
    try list.setPipeline(.none);
    try list.draw(.{ .vertex_count = 3 });
    try list.endPass();

    const recorded = list.commands();
    try std.testing.expectEqual(@as(usize, 4), recorded.len);
    try std.testing.expectEqual(std.meta.Tag(Command).begin_pass, std.meta.activeTag(recorded[0]));
    try std.testing.expectEqual(@as(u32, 3), recorded[2].draw.vertex_count);
    try std.testing.expectEqual(@as(u32, 1), recorded[2].draw.instance_count);

    list.reset();
    try std.testing.expectEqual(@as(usize, 0), list.commands().len);
}
