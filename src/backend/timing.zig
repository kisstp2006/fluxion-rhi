// SPDX-License-Identifier: BSD-2-Clause

//! What every backend's GPU timings keep the same way: a few frames' worth
//! of marks in a ring, so that the one being recorded never waits on one the
//! GPU has not got to yet.
//!
//! A backend has `sets * max_timestamps` queries of its own, set `s`'s mark
//! `m` at `s * max_timestamps + m`. It writes the mark `query` names when a
//! `timestamp` command comes, and at `Device.timings` closes the set being
//! recorded, then reads the waiting ones oldest first until one is not done:
//! the newest done is the answer. A set still waiting when the ring comes
//! round to it again is forgotten - a GPU that far behind is not timed.

const std = @import("std");
const types = @import("../types.zig");
const Device = @import("../Device.zig");

/// Frames in the ring.
pub const sets = 4;

pub const Ring = struct {
    /// The set being recorded into.
    at: u32 = 0,
    /// For each set, a bit for each mark written.
    kept: [sets]u32 = @splat(0),
    /// For each set closed and not read yet, when it was closed; nought for
    /// none.
    waiting: [sets]u64 = @splat(0),
    closings: u64 = 0,

    /// The query of mark `mark` in the set being recorded, and whether it is
    /// the set's first: where a backend starts what a set needs - Vulkan's
    /// reset, Direct3D 11's disjoint query.
    pub fn write(self: *Ring, mark: u32) struct { query: u32, first: bool } {
        const first = self.kept[self.at] == 0;
        self.kept[self.at] |= @as(u32, 1) << @intCast(mark);
        return .{ .query = self.at * types.max_timestamps + mark, .first = first };
    }

    /// Close the set being recorded and start the next, empty. The set
    /// closed, if it kept a mark - the backend then ends what it started.
    pub fn close(self: *Ring) ?u32 {
        const closed = self.at;
        const kept = self.kept[closed] != 0;
        if (kept) {
            self.closings += 1;
            self.waiting[closed] = self.closings;
        }
        self.at = (self.at + 1) % sets;
        self.waiting[self.at] = 0;
        self.kept[self.at] = 0;
        return if (kept) closed else null;
    }

    /// The sets closed and not read, oldest first.
    pub fn oldestFirst(self: *const Ring, into: *[sets]u32) []u32 {
        var n: usize = 0;
        for (0..sets) |s| if (self.waiting[s] != 0) {
            into[n] = @intCast(s);
            n += 1;
        };
        const Order = struct {
            fn less(ring: *const Ring, a: u32, b: u32) bool {
                return ring.waiting[a] < ring.waiting[b];
            }
        };
        std.mem.sort(u32, into[0..n], self, Order.less);
        return into[0..n];
    }

    /// Set `set` read: no longer waiting.
    pub fn read(self: *Ring, set: u32) void {
        self.waiting[set] = 0;
    }

    /// The first query of set `set`.
    pub fn base(set: u32) u32 {
        return set * types.max_timestamps;
    }
};

const testing = std.testing;

/// What each backend's own test of its timings does: two marks around a
/// pass, given back in order and nothing else, once the GPU is done - a
/// read back waits for that.
pub fn expectMarks(device: *Device) !void {
    if (!device.caps().features.timestamps) return error.SkipZigTest;
    const target = try device.createTexture(.{ .width = 64, .height = 64, .usage = .{ .render_target = true } });
    defer device.destroyTexture(target);
    const cmd = device.begin();
    try cmd.timestamp(2);
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 1, 0, 0, 1 } } });
    try cmd.endPass();
    try cmd.timestamp(5);
    try device.submit();
    var got = device.timings();
    var tries: usize = 0;
    while (got == null and tries < 20) : (tries += 1) {
        testing.allocator.free(try device.readTexture(target, testing.allocator));
        got = device.timings();
    }
    const marks = (got orelse return error.TestUnexpectedResult).marks;
    try testing.expectEqual(@as(?u64, 0), marks[2]);
    try testing.expect(marks[5] != null);
    for (marks, 0..) |mark, i| if (i != 2 and i != 5) try testing.expectEqual(@as(?u64, null), mark);
    // Read once: nothing more to give.
    try testing.expectEqual(@as(?types.Timings, null), device.timings());
}

test "marks go into the set being recorded, a closed one waits to be read, and one the ring comes round to is forgotten" {
    var ring: Ring = .{};
    const a = ring.write(3);
    try testing.expect(a.first);
    try testing.expectEqual(@as(u32, 3), a.query);
    try testing.expect(!ring.write(5).first);
    try testing.expectEqual(@as(?u32, 0), ring.close());
    // Nothing written: nothing to wait for.
    try testing.expectEqual(@as(?u32, null), ring.close());
    try testing.expectEqual(@as(u32, 2 * types.max_timestamps + 1), ring.write(1).query);
    try testing.expectEqual(@as(?u32, 2), ring.close());

    var into: [sets]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 0, 2 }, ring.oldestFirst(&into));
    ring.read(0);
    try testing.expectEqualSlices(u32, &.{2}, ring.oldestFirst(&into));

    // Set 2 is still waiting when the ring is back at it: forgotten.
    _ = ring.write(0);
    _ = ring.close();
    _ = ring.close();
    _ = ring.close();
    try testing.expectEqual(@as(u32, 2), ring.at);
    try testing.expectEqualSlices(u32, &.{3}, ring.oldestFirst(&into));
}

test "ticks become nanoseconds from the earliest mark kept" {
    var ticks: [types.max_timestamps]u64 = @splat(0);
    ticks[0] = 1000;
    ticks[1] = 1500;
    ticks[4] = 900;
    const t: types.Timings = .ofTicks(&ticks, 0b10011, 2);
    try testing.expectEqual(@as(?u64, 200), t.marks[0]);
    try testing.expectEqual(@as(?u64, 1200), t.marks[1]);
    try testing.expectEqual(@as(?u64, 0), t.marks[4]);
    try testing.expectEqual(@as(?u64, null), t.marks[2]);
    try testing.expectEqual(@as(?u64, 1000), t.between(0, 1));
    try testing.expectEqual(@as(?u64, null), t.between(0, 2));
}
