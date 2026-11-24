// SPDX-License-Identifier: BSD-2-Clause

//! HLSL compiled once: the bytecode Direct3D 11 and 12 make with
//! d3dcompiler, kept in the device's `ShaderCache` and found there the next
//! time. Both backends compile a stage the same way - entry `main`, fxc's
//! default flags, shader model 5.0 - so they read each other's.

const std = @import("std");
const testing = std.testing;
const types = @import("../types.zig");

/// Changed whenever the way the backends compile changes - an entry, a
/// flag, a target - so that nothing kept before is read as what they make
/// now.
const version = "1";

/// The key of `source` compiled for `target`, as both backends compile it.
pub fn keyOf(source: []const u8, target: []const u8) types.ShaderCache.Key {
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update("fluxion-rhi hlsl " ++ version ++ "\x00");
    hash.update(target);
    hash.update("\x00main\x00");
    hash.update(source);
    return hash.finalResult();
}

/// `keyOf`, for a compile `optimization` says how to make: a fully
/// optimised one keeps the key it always had, a quick one has its own.
pub fn keyFor(source: []const u8, target: []const u8, optimization: types.HlslOptimization) types.ShaderCache.Key {
    if (optimization == .full) return keyOf(source, target);
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update("fluxion-rhi hlsl " ++ version ++ " fast\x00");
    hash.update(target);
    hash.update("\x00main\x00");
    hash.update(source);
    return hash.finalResult();
}

/// The flags fxc is given for `optimization`.
pub fn flagsFor(comptime Flags: type, optimization: types.HlslOptimization) Flags {
    return switch (optimization) {
        .full => .{},
        .fast_compile => .{ .skip_optimization = true },
    };
}

/// The bytecode kept under `key`, owned by `gpa`, where there is some and it
/// reads as a DXBC container; anything else is passed over, to be compiled
/// again.
pub fn load(gpa: std.mem.Allocator, cache: ?types.ShaderCache, key: types.ShaderCache.Key) ?[]u8 {
    const held = cache orelse return null;
    const bytes = held.load(held.context, gpa, key) orelse return null;
    if (isDxbc(bytes)) return bytes;
    gpa.free(bytes);
    return null;
}

pub fn store(cache: ?types.ShaderCache, key: types.ShaderCache.Key, bytes: []const u8) void {
    const held = cache orelse return;
    held.store(held.context, key, bytes);
}

/// The source of the stage under `key`, and its target, kept where the
/// cache keeps sources: see `ShaderCache.keep_source`.
pub fn keepSource(cache: ?types.ShaderCache, key: types.ShaderCache.Key, target: []const u8, source: []const u8) void {
    const held = cache orelse return;
    const keep = held.keep_source orelse return;
    keep(held.context, key, target, source);
}

/// Whether `bytes` are a DXBC container as far as its header goes: `DXBC`,
/// a checksum, a one, and its own length - what the runtime reads before
/// it checks the rest.
pub fn isDxbc(bytes: []const u8) bool {
    if (bytes.len < 32) return false;
    if (!std.mem.eql(u8, bytes[0..4], "DXBC")) return false;
    return std.mem.readInt(u32, bytes[24..28], .little) == bytes.len;
}

/// A `ShaderCache` in memory, counting what it was asked: for the backends'
/// tests.
pub const MemoryCache = struct {
    kept: std.AutoHashMapUnmanaged(types.ShaderCache.Key, []u8) = .empty,
    hits: usize = 0,
    misses: usize = 0,
    stores: usize = 0,

    pub fn cache(self: *MemoryCache) types.ShaderCache {
        return .{ .context = self, .load = loadFrom, .store = storeIn };
    }

    fn loadFrom(context: *anyopaque, gpa: std.mem.Allocator, key: types.ShaderCache.Key) ?[]u8 {
        const self: *MemoryCache = @ptrCast(@alignCast(context));
        const found = self.kept.get(key) orelse {
            self.misses += 1;
            return null;
        };
        self.hits += 1;
        return gpa.dupe(u8, found) catch null;
    }

    fn storeIn(context: *anyopaque, key: types.ShaderCache.Key, bytes: []const u8) void {
        const self: *MemoryCache = @ptrCast(@alignCast(context));
        const copy = testing.allocator.dupe(u8, bytes) catch return;
        const slot = self.kept.getOrPut(testing.allocator, key) catch return testing.allocator.free(copy);
        if (slot.found_existing) testing.allocator.free(slot.value_ptr.*);
        slot.value_ptr.* = copy;
        self.stores += 1;
    }

    pub fn deinit(self: *MemoryCache) void {
        var held = self.kept.valueIterator();
        while (held.next()) |bytes| testing.allocator.free(bytes.*);
        self.kept.deinit(testing.allocator);
    }
};

test "the key changes with the source and the target, and nothing else" {
    const vertex = keyOf("float4 main() : SV_POSITION { return 0; }", "vs_5_0");
    try testing.expectEqualSlices(u8, &vertex, &keyOf("float4 main() : SV_POSITION { return 0; }", "vs_5_0"));
    const other_target = keyOf("float4 main() : SV_POSITION { return 0; }", "ps_5_0");
    try testing.expect(!std.mem.eql(u8, &vertex, &other_target));
    const other_source = keyOf("float4 main() : SV_POSITION { return 1; }", "vs_5_0");
    // A quick compile is kept apart from a full one; a full one keeps its key.
    try testing.expectEqualSlices(u8, &vertex, &keyFor("float4 main() : SV_POSITION { return 0; }", "vs_5_0", .full));
    try testing.expect(!std.mem.eql(u8, &vertex, &keyFor("float4 main() : SV_POSITION { return 0; }", "vs_5_0", .fast_compile)));
    try testing.expect(!std.mem.eql(u8, &vertex, &other_source));
}

test "only what reads as a DXBC container is taken from the cache" {
    var dxbc: [40]u8 = @splat(0);
    @memcpy(dxbc[0..4], "DXBC");
    std.mem.writeInt(u32, dxbc[24..28], dxbc.len, .little);
    try testing.expect(isDxbc(&dxbc));
    try testing.expect(!isDxbc(dxbc[0..36]));
    try testing.expect(!isDxbc("DXBC"));
    var other = dxbc;
    @memcpy(other[0..4], "SPV!");
    try testing.expect(!isDxbc(&other));
}
