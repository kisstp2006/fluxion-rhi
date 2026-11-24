// SPDX-License-Identifier: BSD-2-Clause

//! HLSL compiled ahead of the device: into the shader cache, where the
//! Direct3D backends find it when they make the shader - so a start that
//! has no cache compiles what it will draw on every core at once, before
//! anything is drawn, rather than one stage at a time on the frame that
//! first draws it.
//!
//! A `Precompiler` needs no device. `precompile` is for any thread: the
//! compiler (`d3dcompiler_47`) answers calls from several at once, it holds
//! nothing that changes, and the cache it is given has to as well - the
//! engine's folder does, as it writes a stage into a file of its own and
//! puts it in place whole. It compiles exactly as `createShader` does - the
//! entry `main`, fxc's default flags, `vs_5_0` and `ps_5_0` - and keeps
//! under the same key, so what it keeps is what the backends take.
//!
//! ```zig
//! if (rhi.HlslPrecompiler.init(cache)) |compiler| {
//!     // on each of a few threads:
//!     _ = try compiler.precompile(gpa, hlsl.vertex, .vertex, &log);
//!     _ = try compiler.precompile(gpa, hlsl.fragment, .fragment, &log);
//! }
//! ```

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const d3d = @import("fluxion_d3d");
const com = d3d.com;

const types = @import("../types.zig");
const hlsl_cache = @import("hlsl_cache.zig");

/// Which stage a source is.
pub const Stage = enum {
    vertex,
    fragment,

    /// What the backends compile it for.
    pub fn target(self: Stage) [:0]const u8 {
        return switch (self) {
            .vertex => "vs_5_0",
            .fragment => "ps_5_0",
        };
    }
};

/// What `precompile` did.
pub const Outcome = enum {
    /// The cache had it already.
    cached,
    /// Compiled now, and kept.
    compiled,
};

pub const Precompiler = struct {
    compiler: d3d.Compiler,
    cache: ?types.ShaderCache,
    /// As the device that reads the cache compiles: `DeviceDesc.hlsl_optimization`.
    optimization: types.HlslOptimization = .full,

    /// The compiler loaded, for `cache` - which may be none, and then what
    /// is compiled is compiled for nothing. Null where there is no HLSL
    /// compiler: not Windows, or a Windows without `d3dcompiler_47`.
    pub fn init(cache: ?types.ShaderCache) ?Precompiler {
        if (builtin.target.os.tag != .windows) return null;
        const compiler = d3d.Compiler.load() catch return null;
        return .{ .compiler = compiler, .cache = cache };
    }

    /// The stage `source` is, kept in the cache unless it is there already.
    /// What is wrong with it goes to `log`, and is `error.ShaderFailed`.
    pub fn precompile(self: Precompiler, gpa: Allocator, source: []const u8, stage: Stage, log: *std.Io.Writer) types.Error!Outcome {
        return self.precompileFor(gpa, source, stage.target(), log);
    }

    /// `precompile` for a target by its name - `vs_5_0`, `ps_5_0` - as a
    /// cache's kept sources say it: see `ShaderCache.keep_source`.
    pub fn precompileFor(self: Precompiler, gpa: Allocator, source: []const u8, target: [:0]const u8, log: *std.Io.Writer) types.Error!Outcome {
        const key = hlsl_cache.keyFor(source, target, self.optimization);
        if (hlsl_cache.load(gpa, self.cache, key)) |held| {
            gpa.free(held);
            return .cached;
        }
        var output = self.compiler.compile(source, .{ .target = target, .name = "fluxion-rhi.hlsl", .flags = hlsl_cache.flagsFor(d3d.compiler.Flags, self.optimization) });
        const blob = output.check() catch {
            log.print("{s} did not compile:\n{s}", .{ target, output.text() }) catch {};
            output.release();
            return error.ShaderFailed;
        };
        defer _ = com.release(blob);
        hlsl_cache.store(self.cache, key, blob.bytes());
        return .compiled;
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const test_vertex =
    \\float4 main(uint id : SV_VertexID) : SV_Position {
    \\    return float4(float(id), 0.0, 0.0, 1.0);
    \\}
;

const test_pixel =
    \\float4 main() : SV_Target {
    \\    return float4(1.0, 0.0, 0.0, 1.0);
    \\}
;

/// A cache that several threads may use: the engine's folder is one, and
/// `hlsl_cache.MemoryCache` is not.
const LockedCache = struct {
    mutex: std.Io.Mutex = .init,
    io: std.Io,
    kept: std.AutoHashMapUnmanaged(types.ShaderCache.Key, []u8) = .empty,

    fn cache(self: *LockedCache) types.ShaderCache {
        return .{ .context = self, .load = loadFrom, .store = storeIn };
    }

    fn deinit(self: *LockedCache) void {
        var values = self.kept.valueIterator();
        while (values.next()) |each| testing.allocator.free(each.*);
        self.kept.deinit(testing.allocator);
    }

    fn loadFrom(context: *anyopaque, gpa: Allocator, key: types.ShaderCache.Key) ?[]u8 {
        const self: *LockedCache = @ptrCast(@alignCast(context));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const found = self.kept.get(key) orelse return null;
        return gpa.dupe(u8, found) catch null;
    }

    fn storeIn(context: *anyopaque, key: types.ShaderCache.Key, bytes: []const u8) void {
        const self: *LockedCache = @ptrCast(@alignCast(context));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const copy = testing.allocator.dupe(u8, bytes) catch return;
        const slot = self.kept.getOrPut(testing.allocator, key) catch return testing.allocator.free(copy);
        if (slot.found_existing) testing.allocator.free(slot.value_ptr.*);
        slot.value_ptr.* = copy;
    }
};

test "a stage compiled ahead is in the cache the device reads, and is not compiled twice" {
    var cache: LockedCache = .{ .io = testing.io };
    defer cache.deinit();
    const compiler = Precompiler.init(cache.cache()) orelse return error.SkipZigTest;
    var log: std.Io.Writer.Allocating = .init(testing.allocator);
    defer log.deinit();
    try testing.expectEqual(Outcome.compiled, try compiler.precompile(testing.allocator, test_vertex, .vertex, &log.writer));
    try testing.expectEqual(Outcome.cached, try compiler.precompile(testing.allocator, test_vertex, .vertex, &log.writer));
    // Under the key the backends look it up by, as bytecode.
    const kept = hlsl_cache.load(testing.allocator, cache.cache(), hlsl_cache.keyOf(test_vertex, "vs_5_0")).?;
    defer testing.allocator.free(kept);
    try testing.expect(hlsl_cache.isDxbc(kept));
}

test "a source that does not compile is said, and keeps nothing" {
    var cache: LockedCache = .{ .io = testing.io };
    defer cache.deinit();
    const compiler = Precompiler.init(cache.cache()) orelse return error.SkipZigTest;
    var log: std.Io.Writer.Allocating = .init(testing.allocator);
    defer log.deinit();
    try testing.expectError(error.ShaderFailed, compiler.precompile(testing.allocator, "this is not hlsl", .fragment, &log.writer));
    try testing.expect(log.written().len > 0);
    try testing.expectEqual(@as(u32, 0), cache.kept.count());
}

test "several threads compile at once into one cache" {
    var cache: LockedCache = .{ .io = testing.io };
    defer cache.deinit();
    const compiler = Precompiler.init(cache.cache()) orelse return error.SkipZigTest;
    const Work = struct {
        fn run(c: Precompiler, stage: Stage, source: []const u8, failed: *std.atomic.Value(bool)) void {
            var log: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
            defer log.deinit();
            _ = c.precompile(std.heap.page_allocator, source, stage, &log.writer) catch failed.store(true, .release);
        }
    };
    var failed: std.atomic.Value(bool) = .init(false);
    var threads: [6]std.Thread = undefined;
    for (&threads, 0..) |*thread, at| {
        thread.* = try std.Thread.spawn(.{}, Work.run, .{ compiler, if (at % 2 == 0) Stage.vertex else Stage.fragment, if (at % 2 == 0) test_vertex else test_pixel, &failed });
    }
    for (threads) |thread| thread.join();
    try testing.expect(!failed.load(.acquire));
    try testing.expectEqual(@as(u32, 2), cache.kept.count());
}
