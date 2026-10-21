// SPDX-License-Identifier: BSD-2-Clause

// Written by fxport (the Fluxion port tools); the same in every package that
// has it, but for the licence line above. Do not edit here: change
// fluxion-tools/src/fx_meta.zig and run `fxport fix reflect --write` again.

//! The container descriptions `@typeInfo` gives in Zig 0.17 hold their fields
//! as parallel arrays (`field_names`, `field_types`, `field_attrs`). Most of
//! the Fluxion code walks a struct's fields one by one, as it did in 0.16
//! with `info.fields`; these functions give that view back, over the new
//! arrays, so a loop keeps its shape:
//!
//! ```zig
//! inline for (fx_meta.structFields(@typeInfo(T).@"struct")) |field| { ... field.name, field.type ... }
//! ```
//!
//! Everything here is comptime: a result holds `type` values. The functions
//! are `inline`, so that a call is comptime-known where it stands - as the
//! old `info.fields` was - and `inline for (fx_meta.structFields(info))` needs
//! no `comptime` at the call. Each raises the evaluation quota, since a program
//! that walks every declaration of a big type (a binding with hundreds of
//! functions) would run out of the default 1000 branches.

const std = @import("std");
const Type = std.lang.Type;

pub const StructField = struct {
    name: [:0]const u8,
    type: type,
    default_value_ptr: ?*const anyopaque,
    is_comptime: bool,
    alignment: comptime_int,

    /// What `Type.StructField.defaultValue` was in 0.16: the default, if the
    /// field has one.
    pub inline fn defaultValue(comptime field: StructField) ?field.type {
        const pointer: *const field.type = @ptrCast(@alignCast(field.default_value_ptr orelse return null));
        return pointer.*;
    }
};

pub const EnumField = struct {
    name: [:0]const u8,
    value: comptime_int,
};

pub const UnionField = struct {
    name: [:0]const u8,
    type: type,
    alignment: comptime_int,
};

pub const Declaration = struct {
    name: [:0]const u8,
};

pub const Error = struct {
    name: [:0]const u8,
};

/// What `Type.Fn.Param` was in 0.16. A parameter with no type (`anytype`, or
/// one whose type depends on an earlier argument) has `type == null`.
pub const FnParam = struct {
    is_generic: bool,
    is_noalias: bool,
    type: ?type,
};

/// The alignment a field has when none is written: that of its type, or 0
/// for a type that only exists at compile time. An opaque type is 1, as its
/// pointer's alignment was in 0.16.
fn alignOf(comptime T: type) comptime_int {
    return switch (@typeInfo(T)) {
        .type, .comptime_int, .comptime_float, .undefined, .null, .noreturn => 0,
        .@"opaque" => 1,
        else => @alignOf(T),
    };
}

/// What `Type.Pointer.address_space` was in 0.16.
pub inline fn pointerAddressSpace(comptime pointer: Type.Pointer) std.lang.AddressSpace {
    return pointer.attrs.@"addrspace" orelse .generic;
}

/// What `Type.Fn.params` was in 0.16.
pub inline fn fnParams(comptime info: Type.Fn) []const FnParam {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var out: [info.param_types.len]FnParam = undefined;
        for (info.param_types, info.param_attrs, 0..) |T, attrs, i| {
            out[i] = .{ .is_generic = T == null, .is_noalias = attrs.@"noalias", .type = T };
        }
        const final = out;
        break :blk &final;
    };
}

pub inline fn structFields(comptime info: Type.Struct) []const StructField {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var out: [info.field_names.len]StructField = undefined;
        for (info.field_names, info.field_types, info.field_attrs, 0..) |name, T, attrs, i| {
            out[i] = .{
                .name = name,
                .type = T,
                .default_value_ptr = attrs.default_value_ptr,
                .is_comptime = attrs.@"comptime",
                .alignment = attrs.@"align" orelse alignOf(T),
            };
        }
        const final = out;
        break :blk &final;
    };
}

pub inline fn enumFields(comptime info: Type.Enum) []const EnumField {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var out: [info.field_names.len]EnumField = undefined;
        for (info.field_names, info.field_values, 0..) |name, value, i| out[i] = .{ .name = name, .value = value };
        const final = out;
        break :blk &final;
    };
}

pub inline fn unionFields(comptime info: Type.Union) []const UnionField {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var out: [info.field_names.len]UnionField = undefined;
        for (info.field_names, info.field_types, info.field_attrs, 0..) |name, T, attrs, i| {
            out[i] = .{ .name = name, .type = T, .alignment = attrs.@"align" orelse alignOf(T) };
        }
        const final = out;
        break :blk &final;
    };
}

/// The declarations of a struct, enum, union or opaque description.
pub inline fn declarations(comptime info: anytype) []const Declaration {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        var out: [info.decl_names.len]Declaration = undefined;
        for (info.decl_names, 0..) |name, i| out[i] = .{ .name = name };
        const final = out;
        break :blk &final;
    };
}

pub inline fn errors(comptime info: Type.ErrorSet) []const Error {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        const names = info.error_names orelse @compileError("an anyerror has no list of errors");
        var out: [names.len]Error = undefined;
        for (names, 0..) |name, i| out[i] = .{ .name = name };
        const final = out;
        break :blk &final;
    };
}

fn FieldsOf(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .@"struct" => []const StructField,
        .@"enum" => []const EnumField,
        .@"union" => []const UnionField,
        .error_set => []const Error,
        else => @compileError("fields: a struct, enum, union or error set is needed, not " ++ @typeName(T)),
    };
}

/// What `std.meta.fields` gave in 0.16.
pub inline fn fields(comptime T: type) FieldsOf(T) {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| structFields(info),
        .@"enum" => |info| enumFields(info),
        .@"union" => |info| unionFields(info),
        .error_set => |info| errors(info),
        else => unreachable,
    };
}

/// What `std.meta.fieldNames` gave in 0.16.
pub inline fn fieldNames(comptime T: type) []const [:0]const u8 {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| info.field_names,
        .@"enum" => |info| info.field_names,
        .@"union" => |info| info.field_names,
        .error_set => |info| info.error_names orelse @compileError("an anyerror has no list of errors"),
        else => @compileError("fieldNames: a struct, enum, union or error set is needed, not " ++ @typeName(T)),
    };
}

const testing = std.testing;

test "struct fields keep their order, types, defaults and flags" {
    const S = struct {
        a: u8 = 7,
        b: f32,
        comptime c: u8 = 3,
        d: u64 align(16) = 0,
    };
    const list = comptime structFields(@typeInfo(S).@"struct");
    try testing.expectEqual(@as(usize, 4), list.len);
    inline for (list, 0..) |field, i| {
        switch (i) {
            0 => {
                try testing.expectEqualStrings("a", field.name);
                try testing.expect(field.type == u8);
                try testing.expect(field.default_value_ptr != null);
                try testing.expect(!field.is_comptime);
                try testing.expectEqual(@as(comptime_int, 1), field.alignment);
            },
            1 => {
                try testing.expect(field.type == f32);
                try testing.expect(field.default_value_ptr == null);
            },
            2 => try testing.expect(field.is_comptime),
            3 => try testing.expectEqual(@as(comptime_int, 16), field.alignment),
            else => unreachable,
        }
    }
}

test "a default value can be read back through the pointer" {
    const S = struct { a: u8 = 7 };
    const field = comptime structFields(@typeInfo(S).@"struct")[0];
    const dv: *const field.type = @ptrCast(@alignCast(field.default_value_ptr.?));
    try testing.expectEqual(@as(u8, 7), dv.*);
}

test "defaultValue gives the default or null" {
    const S = struct { a: u8 = 7, b: f32 };
    const list = comptime structFields(@typeInfo(S).@"struct");
    try testing.expectEqual(@as(?u8, 7), list[0].defaultValue());
    try testing.expectEqual(@as(?f32, null), list[1].defaultValue());
}

test "enum fields carry names and values" {
    const E = enum(u8) { a = 1, b = 5 };
    const list = comptime enumFields(@typeInfo(E).@"enum");
    try testing.expectEqual(@as(usize, 2), list.len);
    try testing.expectEqualStrings("b", list[1].name);
    try testing.expectEqual(@as(comptime_int, 5), list[1].value);
}

test "union fields and error set errors" {
    const U = union(enum) { x: u8, y: f32 };
    const ufields = comptime unionFields(@typeInfo(U).@"union");
    try testing.expectEqual(@as(usize, 2), ufields.len);
    try testing.expect(ufields[1].type == f32);
    const Errs = error{ Boom, Bang };
    const list = comptime errors(@typeInfo(Errs).error_set);
    try testing.expectEqual(@as(usize, 2), list.len);
}

test "declarations list names" {
    const S = struct {
        pub const one = 1;
        pub fn two() void {}
    };
    const list = comptime declarations(@typeInfo(S).@"struct");
    try testing.expectEqual(@as(usize, 2), list.len);
}

test "fields and fieldNames work by type" {
    const S = struct { a: u8, b: u8 };
    try testing.expectEqual(@as(usize, 2), comptime fields(S).len);
    try testing.expectEqual(@as(usize, 2), comptime fieldNames(S).len);
    const E = enum { p, q, r };
    try testing.expectEqual(@as(usize, 3), comptime fields(E).len);
    try testing.expectEqualStrings("q", comptime fieldNames(E)[1]);
}

test "a field of a comptime-only type has no alignment" {
    const S = struct { t: type };
    const list = comptime structFields(@typeInfo(S).@"struct");
    try testing.expectEqual(@as(comptime_int, 0), list[0].alignment);
}

fn countFields(comptime T: type) usize {
    var n: usize = 0;
    // No `comptime` at the call: the helper is inline.
    inline for (structFields(@typeInfo(T).@"struct")) |field| {
        _ = field;
        n += 1;
    }
    return n;
}

test "an inline for over the helper needs no comptime at the call" {
    try testing.expectEqual(@as(usize, 3), countFields(struct { a: u8, b: u8, c: u8 }));
}

test "a pointer's address space is generic when none is written" {
    const plain = @typeInfo(*const u32).pointer;
    try testing.expectEqual(std.lang.AddressSpace.generic, pointerAddressSpace(plain));
    try testing.expectEqual(@as(?usize, null), plain.attrs.@"align");
    const wide = @typeInfo(*align(16) u8).pointer;
    try testing.expectEqual(@as(?usize, 16), wide.attrs.@"align");
}

fn sample(a: u8, noalias b: *u8, c: anytype) void {
    _ = .{ a, b, c };
}

test "a function's parameters keep their order, types and flags" {
    const info = @typeInfo(@TypeOf(sample)).@"fn";
    const params = fnParams(info);
    try testing.expectEqual(@as(usize, 3), params.len);
    try testing.expect(params[0].type.? == u8);
    try testing.expect(params[1].is_noalias);
    try testing.expect(params[2].is_generic and params[2].type == null);
}
