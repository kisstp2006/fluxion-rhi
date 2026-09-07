// SPDX-License-Identifier: BSD-2-Clause

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gl = b.dependency("fluxion_gl", .{ .target = target, .optimize = optimize });
    const d3d = b.dependency("fluxion_d3d", .{ .target = target, .optimize = optimize });
    const math = b.dependency("fluxion_math", .{ .target = target, .optimize = optimize });
    const id = b.dependency("fluxion_id", .{ .target = target, .optimize = optimize });

    // The importable module. Consumers do:
    //   const rhi = @import("fluxion_rhi");
    const mod = b.addModule("fluxion_rhi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fluxion_gl", .module = gl.module("fluxion_gl") },
            // Imported on every target, analysed only on Windows: the module
            // refuses to compile anywhere else, and `Device` never names it
            // there.
            .{ .name = "fluxion_d3d", .module = d3d.module("fluxion_d3d") },
            .{ .name = "fluxion_math", .module = math.module("fluxion_math") },
            .{ .name = "fluxion_id", .module = id.module("fluxion_id") },
        },
    });

    // zig build test
    const tests = b.addTest(.{
        .name = "fluxion-rhi-tests",
        .root_module = mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the library test suite");
    test_step.dependOn(&run_tests.step);

    // zig build docs -> zig-out/docs
    const docs_lib = b.addLibrary(.{
        .name = "fluxion-rhi",
        .root_module = mod,
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate API documentation into zig-out/docs");
    docs_step.dependOn(&install_docs.step);

    // -------------------------------------------------------------------
    // Examples
    // -------------------------------------------------------------------

    // The examples need a window to draw into and a way to save a frame, and
    // neither is this library's business: the window is `fluxion-platform`
    // and the PNG is `fluxion-image`. Both are lazy dependencies, fetched only
    // when the examples are actually wanted - which is when this is the
    // package being built, not when it is somebody else's dependency.
    const examples_wanted = b.option(
        bool,
        "examples",
        "Build the examples and their tests (pulls fluxion-platform and fluxion-image)",
    ) orelse (b.pkg_hash.len == 0);
    if (!examples_wanted) return;

    const platform_dep = b.lazyDependency("fluxion_platform", .{
        .target = target,
        .optimize = optimize,
    }) orelse return;
    const image_dep = b.lazyDependency("fluxion_image", .{
        .target = target,
        .optimize = optimize,
    }) orelse return;

    // A window with or without a GL context in it, as the backend needs, and
    // the hooks that hand the context to the device.
    const window_mod = b.createModule(.{
        .root_source_file = b.path("examples/window.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fluxion_rhi", .module = mod },
            .{ .name = "fluxion_platform", .module = platform_dep.module("fluxion_platform") },
        },
    });

    // A window needs a windowing system, which a cross-compiled build has no
    // way to reach. Everything still builds for any target; the runs and the
    // tests that open a window are for the host only, and they skip themselves
    // on a host with no display.
    const host = target.result.os.tag == @import("builtin").os.tag;

    const window_tests = b.addTest(.{ .name = "fluxion-rhi-window-tests", .root_module = window_mod });
    if (host) test_step.dependOn(&b.addRunArtifact(window_tests).step);

    // zig build example runs the tour; zig build example-<name> runs one of
    // the others; zig build examples runs all of them, in this order. The
    // one with a window in it writes a frame to a file for the aggregate run
    // rather than opening anything.
    const examples = [_]struct {
        name: []const u8,
        step: []const u8,
        about: []const u8,
        needs_window: bool = false,
        chained_args: []const []const u8 = &.{},
    }{
        .{ .name = "demo", .step = "example", .about = "Which backends this machine has, and a triangle through one of them" },
        .{
            .name = "sprites",
            .step = "example-sprites",
            .about = "2D: textured sprites in a window, on whichever backend is asked for",
            .needs_window = true,
            .chained_args = &.{ "--capture", "zig-out/sprites.png" },
        },
    };

    const all_examples = b.step("examples", "Build and run every example in turn");
    var previous: ?*std.Build.Step = null;

    for (examples) |example| {
        const example_mod = b.createModule(.{
            .root_source_file = b.path(b.fmt("examples/{s}.zig", .{example.name})),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fluxion_rhi", .module = mod },
                .{ .name = "fluxion_math", .module = math.module("fluxion_math") },
                .{ .name = "fluxion_image", .module = image_dep.module("fluxion_image") },
                .{ .name = "window", .module = window_mod },
            },
        });
        const exe = b.addExecutable(.{
            .name = b.fmt("fluxion-rhi-{s}", .{example.name}),
            .root_module = example_mod,
        });
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        run.step.dependOn(b.getInstallStep());
        // Anything after `--` goes through: `zig build example-sprites -- --backend gl`.
        if (b.args) |args| run.addArgs(args);
        b.step(example.step, example.about).dependOn(&run.step);

        if (example.needs_window and !host) continue;

        const in_order = b.addRunArtifact(exe);
        in_order.step.dependOn(b.getInstallStep());
        in_order.addArgs(example.chained_args);
        if (previous) |earlier| in_order.step.dependOn(earlier);
        previous = &in_order.step;
        all_examples.dependOn(&in_order.step);

        const example_tests = b.addTest(.{
            .name = b.fmt("fluxion-rhi-{s}-tests", .{example.name}),
            .root_module = example_mod,
        });
        test_step.dependOn(&b.addRunArtifact(example_tests).step);
    }
}
