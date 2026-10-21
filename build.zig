// SPDX-License-Identifier: BSD-2-Clause

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gl = b.dependency("fluxion_gl", .{ .target = target, .optimize = optimize });
    const d3d = b.dependency("fluxion_d3d", .{ .target = target, .optimize = optimize });
    const vulkan = b.dependency("fluxion_vulkan", .{ .target = target, .optimize = optimize });
    const math = b.dependency("fluxion_math", .{ .target = target, .optimize = optimize });
    const id = b.dependency("fluxion_id", .{ .target = target, .optimize = optimize });
    const webgl = b.dependency("fluxion_webgl", .{ .target = target, .optimize = optimize });

    // The importable module. Consumers do:
    //   const rhi = @import("fluxion_rhi");
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "fluxion_gl", .module = gl.module("fluxion_gl") },
        // Imported on every target, analysed only on Windows: the module
        // refuses to compile anywhere else, and `Device` never names it
        // there.
        .{ .name = "fluxion_d3d", .module = d3d.module("fluxion_d3d") },
        .{ .name = "fluxion_vulkan", .module = vulkan.module("fluxion_vulkan") },
        .{ .name = "fluxion_math", .module = math.module("fluxion_math") },
        .{ .name = "fluxion_id", .module = id.module("fluxion_id") },
        // A browser's WebGL on wasm, and a stub on every other target -
        // which is what the WebGL backend's tests run against.
        .{ .name = "fluxion_webgl", .module = webgl.module("fluxion_webgl") },
    };
    const mod = b.addModule("fluxion_rhi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    });

    // zig build test. On Linux with the C library: the system's Vulkan and
    // OpenGL are opened with its `dlopen`, and without it every test that
    // draws there would skip itself, as if the machine had no GPU.
    const tests = b.addTest(.{
        .name = "fluxion-rhi-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = imports,
            .link_libc = if (target.result.os.tag == .linux) true else null,
        }),
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
    // The web
    // -------------------------------------------------------------------

    // The library again, built for a browser and drawing through the WebGL
    // backend, with a page and fluxion-webgl's glue around it. Nothing in it
    // is lazy, and the suite builds it, because that is what compiles the
    // backend for the target it exists for, rather than only against the stub.
    // (The examples that draw in a window, which need other libraries, are in
    // `fluxion-examples`.)
    const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const wasm_webgl = b.dependency("fluxion_webgl", .{ .target = wasm_target, .optimize = optimize });
    const wasm_math = b.dependency("fluxion_math", .{ .target = wasm_target, .optimize = optimize });
    const wasm_id = b.dependency("fluxion_id", .{ .target = wasm_target, .optimize = optimize });
    const wasm_rhi = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = wasm_target,
        .optimize = optimize,
        // No fluxion-gl and no fluxion-d3d: neither backend is analysed on
        // wasm, so neither is asked for.
        .imports = &.{
            .{ .name = "fluxion_math", .module = wasm_math.module("fluxion_math") },
            .{ .name = "fluxion_id", .module = wasm_id.module("fluxion_id") },
            .{ .name = "fluxion_webgl", .module = wasm_webgl.module("fluxion_webgl") },
        },
    });
    const web = b.addExecutable(.{
        .name = "rhi-web",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/web.zig"),
            .target = wasm_target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fluxion_rhi", .module = wasm_rhi },
                .{ .name = "fluxion_webgl", .module = wasm_webgl.module("fluxion_webgl") },
            },
        }),
    });
    // What makes an executable a module a page can instantiate: no `main`,
    // the exports kept rather than dropped, and memory of its own.
    web.entry = .disabled;
    web.rdynamic = true;
    web.import_memory = false;
    test_step.dependOn(&web.step);

    const web_step = b.step("example-web", "The WebGL backend in a browser: built into zig-out/web, to be served from there");
    web_step.dependOn(&b.addInstallArtifact(web, .{ .dest_dir = .{ .override = .{ .custom = "web" } } }).step);
    web_step.dependOn(&b.addInstallFile(b.path("examples/web/index.html"), "web/index.html").step);
    // The glue is fluxion-webgl's, and comes from there rather than being
    // copied into this repository to drift - asked for by the name that
    // package gives it, not by where it keeps the file.
    web_step.dependOn(&b.addInstallFile(webgl.namedLazyPath("glue"), "web/fluxion-webgl.js").step);

    // And its tests, on the host against the stub, with the library's.
    const web_host = b.createModule(.{
        .root_source_file = b.path("examples/web.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fluxion_rhi", .module = mod },
            .{ .name = "fluxion_webgl", .module = webgl.module("fluxion_webgl") },
        },
    });
    const web_tests = b.addTest(.{ .name = "fluxion-rhi-web-tests", .root_module = web_host });
    test_step.dependOn(&b.addRunArtifact(web_tests).step);
}
