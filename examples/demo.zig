// SPDX-License-Identifier: BSD-2-Clause

//! The tour: which backends this build has, a frame recorded against the one
//! that draws nothing, and - where there is one - a triangle through a real
//! backend with no window anywhere, printed as text.
//!
//! Run it with `zig build example`. It opens nothing and needs no display;
//! the picture at the end comes from Direct3D's software rasteriser, so it is
//! the same on every Windows machine.

const std = @import("std");
const Io = std.Io;

const rhi = @import("fluxion_rhi");

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout.interface;
    const gpa = init.gpa;

    try out.writeAll("--- backends this build can open ---\n");
    for (rhi.available()) |backend| try out.print("   {t}\n", .{backend});

    // --- the backend that draws nothing ---------------------------------
    try out.writeAll("\n--- a frame against `none` ---\n");
    {
        var device = try rhi.Device.init(gpa, .{ .backend = .none });
        defer device.deinit();

        const shader = try device.createShader(.{});
        const pipeline = try device.createPipeline(.{
            .shader = shader,
            .attributes = &.{.{ .location = 0, .format = .float2, .offset = 0 }},
            .buffers = &.{.{ .stride = 8 }},
        });
        const quad = try device.createBuffer(.{ .kind = .vertex, .size = 32 });
        const target = try device.createTexture(.{ .width = 8, .height = 8, .usage = .{ .render_target = true } });

        const cmd = device.begin();
        try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target } } });
        try cmd.setPipeline(pipeline);
        try cmd.setVertexBuffer(0, quad, 0);
        try cmd.draw(.{ .vertex_count = 4 });
        try cmd.endPass();
        try device.submit();
        try out.print("   {f}: {d} commands accepted, nothing drawn\n", .{ device.info(), 5 });

        // And what validation catches, before any backend would.
        const bad = device.begin();
        try bad.draw(.{ .vertex_count = 3 });
        device.submit() catch |err| try out.print("   refused as expected ({t}): {s}\n", .{ err, device.diagnostics() });
    }

    // --- a real backend, with no window ---------------------------------
    try out.writeAll("\n--- a triangle through Direct3D 11, on WARP ---\n");
    var device = rhi.Device.init(gpa, .{ .backend = .d3d11, .software = true }) catch |err| {
        try out.print("   not on this machine: {t}\n", .{err});
        return out.flush();
    };
    defer device.deinit();
    try out.print("   {f}\n", .{device.info()});

    const width = 64;
    const height = 24;
    const target = try device.createTexture(.{ .width = width, .height = height, .usage = .{ .render_target = true } });
    const shader = device.createShader(.{ .hlsl = .{
        .vertex =
        \\struct In { float2 position : ATTR0; float3 colour : ATTR1; };
        \\struct Out { float4 position : SV_POSITION; float3 colour : COLOR0; };
        \\Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.colour = i.colour; return o; }
        ,
        .fragment =
        \\struct Out { float4 position : SV_POSITION; float3 colour : COLOR0; };
        \\float4 main(Out i) : SV_TARGET { return float4(i.colour, 1); }
        ,
    } }) catch |err| {
        try out.print("   shader: {s}\n", .{device.diagnostics()});
        return err;
    };
    const pipeline = try device.createPipeline(.{
        .shader = shader,
        .attributes = &.{
            .{ .location = 0, .format = .float2, .offset = 0 },
            .{ .location = 1, .format = .float3, .offset = 8 },
        },
        .buffers = &.{.{ .stride = 20 }},
    });
    const vertices = [_]f32{
        -0.8, -0.8, 1, 0, 0,
        0.0,  0.8,  0, 1, 0,
        0.8,  -0.8, 0, 0, 1,
    };
    const buffer = try device.createBuffer(.{ .kind = .vertex, .size = @sizeOf(@TypeOf(vertices)), .data = std.mem.asBytes(&vertices) });

    const cmd = device.begin();
    try cmd.beginPass(.{ .color = .{ .target = .{ .texture = target }, .clear_color = .{ 0, 0, 0, 1 } } });
    try cmd.setPipeline(pipeline);
    try cmd.setVertexBuffer(0, buffer, 0);
    try cmd.draw(.{ .vertex_count = 3 });
    try cmd.endPass();
    try device.submit();

    const pixels = try device.readTexture(target, gpa);
    defer gpa.free(pixels);

    // Red at the left, green at the top, blue at the right - as letters.
    try out.writeAll("\n");
    for (0..height) |y| {
        try out.writeAll("   ");
        for (0..width) |x| {
            const p = pixels[(y * width + x) * 4 ..][0..4];
            const glyph: u8 = if (p[0] + @as(u16, p[1]) + p[2] < 30) '.' else if (p[0] >= p[1] and p[0] >= p[2]) 'r' else if (p[1] >= p[2]) 'g' else 'b';
            try out.writeByte(glyph);
        }
        try out.writeAll("\n");
    }
    try out.writeAll("\nTop-left origin, whatever the API: the green corner is at the top.\n");
    try out.flush();
}
