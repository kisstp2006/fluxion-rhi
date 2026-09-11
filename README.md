# Fluxion RHI

One way to draw, on whichever API the machine has. For Zig 0.16.

| Module | What it is |
| --- | --- |
| `Device` | One backend, the resources made on it, and the frame being recorded. Handles in, validated command lists out. |
| `types` | Everything a program says to a device - formats, descriptions, states, passes - as plain values that know no backend. |
| `commands` | A frame written down before it is drawn: a `CommandList` the backend walks once at `submit`. |
| `backend` | What a backend has to answer to: a vtable of twenty calls. The seam a new one is written against. |
| `backend/gl` | OpenGL 3.3 core, on a context somebody else made, through [Fluxion GL](https://github.com/kisstp2006/fluxion-gl). |
| `backend/d3d11` | Direct3D 11 at feature level 11.0, through [Fluxion D3D](https://github.com/kisstp2006/fluxion-d3d). Windows only. |
| `backend/webgl` | WebGL 2 in a browser, through [Fluxion WebGL](https://github.com/kisstp2006/fluxion-webgl). A `wasm32` build only. |
| `backend/none` | Accepts everything, draws nothing. For build servers, headless programs and tests. |

```zig
const rhi = @import("fluxion_rhi");

var device = try rhi.Device.init(gpa, .{ .gl = window.hooks() });   // or .{} for Direct3D
defer device.deinit();

const surface = try device.createSurface(.{ .native_window = hwnd });
const shader = try device.createShader(.{
    .glsl = .{ .vertex = glsl_vs, .fragment = glsl_fs },
    .glsl_es = .{ .vertex = essl_vs, .fragment = essl_fs },
    .hlsl = .{ .vertex = hlsl_vs, .fragment = hlsl_ps },
});
const pipeline = try device.createPipeline(.{
    .shader = shader,
    .attributes = &.{
        .{ .location = 0, .format = .float2, .offset = 0, .buffer = 0 },
        .{ .location = 1, .format = .float4, .offset = 0, .buffer = 1 },
    },
    .buffers = &.{ .{ .stride = 8 }, .{ .stride = 16, .step = .instance } },
    .topology = .triangle_strip,
    .blend = .alpha,
    .uniform_blocks = &.{"Frame"},
    .textures = &.{"atlas"},
});

const cmd = device.begin();
try cmd.beginPass(.{ .color = .{ .target = .{ .surface = surface }, .clear_color = .{ 0, 0, 0, 1 } } });
try cmd.setPipeline(pipeline);
try cmd.setVertexBuffer(0, quad, 0);
try cmd.setVertexBuffer(1, instances, 0);
try cmd.setUniformBuffer(0, frame);
try cmd.setTexture(0, atlas, sampler);
try cmd.draw(.{ .vertex_count = 4, .instance_count = sprite_count });
try cmd.endPass();
try device.submit();
try device.present(surface);
```

Five decisions run through it:

**2D first, but not 2D only.** What is here today is what sprites, text, UI
and tilemaps need: instanced quads, textures and samplers, alpha blending,
scissor rectangles, an orthographic matrix. The shape is not 2D, though. Depth
states, depth attachments, cull modes and clip-space conventions are all in
the API now, so a 3D renderer is more of the same calls rather than a
different library - and `Device.clip` already answers which clip space a
projection is for.

**Handles, not pointers.** Every `create` returns eight bytes with a
generation in them, from [Fluxion Id](https://github.com/kisstp2006/fluxion-id).
A destroyed handle is `error.InvalidHandle`, from `Device` and before any
backend sees it.

**Validation once, in `Device`.** A draw outside a pass, a vertex buffer bound
where an index buffer was wanted, a texture that cannot be rendered to:
`submit` refuses these with `error.InvalidArgument` and `diagnostics` names
the command. A backend can assume a list that makes sense, and a program gets
the same answer on every backend - including `none`, on a machine with no GPU.

**The origin is the top left.** Viewports, scissor rectangles and the pixels
`readTexture` hands back all count from the top left, the way Direct3D,
Vulkan, Metal and every image file do. OpenGL counts from the bottom left,
and its backend does the arithmetic. The tests draw the same triangle on
both and read the same pixels back.

**Nothing here opens a window.** A device is made from what somebody else
made: a `getProcAddress` and a `swapBuffers` for OpenGL, an `HWND` for
Direct3D, the page's canvas for WebGL. [Fluxion Platform](https://github.com/kisstp2006/fluxion-platform)
provides the first two, and so do GLFW and SDL; the page and
[Fluxion WebGL](https://github.com/kisstp2006/fluxion-webgl)'s glue provide
the third. The library depends on none of them for that.

Nothing here allocates except through the allocator handed to `Device.init`.

## Install

```bash
zig fetch --save git+https://github.com/kisstp2006/fluxion-rhi
```

Or, for a checkout next to your project, add to `build.zig.zon`:

```zig
.dependencies = .{
    .fluxion_rhi = .{ .path = "../fluxion-rhi" },
},
```

Either way, wire it up in `build.zig`:

```zig
const fluxion = b.dependency("fluxion_rhi", .{
    .target = target,
    .optimize = optimize,
});
exe_mod.addImport("fluxion_rhi", fluxion.module("fluxion_rhi"));
```

Five dependencies come with it, fetched the same way and needing nothing from
you: [Fluxion GL](https://github.com/kisstp2006/fluxion-gl),
[Fluxion D3D](https://github.com/kisstp2006/fluxion-d3d) and
[Fluxion WebGL](https://github.com/kisstp2006/fluxion-webgl) are the three
backends' entry points, [Fluxion Math](https://github.com/kisstp2006/fluxion-math)
is where `Clip` comes from, and [Fluxion Id](https://github.com/kisstp2006/fluxion-id)
is the handles. Each backend's library is imported on every target and
analysed only where that backend can run - Fluxion D3D on Windows, Fluxion GL
off the web, Fluxion WebGL on it - so a Linux build never sees Direct3D and a
browser build never sees a driver to load. Fluxion WebGL is the exception
that proves it: off wasm it is a stub, and the WebGL backend's tests run
against that stub on any machine.

Three more are named in `build.zig.zon` and are *not* fetched for you:
[Fluxion Platform](https://github.com/kisstp2006/fluxion-platform) opens the
window the examples draw into,
[Fluxion Image](https://github.com/kisstp2006/fluxion-image) saves a frame as
a PNG, and [Fluxion Shader](https://github.com/kisstp2006/fluxion-shader) is
what the sprite example writes its shader in. All three are `lazy`, asked for
only when this is the package being built. Pass `-Dexamples=false` to skip
them in a checkout of this repository too.

## The shader contract

There is no cross-compiler here. A shader is source in the backend's own
language, and a program that ships on several backends ships each of them.
That is a seam rather than a burden: a shader compiler produces all of them
without `ShaderDesc` changing shape, and
[Fluxion Shader](https://github.com/kisstp2006/fluxion-shader) is the one
these examples use - one source in, GLSL, GLSL ES and HLSL out, plus the
locations and slots to describe the pipeline with. What has to agree between
them, whether a compiler wrote them or a person did:

| | GLSL 330 and GLSL ES 300 | HLSL 5.0 |
| --- | --- | --- |
| Vertex attribute at `location = n` | `layout(location = n) in` | semantic `ATTRn` |
| Uniform buffer in slot `n` | block named in `uniform_blocks[n]`, `std140` | `register(bn)` |
| Texture in slot `n` | sampler named in `textures[n]` | `register(tn)` and `register(sn)` |
| Fragment colour | `out vec4` | `SV_TARGET` |

`uniform_blocks` and `textures` exist because neither GLSL has
`layout(binding = n)`: the OpenGL and WebGL backends bind them by name once,
when the pipeline is made, and after that `setUniformBuffer(slot, ...)` means
the same thing on every API. GLSL ES is `glsl_es` rather than `glsl` because
the two are not interchangeable, even where the text below the first line is
the same: WebGL wants `#version 300 es` and a precision, and compiles a shader
without the version as GLSL ES 1.00.

## Three backends, and a fourth

| Backend | Where | Needs | How it presents |
| --- | --- | --- | --- |
| `gl` | Anywhere with OpenGL 3.3 core | `DeviceDesc.gl`: a context, current on this thread | `GlHooks.swap_buffers`; one surface, the context's own framebuffer |
| `d3d11` | Windows | Nothing; `d3d11.dll` is found at run time | A flip-model swap chain on the `HWND` in `SurfaceDesc` |
| `webgl` | A browser, from a `wasm32` build | Nothing; the page made the context, and Fluxion WebGL's glue hands it over | Returning from the frame callback; one surface, the canvas |
| `none` | Everywhere | Nothing | Nothing |

`Device.init(.{})` with `.backend = .auto` takes OpenGL when hooks were given,
Direct3D on Windows otherwise and WebGL on the web; it never chooses `none`
on its own. `Device.available()` lists what this build could open.

**What the OpenGL backend cannot do**, and says so: instancing and a base
vertex in the same indexed draw (`glDrawElementsInstancedBaseVertex` is 4.2),
and a second surface. **What the Direct3D backend does differently, and
hides**: a dynamic buffer keeps a CPU copy, because a constant buffer cannot
be updated in part and a discarding map hands back nothing - so
`updateBuffer(offset, ...)` means the same thing on both.

**What the WebGL backend does differently, and hides.** WebGL 2 has no base
vertex at all, so an indexed draw with one moves the per-vertex attribute
pointers that many vertices on instead - they are written at the draw anyway,
and unlike on OpenGL 3.3 it combines with instancing. It takes no BGRA upload,
so a `bgra8_unorm` texture is stored as RGBA with red and blue swapped on the
way in. It has no debug callback, so `DeviceDesc.debug` drains the error queue
after every submit instead. And it wants `ShaderDesc.glsl_es`: GLSL ES 3.00,
which is what Fluxion Shader writes beside its GLSL.

Off wasm, `Device` opens the WebGL backend only under test, where it talks to
Fluxion WebGL's stub. The stub draws nothing, and a program that asked for
`.webgl` on the desktop would get a device that answers every call and shows
nothing - so outside a test it gets `error.Unsupported` instead.

**Adding a backend** is one file under `src/backend/` that fills
`backend.Vtable`, and one arm in `Device.init`. The list a backend receives
has already been validated; what it has to get right is its own API, and the
top-left origin. Vulkan and Direct3D 12 would record the same list into a
real command buffer, which is why the list exists.

## Examples

| Example | What it shows |
| --- | --- |
| `zig build example` | Which backends this build has; a frame validated against `none`; a triangle through Direct3D's software rasteriser with no window anywhere, printed as text. |
| `zig build example-sprites` | 2D: two dozen textured sprites bouncing in a window, in one instanced draw, with alpha blending and a top-left orthographic matrix, from one shader source compiled into both languages and an atlas read off the disk. `-- --backend gl` or `d3d11`; `-- --capture out.png` draws one frame to a file instead. |
| `zig build example-web` | The WebGL backend in a browser, built into `zig-out/web` with a page and Fluxion WebGL's glue. It checks its own work before drawing anything - a pattern drawn into a texture and read back - and puts the verdict on the page. Serve the directory and open it; see below. |

The web example's check is the test a stub cannot be. Before the first frame
it draws four cells into a 64-pixel texture and reads it back: two from one
instanced draw out of an atlas uploaded as BGRA, one from an indexed draw
whose square starts four vertices into its buffer, and one through a scissor
rectangle with black left beside it. Every cell is chosen by an integer
attribute, the target size comes from a uniform block, and the atlas is read
through a sampler - so a swap of red and blue gone wrong, a base vertex not
applied, and a rectangle measured from the bottom each show as one wrong
pixel, named in the console. Then it draws the same pattern on the canvas,
four times the size, and sprites crossing the rest of it, every frame. The
suite builds it for `wasm32-freestanding`, which is what compiles the backend
for the target it exists for.

The sprite example loads `examples/atlas.png` at run time, which is why it
wants to be run from the root of this repository; `-- --atlas PATH` says
otherwise. The file in this repository was written by the example itself, and
`-- --write-atlas examples/atlas.png` writes it again - so the one binary
here is not one nobody can account for.

`examples/window.zig` is the window and the device on it, through Fluxion
Platform: the `GlHooks` for one backend and the `HWND` for the other, and the
`TestDevice` every test that needs a real GPU opens - OpenGL on a hidden
window, Direct3D on WARP.

The tests draw. The library's own suite draws a triangle through Direct3D on
WARP and reads it back; the window's suite draws the same triangle through
OpenGL, checks that the point is at the top and that a scissor rectangle
counts from the top left; the sprite example renders its frame on both
backends and checks that the two pictures agree to within a rasteriser's
rounding. On a machine without a display or a GPU, each of those skips rather
than fails, and the `none` backend carries the validation tests everywhere.

## Build

```bash
zig build test                  # run the test suite
zig build example               # the tour, no window
zig build example-sprites       # sprites in a window, default backend
zig build example-sprites -- --backend gl
zig build examples              # every example in turn, frames to zig-out/
zig build example-web           # the WebGL backend, into zig-out/web
zig build docs                  # generate API docs into zig-out/docs
```

A page cannot load a module from `file://`, so the web example wants a
server - any that sends `.js` as JavaScript:

```bash
python -m http.server 8000 --directory zig-out/web
```

## Licence

`BSD-2-Clause`. See [LICENSE](LICENSE).
