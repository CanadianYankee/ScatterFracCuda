# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Windows screen saver (`.scr`) and desktop app (Win32 + Direct3D 11) that renders random iterated-function-system ("scatter") fractals on the GPU with CUDA. It continually generates a random fractal, accumulates it progressively while displaying it, then discards it and starts a new one. There are no tests and no lint setup.

## Build & run

- Visual Studio solution `ScatterFracCuda.sln`, single project, configurations `Debug|x64` and `Release|x64` only.
- Requires VS 2022 (toolset v143) with the **CUDA 13.5** Visual Studio build customization (`CUDA 13.5.props/.targets` are imported by name in the `.vcxproj`; when upgrading CUDA, update both imports). CUDA code generation targets `sm_75` and `sm_86` (the dev machine has an RTX 3080 Ti, sm_86). CUDA 13 can't compile for Maxwell/Pascal (sm_50/sm_61), so don't add those back.
- Command-line build (from a VS Developer prompt):
  `msbuild ScatterFracCuda.sln /p:Configuration=Release /p:Platform=x64`
- Output goes to `x64\<Config>\ScatterFracCuda.exe`. A post-build step copies it to `ScatterFracCuda.scr`. HLSL shaders are compiled by FxCompile to `TextureVS.cso` / `TexturePS.cso` in the same folder. `DXUtils::LoadShader` loads them from the exe's directory, so they must stay next to the exe or `.scr`, including when the saver is installed somewhere else.
- `.cu` files are compiled by nvcc. Other files are compiled by MSVC. Headers shared by both (`AccumData.h`, `CudaArray.h`, `Transform.h`, `VectorMatrix.h`, `FloatColor.h`, `randgen.h`) use `__host__`/`__device__` qualifiers and must stay nvcc-compatible.
- Debug output (e.g. the hit fraction) goes through `OutputDebugString`, so you see it in the VS Output window or DebugView, not in a console.

## Running

The mode depends on the executable's extension and the standard screen-saver switches (`CSaverEngine::ParseForOptions`). The structure is ported from the sibling repo InteractDX11; keep the two in step where practical.
- `ScatterFracCuda.exe` with no arguments runs in a normal resizable window with the File menu. This is the easiest way to debug. It draws at the `CONFIG_DATA` size (3840×2160), letterboxed into the window.
- `.scr /s` is full-screen saver mode. Every monitor runs its own independent `CFractalSaver`, with its own D3D device on that monitor's adapter (`CWindowCluster::m_bMultiRender`). Each draws at its monitor's native resolution. The savers are ticked one after another in a single loop, so only the last *active* one presents with vsync and the others use `Present(0)`. Otherwise N monitors would run at 1/N of the refresh rate. Any key, click, or mouse movement quits.
- `.scr /p <hwnd>` is the preview nested in the Screen Saver Settings dialog. It draws at the preview's size and quits once the host window is gone.
- `.scr /c[:hwnd]`, or `.scr` with no arguments, shows a "no settings" message box.
- Don't test the `.scr` with `Start-Process` or ShellExecute: the registered open verb adds `/S`, so it always runs full screen. Launch it with `UseShellExecute = false` to pass `/p` or `/c`.

## Architecture

**Control flow** (`ScatterFracCuda.cpp` → `SaverEngine` → `WindowCluster` → `SaverBase` / `FractalSaver`):
- `CSaverEngine` parses the command line and creates windows through `CWindowCluster`, one per monitor via `EnumDisplayMonitors`.
- `CWindowCluster` owns the windows, the shared window procedure (through the static `s_pTheCluster`), the `PeekMessage` loop, and the input handling that quits the saver.
- `CSaverBase` owns one window's D3D11 device and swap chain and handles resizing, `Tick()`, and `PresentFrame()`.
- `CFractalSaver` is the actual saver. On each `Tick()` it:
  1. creates a new `CGenerator` when there is none or `RestartNeeded()` is true,
  2. otherwise calls `Iterate()` + `RenderScene()` while `IsIncomplete()`,
  3. resets the generator when generation finishes, so a fresh fractal starts on the next tick.
- `IDM_FILE_NEW` (File > New, windowed mode only) also forces a restart.

**Multiple GPUs / CUDA contexts:** each `CFractalSaver` finds its CUDA device with `cudaD3D11GetDevice`. All savers share one thread, so it calls `cudaSetDevice` before every piece of CUDA work. If a monitor's adapter can't run CUDA, `AcceptAdapter` makes `CSaverBase` fall back to the first CUDA-capable adapter. Generators on the same GPU share its primary context, so **never call `cudaDeviceReset()`** while savers run. Nothing resets the context between fractals, so every CUDA allocation and event must be freed explicitly (`CCudaEventPair` handles the timing events).

**Idle and power behavior** (keep intact, since a screen saver runs for hours):
- `CSaverBase::PresentFrame` treats `DXGI_STATUS_OCCLUDED` as "stop iterating" and re-tests every 250 ms with `DXGI_PRESENT_TEST`. Always present through `PresentFrame()`.
- `CWindowCluster` registers for `GUID_CONSOLE_DISPLAY_STATE` and WTS session notifications, and pauses every saver while the displays are off or the session is locked.
- When no saver did any work in a pass, `Run()` blocks in `MsgWaitForMultipleObjectsEx` (100 ms cap) instead of spinning.

**DPI:** `ScatterFracCuda.manifest` (merged via `AdditionalManifestFiles`) declares Per-Monitor V2 DPI awareness, so window and monitor coordinates are physical pixels. Don't remove it.

**`CGenerator`** (`Generator.cpp`) owns all GPU state and the host-side randomization:
- `RandomizeTransforms()` picks 2–10 random affine maps (rotation × scale, optional elongation and skew, plus an offset), random colors, and an optional rotational (N ≤ 12) and/or mirror symmetry (`SYMMETRY`). Weights are stored **cumulatively, normalized to 1**, because the kernel selects a transform by comparing one uniform random number against them in order.
- `Initialize()` allocates the arrays, then runs a short `bInit` pass of `cuda_iterate` to seed the per-thread RNGs and find a bounding box, which sets `RECT_SCALE`. If any point goes non-finite, the kernel sets `bAbort` and the generator asks for a restart.
- Each `Iterate()` runs one cycle of `cuda_iterate`. On the first cycle it also measures the fraction of new pixels hit (`bHitPercent`). If that fraction is below `CONFIG_DATA::iHitPercent`, the fractal is rejected as too sparse and the generator restarts. When `bRender` is set, the cycle then renders into the shared texture.

**GPU pipeline** (data flows through these buffers):
1. `Generate.cu` `iterate`: 128×128 threads, each with its own `ITERATOR` (position, color, xorshift `CRandgen`). Each step applies a random transform and then a uniformly sampled element of the symmetry group. The thread then uses atomics to add count and color into `m_AccumArray` (`ACCUM`). That array is padded by the kernel radius on each side and scaled up by the anti-alias factor. Global stats (`ACCUM_STATS`: max count, max color, bbox, hit counts, abort flag) live in **managed memory**, so the CPU reads them directly after the kernels synchronize.
2. `Render.cu` `rescale_filter`: log-scales color, applies HSV gamma (`fGammaValue`, `fGammaSatur`), and does density-estimation blur. The Gaussian kernel's width shrinks with hit count (`fKernelAlpha`). The result goes to `m_FilteredArray` through atomic adds, using 8-way symmetry of the kernel.
3. `Render.cu` `render_texture`: box-downsamples the anti-aliased buffer into a float4 texture.

**D3D/CUDA interop** (`DX11CudaTexture.*`): an `R32G32B32A32_FLOAT` D3D11 texture registered with CUDA. Each frame `MapToCudaArray` maps it, the render kernels write to it, and `UnmapFromCudaArray` unmaps it. The window then draws one fullscreen quad (`TextureVS.hlsl`, 4-vertex triangle strip with no vertex buffer, aspect corrected through a constant buffer) sampling it with `TexturePS.hlsl`.

**GPU memory wrappers** (`CudaArray.h`): `CCudaArray1D<T>`, `CCudaArray2D<T>` (pitched), and `CCudaTexture2D` (float4 view). These are plain handles passed **by value** into kernels. They do not free in their destructors, so the owner must call `Free()` explicitly (see `~CGenerator`).

**`CTransform`** (`Transform.h`) is a flat `float[10]` (weight, RGB, 2×2 matrix, offset) with host mutable accessors and device const accessors, so it can be `cudaMemcpy`'d directly.

## Conventions

- Hungarian-style naming (`m_`, `n`, `f`, `b`, `p`, `C`-prefixed classes, ALL_CAPS POD structs), Win32 types (`UINT`, `BOOL`, `HRESULT`), `ComPtr` for COM objects.
- Host functions return `HRESULT` or `cudaError_t`; CUDA errors are usually mapped to `E_FAIL` on the host side.
- All tunables are the defaults in the `CONFIG_DATA` constructor (`ConfigData.h`). There is no UI or file for configuration yet.
