# WinRunner — Local Inference Server

WinRunner runs GGUF language and vision models on your own PC and serves them to every device on your
network through an **OpenAI- and LM Studio-compatible API** at

```
http://<this-computer>:5070/v1
```

It drives the official [llama.cpp](https://github.com/ggml-org/llama.cpp) engine (`llama-server`), keeps the whole
model on your GPUs and fills their memory with context, passes each model's own GGUF settings and chat template
through unchanged, pairs vision projectors (mmproj) automatically, and ships a full control panel with live
monitoring, laid out for portrait monitors.

Target system this release is tuned for: **Windows 10 x64 · AMD Ryzen 5 3600 · 64 GB RAM · 2 × AMD Radeon RX 6800 (16 GB)**.
Other Windows or Linux PCs with AMD, NVIDIA or Intel GPUs (or no GPU) work too.

---

## Contents

1. [Requirements](#requirements)
2. [Installation (conda)](#installation-conda)
3. [First start](#first-start)
4. [Connecting clients](#connecting-clients)
5. [API compatibility](#api-compatibility)
6. [GPU allocation and context length](#gpu-allocation-and-context-length)
7. [GGUF settings and chat templates](#gguf-settings-and-chat-templates)
8. [Vision models](#vision-models)
9. [Tuning for Ryzen 5 3600 + 2 × RX 6800](#tuning-for-ryzen-5-3600--2--rx-6800)
10. [The control panel](#the-control-panel)
11. [Command line](#command-line)
12. [Files and folders](#files-and-folders)
13. [Troubleshooting](#troubleshooting)
14. [Development](#development)

---

## Requirements

| Item | Version |
|---|---|
| **Python** | **3.11** (3.10–3.12 work; the conda environment uses 3.11) |
| Conda | Miniconda, Miniforge or Anaconda |
| OS | Windows 10/11 x64 (Linux x64 also supported) |
| GPU driver | AMD Software: Adrenalin Edition (recent); it provides Vulkan and the ADL sensor library |
| Browser engine for the app window | Microsoft Edge WebView2 Runtime (`install.bat` offers to install it; without it the control panel opens in the browser) |
| Disk | ~300 MB for WinRunner + engine, plus your models |

Python packages (`requirements.txt`): `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic` 2, `psutil`, `pillow`,
`jinja2`, and `pywebview` (Windows, for the native window). No compiler, CUDA/ROCm SDK or Node.js is needed.

## Installation (conda)

### Automatic (Windows)

1. Install Miniconda if you do not have conda: <https://docs.conda.io/en/latest/miniconda.html>
2. Download or clone this repository, e.g. to `C:\WinRunner`.
3. Double-click **`install.bat`** (or run it from an Anaconda Prompt). It:
   - creates the conda environment **`winrunner` with Python 3.11** from conda-forge (Anaconda's default
     channels are not used, so no Terms-of-Service prompt / `CondaToSNonInteractiveError`),
   - installs `requirements.txt`,
   - downloads the latest official **llama.cpp Vulkan** build into `data\engines\`.

   Use `install.bat rocm` to download the ROCm/HIP build instead (see [backends](#vulkan-or-rocm)).
4. Optional, for access from other computers: right-click **`scripts\firewall.bat`** → *Run as administrator*
   (opens TCP 5070 on private networks).
5. Recommended for large models: right-click **`scripts\gpu-timeout.bat`** → *Run as administrator*, then restart
   Windows. It raises the Windows GPU timeout from 2 to 60 seconds (see [GPU device lost](#gpu-device-lost)).

### Manual

```bat
conda create -n winrunner --override-channels -c conda-forge python=3.11 pip -y
conda activate winrunner
cd C:\WinRunner
pip install -r requirements.txt
python -m winrunner --install-engine vulkan
```

or, equivalently, `conda env create -f environment.yml`.

## First start

| Launcher | What it does |
|---|---|
| **`WinRunner.bat`** | Starts WinRunner in its own application window (no console). |
| `WinRunner-Console.bat` | Same, with a console showing the application log. Add `--headless` to run only the API server, `--browser` to use your web browser. |

On first start WinRunner scans these folders for `.gguf` files:

- `<install folder>\models`
- `%USERPROFILE%\.lmstudio\models` (LM Studio's folder: existing LM Studio downloads are reused as-is)
- `%USERPROFILE%\.cache\lm-studio\models`

Add more under **Library › Folders**, or download models inside WinRunner (**Library › Download**, Hugging Face).

Select a model in the **Library**, check the memory plan, and press **Load model**. You can also skip this:
with just-in-time loading enabled (default), the first API request that names a model loads it.

## Connecting clients

Use the base URL shown in the header or on the **Server** page:

```
http://192.168.x.x:5070/v1        (from other computers)
http://127.0.0.1:5070/v1          (on this PC)
```

- **API key:** not required by default. Any string works as the key in clients that insist on one. You can require a
  key under **Settings › Network**.
- **Model name:** use the model id shown in the Library (e.g. `qwen3-32b-q4_k_m`). Aliases, file names, LM Studio
  style `publisher/repo` names and unique prefixes are also accepted. Requests never fail because of the model
  name:
  - A name that is not in the library (e.g. `gpt-4o`, or a client's fixed default) is answered by the currently
    loaded model.
  - If no model is loaded, the model used last is loaded first (just-in-time loading).
  - With just-in-time loading off, the loaded model also answers requests that name a different library model,
    instead of switching models.

  The activity log notes each substituted name once, and responses carry the id of the model that answered.

Works with Open WebUI, SillyTavern, Continue, Cline, AnythingLLM, Msty, Chatbox, the OpenAI Python/JS SDKs,
the LM Studio SDK and anything else that speaks the OpenAI API.

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.50:5070/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="qwen3-32b-q4_k_m",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(r.choices[0].message.content)
```

## API compatibility

| Endpoint | Notes |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | All library models when JIT loading is on (LM Studio behaviour), otherwise loaded models. |
| `POST /v1/chat/completions` | Streaming and non-streaming; tools / function calling; `response_format` (JSON schema); `logprobs`; images (`image_url`, base64 or http URL); `reasoning_content` for thinking models; `stream_options.include_usage`; `max_tokens: -1` (LM Studio) means "no limit". |
| `POST /v1/completions` | Raw text completion. |
| `POST /v1/responses` | OpenAI Responses API including `input_image` and streaming events. |
| `POST /v1/messages` | Anthropic Messages format (text, images, thinking). |
| `POST /v1/embeddings`, `POST /v1/rerank` | For embedding / reranker GGUFs. |
| `GET /api/v0/models`, `GET /api/v0/models/{id}` | LM Studio REST API: `type` (`llm`/`vlm`/`embeddings`), `state`, `arch`, `quantization`, `max_context_length`, `capabilities`. |
| `POST /api/v0/chat/completions`, `/api/v0/completions`, `/api/v0/embeddings` | LM Studio REST API with `stats` (`tokens_per_second`, `time_to_first_token`, `generation_time`, `stop_reason`), `model_info` and `runtime`. |

**Behaviour details.**

- *Just-in-time loading.* A request for a model that is not loaded loads it. Streaming clients receive SSE
  keep-alive comments with the load progress while they wait. The previous model is unloaded after its in-flight
  requests finish. Both can be switched off on the Server page.
- *Parallel requests.* Up to 4 requests run concurrently (engine slots) and share one unified KV cache, so a single
  request can still use the full context.
- *Live telemetry for every client.* Non-streaming chat requests are streamed internally so the control panel can
  show prompt progress and tokens live. The response returned to the client is the normal non-streaming JSON, the
  same shape the engine produces.
- *Automatic recovery.* If the engine crashes, or reports that it lost its GPU (`ErrorDeviceLost`), WinRunner takes
  it out of service at once and starts a new engine process with the same settings. It waits until the GPUs are usable
  again after a reset, and tries up to three times if the reload fails. Requests that arrive in the meantime wait
  for the new engine. A request that was interrupted before any output reached the client (non-streaming requests,
  and streaming requests before the first token) is run again automatically, once. Streaming clients that already
  received part of the answer get an error event and can retry. If a model fails again after 3 automatic restarts
  within 10 minutes, WinRunner unloads it instead. It is loaded again from the Library, or by the next request
  that names it when just-in-time loading is on.

## GPU allocation and context length

The default context is **65,536 tokens**. In automatic allocation WinRunner keeps the **whole model on the GPUs**
whenever it fits with at least a 4,096-token context, and sizes the context to the VRAM. Before every load:

1. **Context.** The requested length is clamped to the model's trained context unless you tick
   *Allow above trained context*, which uses RoPE scaling.
2. **KV cache precision.** *Auto* keeps an **F16** cache. It switches to **Q8_0** (near-lossless, half the size)
   only if F16 cannot reach the requested context in VRAM. Q4 is never chosen automatically.
3. **Devices.** Free VRAM of each GPU is read from the engine itself (`llama-server --list-devices`). It already
   excludes what other programs use; a safety margin stays free on top of that (256 MiB per GPU by default;
   adjustable, also per GPU).
4. **Context in VRAM** (load setting):
   - *Fill VRAM* (default): the largest context that fits. The KV cache takes the free VRAM, so the GPUs are used up
     to the safety margin (a 16 GB card ends up at about 15.8 GB). The context grows beyond the requested length, up
     to the trained context, and shrinks below it when the whole model would not fit otherwise.
   - *Up to requested*: the requested length, reduced only when the whole model would not fit otherwise.
   - *Exact*: always the requested length. If it does not fit, part of the model runs from system RAM (step 7).

   The memory plan and the activity log show when the context was raised or reduced.
5. **Layer split.** Layers are assigned to GPUs in contiguous ranges sized by each layer's real cost (weights + KV
   cache), so all GPUs fill up at the same rate. The output layer, the compute buffers (including the input copies
   llama.cpp keeps when it runs the GPUs as a pipeline), the vision projector and the GPU backend's run-time scratch
   memory (for example flash attention's F16 copy of a Q8_0 KV cache) are accounted for. The layout is passed to the
   engine explicitly (`-ngl all --tensor-split …`).
6. **Engine verification.** Builds that include `llama-fit-params` (all current releases) measure the layout with
   llama.cpp's own allocator before loading. WinRunner calibrates its estimate with the measurement, per GPU, until
   the measured layout is the one it chooses (usually 1 to 3 measurements), so the loaded model fills the GPUs
   without overcommitting them.
7. **If the model does not fit even at 4,096 tokens.**
   - *Mixture-of-experts models:* all attention and KV stay on the GPUs and only the expert weights of the first
     N layers move to system RAM (`--n-cpu-moe`). This is much faster than moving whole layers.
   - *Dense models:* the minimum number of layers runs on the CPU.

   With engine builds that have `--fit`, llama.cpp itself places what does not fit, within the same margins.

Why the context gives way first: when any part of a model is in system RAM, llama.cpp runs the attention of those
layers on the CPU, copies their weights to the first GPU for every batch of the prompt, and no longer runs the GPUs
as a pipeline. Prompt processing becomes many times slower, with the first GPU and the CPU busy while the other GPU
waits, even when only two layers are affected.

The **Library › Load** tab shows all of this before you load: per-GPU stacked bars (weights, KV cache, compute
buffers, vision projector, margin, free), a per-layer placement map, the largest context that still fits entirely
in VRAM for F16 and Q8_0, and the exact `llama-server` command line. *Verify with engine* runs the engine
measurement; loading always runs it.

*Manual* mode lets you set GPU layers, tensor split, main GPU, split mode and MoE CPU layers yourself. The context
is fitted to that layout in the same way (choose *Exact* to keep it as requested).

## GGUF settings and chat templates

WinRunner follows a **GGUF-first** rule: nothing that the GGUF defines is overridden unless you explicitly
configure it.

- **Chat template.** The model's embedded Jinja template (`tokenizer.chat_template`) is used with llama.cpp's Jinja
  engine (`--jinja`). After loading, WinRunner compares the template the engine reports with the one in the GGUF
  and shows *Template: GGUF ✓*. The Chat Template tab shows the template, its format family (ChatML, Llama 3,
  Gemma, Mistral, Harmony/gpt-oss, DeepSeek, GLM, Phi, …), its capabilities (tools, reasoning, system role) and a
  rendered prompt preview. Overrides are available when you need them: built-in templates, a custom template, or
  template arguments such as `{"enable_thinking": false}`.
- **Sampling defaults.** Recommended values stored in the GGUF (`general.sampling.*`) are applied by the engine
  itself. WinRunner does not pass sampling flags. Per-model *API presets* (Library › API preset) only fill in
  values the client did not send.
- **Special tokens, RoPE, context.** BOS/EOS/EOT tokens, RoPE base and scaling, and the trained context come from
  the GGUF. RoPE fields in the load form stay empty (meaning "from model") unless you set them.
- **Reasoning.** Thinking text goes to `reasoning_content` by default, which Open WebUI, SillyTavern and others
  display. Choose *Inline `<think>` tags* for clients that expect them in `content`.

## Vision models

Vision models in GGUF form are two files: the language model and a **multimodal projector** (`mmproj-*.gguf`) from
the same repository.

- WinRunner pairs projectors automatically: any mmproj GGUF in the model's folder, preferring **F16** precision for
  best image quality. It checks that the projector's output dimension matches the model. The pairing can be
  changed on the Vision tab.
- Models with a projector carry a **VISION** badge (eye icon) in the Library, the API model list
  (`type: "vlm"`, `capabilities: ["vision"]`), the chat console and the engine panel.
- Images are accepted as OpenAI `image_url` parts (base64 data URI or http/https URL), Responses API `input_image`,
  Anthropic `image` blocks and Ollama-style `images` arrays. Before they reach the engine they are normalised:
  - WebP, TIFF, HEIC/AVIF and other formats are converted. The engine decodes only JPEG/PNG/BMP/GIF; WebP fails
    without this.
  - Phone-photo EXIF rotation is applied.
  - CMYK images are converted to RGB.
  - Remote URLs are downloaded.
  - Optionally, large images are downscaled.
- Sending an image to a model without a projector returns a clear HTTP 400 error (`model_not_vision_capable`)
  instead of a silently ignored image.
- When downloading from Hugging Face, the projector is offered together with the model.

## Tuning for Ryzen 5 3600 + 2 × RX 6800

What WinRunner does by default on this machine, and why:

| Setting | Default | Reason |
|---|---|---|
| Backend | **Vulkan** | Needs only the Adrenalin driver; supports flash attention, quantized KV cache and multi-GPU on RDNA2 (gfx1030). |
| Multi-GPU | **Layer split** across both RX 6800 | Only small activations cross PCIe between GPUs, so a secondary slot running at x4/x8 costs little. Row split needs fast inter-GPU links and is not recommended. |
| Tensor split | Computed per model | Balances weights + KV per GPU. The display GPU usually has less free VRAM; the plan uses the measured free memory. |
| Flash attention | Auto | Removes the huge attention scratch buffer at long context (tens of GiB at 64K without it). Required for a quantized V cache. |
| Context in VRAM | Fill VRAM | The whole model stays on the GPUs and the KV cache takes the rest: about 15.8 of 16 GB in use per GPU. |
| VRAM safety margin | 256 MiB per GPU | The driver's free-memory figure already excludes other programs. Raise it for the display GPU if you run VRAM-hungry programs while a model is loaded. |
| KV cache | F16 → Q8_0 only if needed | Highest quality that still reaches the requested context with the whole model in 32 GB of VRAM. |
| Loading | Full read when fully offloaded, otherwise mmap | Faster, measurable loads straight into VRAM on Windows. mmap for partial offload avoids a second RAM copy. |
| Threads | Engine default (6 = physical cores) | With full offload the CPU only schedules work, so SMT threads do not help. |
| Process priority | Above normal | Keeps token generation smooth while you use the desktop. |
| GPU timeout (TDR) | Windows: 2 s, raise with `scripts\gpu-timeout.bat` | Windows resets a GPU whose job runs longer than 2 s. That kills the engine's GPU context ("ErrorDeviceLost"). *Settings › Hardware* warns while the limit is at the default. |
| Prompt cache | 8 GiB RAM | With 64 GB RAM, recent conversations resume instantly. |
| Large MoE models | Experts in RAM | 64 GB RAM plus 32 GB VRAM run models like gpt-oss-120b / GLM-4.5-Air with attention on the GPUs. |

Rough sizes for 2 × 16 GB, fully offloaded at 64K context: 7–14B models at Q8_0/Q6_K with an F16 KV cache;
24–32B models at Q4_K_M with a Q8_0 KV cache. Larger combinations, such as a 24B model at Q8_0, stay fully on the
GPUs with a somewhat smaller context. Use **Library › Load › Max full-offload ctx** to see the exact limit for each model.

### Vulkan or ROCm?

Both are official llama.cpp releases and can be installed side by side (**Settings › Engine**).

- **Vulkan:** works with the graphics driver alone. Recommended default.
- **ROCm / HIP** (`win-rocm` build): bundles the HIP runtime but loads rocBLAS from the **AMD HIP SDK for
  Windows**, which must be installed. It can be faster at prompt processing.

Compare both with your own models on the **Benchmark** tab (`llama-bench` with your exact load settings).

## The control panel

The panel is designed for a portrait monitor (for example 1080×1920). Tabs on the left, a live activity column on
the right, and a status bar at the bottom.

- **Library.** Models with architecture, parameters, quantization, size, trained context and capability badges
  (Vision, Tools, Reasoning, Embedding, MoE). Per model:
  - *Load:* configuration, memory plan and command line.
  - *Properties:* GGUF metadata and tensor types.
  - *Chat template:* viewer and prompt preview.
  - *Vision:* projector pairing.
  - *API preset:* sampling defaults and alias.
- **Server.** Network endpoints, API on/off, JIT, API key, loaded models with actual per-GPU allocations and slot
  usage, request history (client, tokens, cache hits, TTFT, prompt and generation speed), and ready-to-copy client
  snippets.
- **Chat.** Test console using the public API: streaming, markdown, reasoning blocks, tool calls, image attach /
  paste / drag-and-drop for vision models, per-message speed statistics, saved conversations.
- **Monitor.** Per-GPU graphs:
  - utilisation, and VRAM (total and llama-server's share);
  - edge, junction and memory temperatures;
  - board power and GFX/memory clocks;
  - fan speed.

  Plus CPU per-thread load, RAM, engine process stats and a per-request speed history. Data sources are the Windows
  GPU performance counters (the ones Task Manager uses) and the AMD driver's ADL sensors.
- **Benchmark.** `llama-bench` prompt processing / generation throughput with the model's load settings, with
  history.
- **Logs.** Engine and application logs with level filters, search, follow mode and download.
- **Settings.**
  - Appearance: themes (Classic Olive, Amber, Graphite, Steel, Phosphor, plus your own custom themes), CRT scanlines,
    glow, animation level, interface scale, activity column position.
  - Network, engine management (download / select / custom build), hardware report with optimisation notes and
    per-GPU VRAM margins, model load defaults, storage, startup behaviour.

**Activity column** (always visible):

- engine state and load progress, with the layer placement filling in as tensors load;
- the request pipeline (Receive → Prompt → Generate → Complete) with prompt-evaluation progress and cached tokens;
- live tokens/s gauge and history;
- per-GPU utilisation, VRAM, temperature, power and clock;
- the live **token stream** of the current request, with token boundaries and reasoning shown dimmed;
- the event log.

**Keyboard.** **F2** hides or shows the title bar and the page tabs (remembered across restarts). **F11** switches
the app window to full screen and back; in a web browser, F11 is the browser's own full screen.

All settings persist in `data\settings.json`. Every setting has a tooltip.

## Command line

```
python -m winrunner [--window | --browser | --headless] [--host 0.0.0.0] [--port 5070]
                    [--model MODEL_ID] [--data-dir DIR] [--install-engine {vulkan,rocm,cpu}]
```

## Files and folders

```
WinRunner\
  winrunner\            application (Python package + control panel in winrunner\static)
  data\                 created on first start
    settings.json       all settings (themes, defaults, per-model profiles)
    engines\            downloaded llama.cpp builds (one folder per build/backend)
    cache\              GGUF header index
    chats\              saved chat console conversations
    templates\          custom chat templates
    logs\winrunner.log  application log (rotated)
    benchmarks.json     benchmark history
  models\               default download folder (publisher\repository\file.gguf)
```

The whole folder is portable. Set `WINRUNNER_DATA` or `--data-dir` to keep data elsewhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CondaToSNonInteractiveError` during setup | Update to the current `install.bat` (it installs from conda-forge only). If you create the environment by hand, add `--override-channels -c conda-forge`. |
| `WebView2 initialization failed` / `Couldn't find a compatible Webview2 Runtime` | The app window needs the [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/) (Evergreen Bootstrapper), or re-run `install.bat`. Until it is installed, WinRunner opens the control panel in your browser; use the power button in the header to exit. |
| Other computers cannot connect | Run `scripts\firewall.bat` as administrator (or allow Python on *private* networks when Windows asks). Check that Settings › Network › Bind address is `0.0.0.0`. |
| "No llama.cpp engine installed" | Settings › Engine › *Check for llama.cpp releases* › Install (Vulkan). |
| Engine reports no GPUs | Update the Adrenalin driver; *Re-detect engine and devices*. For ROCm builds install the AMD HIP SDK. |
| Load fails with out-of-memory | Lower the context, set KV cache to Q8_0, or raise the safety margin if other applications use VRAM. The memory plan shows the largest context that fits. |
| Prompt processing is very slow; one GPU and the CPU are busy while the other GPU idles; VRAM is not full | Part of the model runs from system RAM. With *Context in VRAM: Fill VRAM* (default) that only happens when the model does not fit even with a 4,096-token context. Check the memory plan (**FULL GPU OFFLOAD**) and the activity log line *Placement: …*; if *Context in VRAM* is *Exact*, switch it back to *Fill VRAM* or lower the context. |
| Port 5070 already in use | Close the other program (e.g. a second WinRunner) or change the port in Settings › Network (restart required). |
| Model answers in a strange format | Keep *Template source: GGUF embedded*. Check the Chat Template tab; some old GGUFs have no template and need a built-in one. |
| Images rejected | The model needs its mmproj file in the same folder (Vision tab). |
| Control panel "not available on the network" | By default only this PC may open the panel; enable *Allow the control panel from other computers*. |
| Requests fail with `vk::Queue::submit: ErrorDeviceLost`, or the engine exits with code 3221226505 (`0xC0000409`) | The GPU was reset while the engine was using it. WinRunner restarts the engine automatically. To prevent it, see [GPU device lost](#gpu-device-lost) below. |

Logs: **Logs** tab, or `data\logs\winrunner.log`.

### GPU device lost

`decode() failed: vk::Queue::submit: ErrorDeviceLost` means the GPU was reset while llama.cpp was using it. The
engine's GPU context is lost for good: the old engine process keeps running, but every later request fails. It
often aborts on the next request (exit code 3221226505 = `0xC0000409`), even after sitting idle for hours.
WinRunner detects both cases and restarts the engine automatically (see *Automatic recovery* under
[API compatibility](#api-compatibility)). The activity log shows what happened.

Common causes of the reset, most likely first:

1. **Windows GPU timeout (TDR).** Windows resets a GPU when one GPU job runs longer than 2 seconds. Long prompts on
   large models can exceed this, especially when part of the model runs from system RAM (MoE experts on the CPU,
   partial offload, or VRAM spilling into shared GPU memory). Run `scripts\gpu-timeout.bat` as administrator and
   restart Windows. It sets `TdrDelay` / `TdrDdiDelay` to 60 s; `scripts\gpu-timeout.bat reset` restores the
   defaults. *Settings › Hardware* shows a note while the default applies.
2. **VRAM shortage.** On Windows, when VRAM runs out, memory spills into shared system memory instead of failing.
   Everything becomes very slow, which in turn triggers the timeout. Raise the safety margin (*Settings › Hardware*,
   especially for the GPU driving your displays), lower the context length or use a Q8_0 KV cache. Very low prompt
   speeds in the request history (tens of tokens/s where hundreds are normal) are a sign of spilling.
3. **Driver or hardware instability.** Update the AMD Adrenalin driver. Remove overclocks or undervolts; they are
   often stable in games but not under sustained compute. Check GPU temperatures on the **Monitor** tab.

## Development

```bat
pip install -r requirements-dev.txt
python -m pytest
```

The unit tests cover GGUF parsing, the memory planner, log parsing, command-line generation, image normalisation,
the stream proxy and settings. `tests\test_recovery.py` checks engine failure recovery end to end: it runs the full
server against a scripted fake `llama-server` (`tests\fake_engine.py`, Linux only) that loses its GPU, aborts or
crashes while idle. `tests\test_vram_fill.py` runs the engine verification end to end against a fake
`llama-fit-params` (`tests\fake_fit_params.py`) whose measurements differ from WinRunner's estimate by a known amount,
and checks that the calibrated plan fills the GPUs without overcommitting them.

`tests\test_integration.py` runs the full server against a real engine. Set these first:

- `WINRUNNER_TEST_ENGINE`: path to `llama-server`
- `WINRUNNER_TEST_MODELS`: a folder containing Qwen3-0.6B and SmolVLM-256M + mmproj GGUFs

`tests\preview_server.py` starts the app with two simulated RX 6800s, for UI work on machines without those GPUs.
It is for development only; the product never reports simulated hardware.

## License

WinRunner is released under the MIT License. llama.cpp / ggml are MIT licensed
(© The ggml authors); they are downloaded from their official GitHub releases at install time.
