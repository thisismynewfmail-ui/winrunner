# WinRunner — Local Inference Server

WinRunner runs GGUF language and vision models on your own PC and serves them to every device on your
network through an **OpenAI- and LM Studio-compatible API** at

```
http://<this-computer>:5070/v1
```

It drives the official [llama.cpp](https://github.com/ggml-org/llama.cpp) engine (`llama-server`), fills your GPUs
with the model for the context length you ask for, passes each model's own GGUF settings and chat template through
unchanged, pairs vision projectors (mmproj) automatically, and ships a full control panel with live
monitoring, laid out for portrait monitors.

Target system this release is tuned for: **Linux Mint 22.2 x64 · AMD Ryzen 5 3600 · 64 GB RAM · 2 × AMD Radeon RX 6800 (16 GB)**.
Other Ubuntu 24.04-based systems, other GPUs (or no GPU), and Windows 10/11 work too.

---

## Contents

1. [Requirements](#requirements)
2. [Installation on Linux Mint](#installation-on-linux-mint)
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
14. [Windows](#windows)
15. [Development](#development)

---

## Requirements

| Item | Version |
|---|---|
| OS | **Linux Mint 22.x** (or Ubuntu 24.04 "noble"), x86-64 |
| **Python** | the system Python 3.12 (`/usr/bin/python3`); `setup.sh` creates a virtual environment in `.venv` |
| GPU driver | the kernel's `amdgpu` driver and **Mesa RADV** (Vulkan), both part of Mint. No ROCm install is needed. |
| llama.cpp | release **b11269** (official Ubuntu Vulkan build, downloaded by `setup.sh`) |
| App window | WebKit2GTK 4.1 for Python (`gir1.2-webkit2-4.1`, installed by `setup.sh`); without it the panel opens in the browser |
| Disk | ~300 MB for WinRunner + engine, plus your models |

Python packages (`requirements.txt`): `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic` 2, `psutil`, `pillow`,
`jinja2` and `pywebview` (the app window). No compiler, CUDA/ROCm SDK or Node.js is needed.

**About the llama.cpp version.** WinRunner pins llama.cpp **b11269** (it reports itself as `version 0.5.0-dev
(build 11269)`, ggml 0.25.3). The memory planner was verified against this build's allocator and log output.
llama.cpp publishes builds as `bNNNNN` tags; there is no llama.cpp or ggml release numbered "2.21.0". To install
another build, run `./setup.sh --engine-tag bNNNNN` (or `LLAMA_CPP_TAG=bNNNNN ./setup.sh`). Builds already
installed can be switched under **Settings › Engine**.

## Installation on Linux Mint

```bash
git clone <this repository> ~/winrunner      # or unpack the download
cd ~/winrunner
./setup.sh
```

Run it as your normal user (not with `sudo`); it asks for your password when it installs packages. `setup.sh`:

1. installs the system packages: Python venv, Mesa's Vulkan driver (RADV) and `vulkan-tools`, WebKit2GTK for
   the app window, and the llama.cpp runtime libraries (`libgomp1`, `libssl3`);
2. adds you to the `render` and `video` groups, which are needed for GPU compute (log out and back in once
   afterwards if it says so);
3. creates the Python environment in `.venv` (with the system's PyGObject, for the app window) and installs
   `requirements.txt`;
4. downloads the pinned **llama.cpp b11269 Vulkan** build into `data/engines/`;
5. checks that the engine sees your GPUs (`vulkaninfo --summary` should list both RX 6800s with driver RADV);
6. adds **WinRunner** to the application menu.

Options: `--engine-tag bNNNNN` installs another llama.cpp build; `--cpu` installs the CPU-only build;
`--no-apt` skips the system packages; `--firewall` also opens TCP 5070 in `ufw` for other computers.
Running `setup.sh` again is safe: it updates what is there.

## First start

| Launcher | What it does |
|---|---|
| **WinRunner** in the application menu | Starts WinRunner in its own application window. Console output goes to `data/logs/console.log`. |
| `./run.sh` | Same, from a terminal (log in the terminal). |
| `./run.sh --browser` | Control panel in your web browser. |
| `./run.sh --headless` | API server only (automatic over SSH, when there is no desktop session). |

On first start WinRunner scans these folders for `.gguf` files:

- `<install folder>/models`
- `~/.lmstudio/models` and `~/.cache/lm-studio/models` (LM Studio's folders: existing downloads are reused as-is)
- `~/.cache/llama.cpp` (models downloaded by `llama-server -hf`)

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

The default context is **65,536 tokens**. The context you set is the context the engine gets: WinRunner never
shortens it to make a model fit. (A context above the model's *trained* length is clamped to the trained length
unless you tick *Allow above trained context*, which uses RoPE scaling.)

In **Automatic** mode (default) WinRunner places the model **GPU first**:

1. **Free memory per GPU** is read from the engine itself (`llama-server --list-devices`), after any previously
   loaded model has actually released its memory. A **safety margin of 512 MiB per GPU** is kept free (adjustable,
   also per GPU, in *Settings › Hardware*). A 16 GB RX 6800 therefore ends at about 15.5 GiB in use.
2. **Everything on the GPUs if it fits.** Layers are split across the GPUs in contiguous ranges sized by each
   layer's real cost (weights + KV cache), with the output layer on the last GPU and the compute buffers and vision
   projector accounted for.
3. **If it does not fit: attention first.** The attention weights and the **KV cache of every layer stay on the
   GPUs**; only **feed-forward weights** (for mixture-of-experts models: expert weights) move to system RAM, one
   matrix at a time, until each GPU is filled exactly to its margin (`--override-tensor ...=CPU`). Only if the
   attention part of all layers cannot fit do the first layers run completely on the CPU.

   Why this matters: llama.cpp's own fallback moves *whole layers* to the CPU, including their KV cache and the
   attention over the whole context. The CPU then processes attention for every token of a long prompt, and the
   scheduler copies data back and forth between the GPUs and the CPU. That was the cause of the extreme slowness.
   Feed-forward weights in RAM cost far less: the CPU reads them once per generated token, and for long prompts
   llama.cpp streams them to the first GPU in large batches.
4. **KV cache precision.** *Auto* keeps an **F16** cache when model and context fit entirely in VRAM, otherwise it
   uses **Q8_0** (near-lossless, half the size) so that more weights stay on the GPUs. Q4 is never chosen
   automatically. The context length is not affected.
5. **Measured by the engine.** Before each load the plan is checked with llama.cpp's own allocator: WinRunner
   starts `llama-server` with the exact command line plus `--fit on`, reads its per-GPU memory projection and stops
   it before it reads the weights (a few seconds). The plan is corrected until each GPU lands on its margin.

The **Library › Load** tab shows all of this before you load: per-GPU stacked bars (weights, KV cache, compute
buffers, vision projector, margin, free), system RAM use, a per-layer placement map (a grey band marks layers whose
feed-forward weights are in RAM), the largest context that still fits entirely in VRAM for F16 and Q8_0, and the
exact `llama-server` command line.

*Manual* mode lets you set GPU layers, feed-forward / MoE layers on the CPU, tensor split, main GPU and split
mode yourself. *Settings › Engine › Automatic placement* can hand placement to llama.cpp's own `--fit` instead
(not recommended: it moves whole layers).

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
| Backend | **Vulkan (Mesa RADV)** | Part of Linux Mint; supports flash attention, quantized KV cache and multi-GPU on RDNA2 (gfx1030). WinRunner selects RADV even if AMD's AMDVLK driver is installed, and asks the driver to keep the model's buffers resident in VRAM (`GGML_VK_ENABLE_MEMORY_PRIORITY`). |
| Multi-GPU | **Layer split** across both RX 6800 | Only small activations cross PCIe between GPUs, so a secondary slot running at x4 costs little. Row split needs fast inter-GPU links and is not recommended. |
| VRAM use | Free VRAM − 512 MiB per GPU | Each GPU is filled to about 15.5 GiB. The GPU driving your display has less free VRAM; the plan uses the measured free memory of each GPU. |
| Overflow | Feed-forward weights to RAM, attention + KV on the GPUs | See [GPU allocation](#gpu-allocation-and-context-length). |
| Flash attention | Auto | Removes the huge attention scratch buffer at long context (tens of GiB at 64K without it). Required for a quantized V cache. |
| KV cache | F16 → Q8_0 when the model does not fit | Highest quality that still keeps the most on the GPUs. |
| Loading | Full read unless whole layers run on the CPU | Reads the file straight into VRAM / RAM once. |
| Threads | Engine default (6 = physical cores) | The CPU computes the feed-forward weights kept in RAM; that is limited by memory bandwidth, so SMT threads do not help. |
| Prompt cache | 8 GiB RAM | With 64 GB RAM, recent conversations resume instantly. |
| Large MoE models | Experts in RAM | 64 GB RAM plus 32 GB VRAM run models like gpt-oss-120b / GLM-4.5-Air with attention on the GPUs. |

Rough sizes for 2 × 16 GB, fully offloaded at 64K context: 7–14B models at Q8_0/Q6_K with an F16 KV cache;
24–32B models at Q4_K_M with a Q8_0 KV cache. Use **Library › Load › Max full-offload ctx** to see the exact limit
for each model. Larger models still keep attention on the GPUs and only overflow feed-forward weights.

Hardware tips (*Settings › Hardware* checks these):

- **Resizable BAR.** Enable *Above 4G decoding* and *Re-Size BAR* in the BIOS. Without it the CPU sees only a
  256 MiB window of each GPU's memory.
- **PCIe slots.** When weights are kept in RAM, llama.cpp streams them to the first GPU (`Vulkan0`) for long
  prompts, so that GPU should be the one in the x16 slot. Mesa lists the GPU driving the display first.
- A larger **micro-batch** (1024 or 2048, *Library › Load › Batching*) speeds up long prompts when weights are in
  RAM, at the cost of some VRAM for compute buffers.

### Vulkan or ROCm?

llama.cpp publishes official Linux builds for Vulkan and CPU only. The Vulkan build with RADV is the recommended
backend for the RX 6800 on Linux. A ROCm/HIP build of llama.cpp that you compiled yourself can be selected under
**Settings › Engine › Custom engine**; compare both with your own models on the **Benchmark** tab (`llama-bench`
with your exact load settings).

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

  Plus CPU per-thread load, RAM, engine process stats and a per-request speed history. On Linux the data comes from the
  amdgpu driver (`/sys/class/drm`: VRAM, utilisation, sensors) and each process's DRM memory statistics; on Windows
  from the GPU performance counters and the AMD driver's ADL sensors.
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

All settings persist in `data/settings.json`. Every setting has a tooltip.

## Command line

```
./run.sh [--window | --browser | --headless] [--host 0.0.0.0] [--port 5070] [--model MODEL_ID] [--data-dir DIR]
.venv/bin/python -m winrunner --install-engine {vulkan,cpu} [--engine-tag b11269]
.venv/bin/python -m winrunner --check          # engine, driver and GPUs as the engine sees them
```

## Files and folders

```
winrunner/
  setup.sh, run.sh      installer and launcher (Linux)
  winrunner/            application (Python package + control panel in winrunner/static)
  .venv/                Python environment created by setup.sh
  data/                 created on first start
    settings.json       all settings (themes, defaults, per-model profiles)
    engines/            downloaded llama.cpp builds (one folder per build/backend)
    cache/              GGUF header index
    chats/              saved chat console conversations
    templates/          custom chat templates
    logs/winrunner.log  application log (rotated); console.log when started from the menu
    benchmarks.json     benchmark history
  models/               default download folder (publisher/repository/file.gguf)
```

The whole folder is portable. Set `WINRUNNER_DATA` or `--data-dir` to keep data elsewhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `--check` / Settings › Hardware shows no GPU | Log out and back in after `setup.sh` (new `render`/`video` group membership). Check `vulkaninfo --summary`: it should list 2 × *AMD Radeon RX 6800 (RADV NAVI21)*. |
| Only one GPU is used | Both must appear under *Settings › Hardware › Engine devices*. Check *Library › Load › Devices* (both ticked) and the per-GPU margins. |
| GPUs are not filled to ~15.5 GiB | Other programs (browser, desktop) use VRAM on the display GPU; the plan uses what is actually free. The bars in *Library › Load* show "in use (other apps)". Lower the margin under *Settings › Hardware* if you want less headroom. |
| App window does not open, panel opens in the browser | Install WebKit2GTK for Python: `sudo apt install python3-gi gir1.2-webkit2-4.1`, then run `./setup.sh` again. |
| Other computers cannot connect | `./setup.sh --firewall` (or `sudo ufw allow 5070/tcp`). Check that Settings › Network › Bind address is `0.0.0.0`. |
| "No llama.cpp engine installed" | `./setup.sh`, or Settings › Engine › *Check for llama.cpp releases* › Install (Vulkan). |
| Load fails with out-of-memory | Raise the safety margin (*Settings › Hardware*) if other applications grab VRAM while the model loads. |
| Port 5070 already in use | Close the other program (e.g. a second WinRunner) or change the port in Settings › Network (restart required). |
| Model answers in a strange format | Keep *Template source: GGUF embedded*. Check the Chat Template tab; some old GGUFs have no template and need a built-in one. |
| Images rejected | The model needs its mmproj file in the same folder (Vision tab). |
| Control panel "not available on the network" | By default only this PC may open the panel; enable *Allow the control panel from other computers*. |
| Requests fail with `vk::Queue::submit: ErrorDeviceLost` | The GPU was reset. WinRunner restarts the engine automatically. On Linux check `sudo dmesg \| grep amdgpu` for ring timeouts, and remove GPU overclocks / undervolts. |

Logs: **Logs** tab, or `data/logs/winrunner.log`.

## Windows

WinRunner still runs on Windows 10/11: double-click `install.bat` (conda environment, Vulkan engine), then start
`WinRunner.bat`. `scripts\firewall.bat` opens the port, and `scripts\gpu-timeout.bat` raises the Windows GPU
timeout (TDR) from 2 to 60 seconds, which long prompts on large models can otherwise exceed ("ErrorDeviceLost").
The GPU-first allocation works the same way. On Windows the Vulkan driver reports only the video memory *budget*
Windows grants each process, about 0.7–1.2 GB less than the free VRAM Task Manager shows; planning with that figure
stopped the GPUs at ~14.4–14.8 GB. WinRunner therefore plans with the physical free VRAM from the GPU performance
counters (total minus the dedicated memory of all processes), so each 16 GB card is filled to about 15.5 GB with the
512 MiB margin (*Settings › Hardware › Use all physical VRAM*, on by default). After every load it checks with the
same counters that the engine's buffers really are in VRAM; if Windows moved some of them to shared system memory
(slow), the activity log says so and the next load leaves that much more room on that GPU.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The unit tests cover GGUF parsing, the memory planner, log parsing, command-line generation, image normalisation,
the stream proxy and settings. `tests/test_recovery.py` checks engine failure recovery end to end: it runs the full
server against a scripted fake `llama-server` (`tests/fake_engine.py`, Linux only) that loses its GPU, aborts or
crashes while idle.

`tests/test_integration.py` runs the full server against a real engine. Set these first:

- `WINRUNNER_TEST_ENGINE`: path to `llama-server`
- `WINRUNNER_TEST_MODELS`: a folder containing Qwen3-0.6B and SmolVLM-256M + mmproj GGUFs

`tests/preview_server.py` starts the app with two simulated RX 6800s, for UI work on machines without those GPUs.
It is for development only; the product never reports simulated hardware.

## License

WinRunner is released under the MIT License. llama.cpp / ggml are MIT licensed
(© The ggml authors); they are downloaded from their official GitHub releases at install time.
