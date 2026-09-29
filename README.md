# WinRunner — Local Inference Server

WinRunner runs GGUF language and vision models on your own PC and serves them to every device on your
network through an **OpenAI- and LM Studio-compatible API** at

```
http://<this-computer>:5070/v1
```

It drives the official [llama.cpp](https://github.com/ggml-org/llama.cpp) engine (`llama-server`), plans GPU
memory for the context length you ask for, passes each model's own GGUF settings and chat template through
unchanged, pairs vision projectors (mmproj) automatically, and ships a full control panel with live
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
| Browser engine for the app window | Microsoft Edge WebView2 Runtime (preinstalled on current Windows 10/11) |
| Disk | ~300 MB for WinRunner + engine, plus your models |

Python packages (`requirements.txt`): `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic` 2, `psutil`, `pillow`,
`jinja2`, and `pywebview` (Windows, for the native window). No compiler, CUDA/ROCm SDK or Node.js is needed.

## Installation (conda)

### Automatic (Windows)

1. Install Miniconda if you do not have conda: <https://docs.conda.io/en/latest/miniconda.html>
2. Download or clone this repository, e.g. to `C:\WinRunner`.
3. Double-click **`install.bat`** (or run it from an Anaconda Prompt). It:
   - creates the conda environment **`winrunner` with Python 3.11**,
   - installs `requirements.txt`,
   - downloads the latest official **llama.cpp Vulkan** build into `data\engines\`.

   Use `install.bat rocm` to download the ROCm/HIP build instead (see [backends](#vulkan-or-rocm)).
4. Optional, for access from other computers: right-click **`scripts\firewall.bat`** → *Run as administrator*
   (opens TCP 5070 on private networks).

### Manual

```bat
conda create -n winrunner python=3.11 -y
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
  style `publisher/repo` names and unique prefixes are also accepted. If a client sends an unknown name
  (e.g. `gpt-4o`), the currently loaded model answers.

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

## GPU allocation and context length

The default context is **65,536 tokens**. Before every load WinRunner plans memory for the chosen context:

1. **Context.** The requested length is clamped to the model's trained context unless you tick
   *Allow above trained context*, which uses RoPE scaling.
2. **KV cache precision.** *Auto* keeps an **F16** cache. It switches to **Q8_0** (near-lossless, half the size)
   only if that is what allows the entire model plus the full context to stay in VRAM. Q4 is never chosen
   automatically.
3. **Devices.** Free VRAM of each GPU is read from the engine itself (`llama-server --list-devices`), minus a
   safety margin (1 GiB per GPU by default; adjustable, also per GPU).
4. **Layer split.** Layers are assigned to GPUs in contiguous ranges sized by each layer's real cost:
   weights + KV cache, with the output layer on the last GPU and the vision projector and compute buffers
   accounted for. This mirrors llama.cpp's own assignment (`--tensor-split`).
5. **If the model does not fit.**
   - *Mixture-of-experts models:* all attention and KV stay on the GPUs and only the expert weights of the first
     N layers move to system RAM (`--n-cpu-moe`). This is much faster than moving whole layers.
   - *Dense models:* the minimum number of layers runs on the CPU.
6. **Engine verification.** Builds that include `llama-fit-params` (all current releases) verify the plan with
   llama.cpp's own allocator before loading. In automatic mode the engine's `--fit` then places layers within
   your safety margins.

The **Library › Load** tab shows all of this before you load: per-GPU stacked bars (weights, KV cache, compute
buffers, vision projector, margin, free), a per-layer placement map, the largest context that still fits entirely
in VRAM for F16 and Q8_0, and the exact `llama-server` command line.

*Manual* mode lets you set GPU layers, tensor split, main GPU, split mode and MoE CPU layers yourself.

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
| KV cache | F16 → Q8_0 only if needed | Highest quality that still keeps the whole model in 32 GB of VRAM. |
| Loading | Full read when fully offloaded, otherwise mmap | Faster, measurable loads straight into VRAM on Windows. mmap for partial offload avoids a second RAM copy. |
| Threads | Engine default (6 = physical cores) | With full offload the CPU only schedules work, so SMT threads do not help. |
| Process priority | Above normal | Keeps token generation smooth while you use the desktop. |
| Prompt cache | 8 GiB RAM | With 64 GB RAM, recent conversations resume instantly. |
| Large MoE models | Experts in RAM | 64 GB RAM plus 32 GB VRAM run models like gpt-oss-120b / GLM-4.5-Air with attention on the GPUs. |

Rough sizes for 2 × 16 GB, fully offloaded at 64K context: 7–14B models at Q8_0/Q6_K with an F16 KV cache;
24–32B models at Q4_K_M with a Q8_0 KV cache. Use **Library › Load › Max full-offload ctx** to see the exact limit for each model.

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
| Other computers cannot connect | Run `scripts\firewall.bat` as administrator (or allow Python on *private* networks when Windows asks). Check that Settings › Network › Bind address is `0.0.0.0`. |
| "No llama.cpp engine installed" | Settings › Engine › *Check for llama.cpp releases* › Install (Vulkan). |
| Engine reports no GPUs | Update the Adrenalin driver; *Re-detect engine and devices*. For ROCm builds install the AMD HIP SDK. |
| Load fails with out-of-memory | Lower the context, set KV cache to Q8_0, or raise the safety margin if other applications use VRAM. The memory plan shows the largest context that fits. |
| Port 5070 already in use | Close the other program (e.g. a second WinRunner) or change the port in Settings › Network (restart required). |
| Model answers in a strange format | Keep *Template source: GGUF embedded*. Check the Chat Template tab; some old GGUFs have no template and need a built-in one. |
| Images rejected | The model needs its mmproj file in the same folder (Vision tab). |
| Control panel "not available on the network" | By default only this PC may open the panel; enable *Allow the control panel from other computers*. |

Logs: **Logs** tab, or `data\logs\winrunner.log`.

## Development

```bat
pip install -r requirements-dev.txt
python -m pytest
```

The unit tests cover GGUF parsing, the memory planner, log parsing, command-line generation, image normalisation,
the stream proxy and settings.

`tests\test_integration.py` runs the full server against a real engine. Set these first:

- `WINRUNNER_TEST_ENGINE`: path to `llama-server`
- `WINRUNNER_TEST_MODELS`: a folder containing Qwen3-0.6B and SmolVLM-256M + mmproj GGUFs

`tests\preview_server.py` starts the app with two simulated RX 6800s, for UI work on machines without those GPUs.
It is for development only; the product never reports simulated hardware.

## License

WinRunner is released under the MIT License. llama.cpp / ggml are MIT licensed
(© The ggml authors); they are downloaded from their official GitHub releases at install time.
