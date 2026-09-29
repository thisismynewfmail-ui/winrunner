"""WinRunner - local LLM inference server with an LM Studio / OpenAI compatible API.

WinRunner manages llama.cpp's ``llama-server`` engine: it reads GGUF metadata,
plans GPU memory allocation, launches and supervises the engine, and exposes an
OpenAI / LM Studio compatible HTTP API (with vision) on the local network,
together with a monitoring control panel.
"""

__version__ = "1.0.0"
PRODUCT_NAME = "WinRunner"
PRODUCT_TAGLINE = "Local Inference Server"
