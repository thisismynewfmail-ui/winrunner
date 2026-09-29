"""Image input normalisation for vision models.

llama.cpp's multimodal loader decodes JPEG, PNG, BMP and GIF. Clients send
images in many shapes (OpenAI ``image_url`` objects or strings, Responses API
``input_image``, Anthropic ``image`` blocks, Ollama-style ``images`` arrays,
remote URLs, WebP/HEIC/TIFF files, phone photos with EXIF rotation). This module
converts all of them into data URIs the engine accepts, without re-encoding
images that are already compatible.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("winrunner.vision")

NATIVE = {"jpeg", "png", "bmp", "gif"}
MAX_IMAGE_BYTES = 40 * 1024 * 1024
_DATA_URI = re.compile(r"^data:([\w/+.-]*)?((?:;[\w-]+=[^;,]*)*)(;base64)?,(.*)$", re.DOTALL | re.IGNORECASE)


class ImageError(Exception):
    pass


@dataclass
class VisionStats:
    images: int = 0
    audio: int = 0
    converted: list[str] = field(default_factory=list)
    fetched: int = 0


def sniff(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] in (b"GIF8",):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypmsf1", b"ftyphevc"):
        return "heic"
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return "avif"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def _needs_exif_fix(data: bytes) -> bool:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            return int(exif.get(0x0112, 1)) not in (0, 1)
    except Exception:
        return False


def _cmyk_or_unusual(data: bytes) -> bool:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.mode in ("CMYK", "YCbCr", "I;16", "I", "F", "LAB")
    except Exception:
        return False


def _dims(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def convert_image(data: bytes, max_edge: int = 0) -> tuple[bytes, str, str]:
    """Return (bytes, format, note). Only re-encodes when necessary."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
    fmt = sniff(data)
    reasons = []
    if fmt not in NATIVE:
        reasons.append(f"{fmt} -> png/jpeg")
    if fmt == "jpeg" and _needs_exif_fix(data):
        reasons.append("EXIF orientation applied")
    if fmt in ("jpeg", "png", "tiff") and _cmyk_or_unusual(data):
        reasons.append("colour mode -> RGB")
    if max_edge > 0:
        d = _dims(data)
        if d and max(d) > max_edge:
            reasons.append(f"downscaled from {d[0]}x{d[1]}")
    if not reasons:
        return data, fmt, ""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise ImageError("Pillow is required to convert this image") from exc
    if fmt in ("heic", "avif"):
        try:  # optional plugin
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except Exception:
            pass
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
            im = im.convert("RGBA" if has_alpha else "RGB")
            if max_edge > 0 and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            lossy_source = fmt in ("jpeg", "webp", "heic", "avif")
            if has_alpha or not lossy_source:
                im.save(out, "PNG", optimize=False)
                new_fmt = "png"
            else:
                im.save(out, "JPEG", quality=95, subsampling=0)
                new_fmt = "jpeg"
            return out.getvalue(), new_fmt, ", ".join(reasons)
    except ImageError:
        raise
    except Exception as exc:
        raise ImageError(f"could not decode image ({fmt}): {exc}") from exc


def decode_data_uri(uri: str) -> bytes:
    m = _DATA_URI.match(uri.strip())
    if not m:
        raise ImageError("malformed data URI")
    if not m.group(3):
        raise ImageError("data URI must be base64 encoded")
    return _b64(m.group(4))


def _b64(s: str) -> bytes:
    s = re.sub(r"\s+", "", s)
    pad = (-len(s)) % 4
    try:
        return base64.b64decode(s + "=" * pad, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"invalid base64 image data: {exc}") from exc


def to_data_uri(data: bytes, fmt: str) -> str:
    return f"data:image/{fmt};base64," + base64.b64encode(data).decode("ascii")


class ImageNormalizer:
    def __init__(self, http: httpx.AsyncClient, max_edge: int = 0, fetch_remote: bool = True):
        self.http = http
        self.max_edge = max_edge
        self.fetch_remote = fetch_remote

    async def _load(self, ref: str, stats: VisionStats) -> bytes:
        ref = ref.strip()
        if ref.startswith("data:"):
            return decode_data_uri(ref)
        if ref.startswith(("http://", "https://")):
            if not self.fetch_remote:
                raise ImageError("remote image URLs are disabled in WinRunner settings; send a base64 data URI")
            try:
                r = await self.http.get(ref, timeout=20.0, follow_redirects=True)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise ImageError(f"could not download image {ref[:120]}: {exc}") from exc
            if len(r.content) > MAX_IMAGE_BYTES:
                raise ImageError("downloaded image is too large")
            stats.fetched += 1
            return r.content
        if len(ref) > 64 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", ref[:256]):
            return _b64(ref)  # raw base64 without a data: prefix
        raise ImageError("unsupported image reference (expected data URI, http(s) URL or base64)")

    async def _process(self, ref: str, stats: VisionStats) -> tuple[str, str]:
        raw = await self._load(ref, stats)
        data, fmt, note = await asyncio.to_thread(convert_image, raw, self.max_edge)
        if note:
            stats.converted.append(note)
        stats.images += 1
        if not note and ref.startswith("data:"):
            declared = (_DATA_URI.match(ref.strip()) or [None, ""])[1] or ""
            if declared.lower() in (f"image/{fmt}", "image/jpg" if fmt == "jpeg" else f"image/{fmt}"):
                return ref, fmt
        return to_data_uri(data, fmt), fmt

    async def normalize_chat(self, messages: list[dict[str, Any]]) -> VisionStats:
        """OpenAI chat messages (in place)."""
        stats = VisionStats()
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            imgs = msg.pop("images", None)  # Ollama-style
            if isinstance(imgs, list) and imgs:
                content = msg.get("content")
                parts: list[Any] = [{"type": "text", "text": content}] if isinstance(content, str) and content else (
                    list(content) if isinstance(content, list) else [])
                for b in imgs:
                    parts.append({"type": "image_url", "image_url": {"url": b if str(b).startswith("data:") else
                                                                     f"data:image/png;base64,{b}"}})
                msg["content"] = parts
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_parts = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                t = part.get("type")
                ref = None
                if t == "image_url":
                    iu = part.get("image_url")
                    ref = iu.get("url") if isinstance(iu, dict) else iu
                elif t in ("input_image", "image"):
                    iu = part.get("image_url") or part.get("image") or part.get("url")
                    if isinstance(iu, dict):
                        iu = iu.get("url")
                    src = part.get("source")
                    if not iu and isinstance(src, dict):
                        if src.get("type") == "base64":
                            iu = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                        else:
                            iu = src.get("url")
                    ref = iu
                elif t in ("input_audio", "audio"):
                    stats.audio += 1
                if ref is None:
                    new_parts.append(part)
                    continue
                if not isinstance(ref, str) or not ref:
                    raise ImageError("image part without an image URL or data")
                url, _ = await self._process(ref, stats)
                np_: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
                if isinstance(part.get("image_url"), dict) and part["image_url"].get("detail"):
                    np_["image_url"]["detail"] = part["image_url"]["detail"]
                new_parts.append(np_)
            msg["content"] = new_parts
        return stats

    async def normalize_responses(self, body: dict[str, Any]) -> VisionStats:
        """OpenAI Responses API ``input`` (in place)."""
        stats = VisionStats()
        items = body.get("input")
        if not isinstance(items, list):
            return stats
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    ref = part.get("image_url")
                    if isinstance(ref, dict):
                        ref = ref.get("url")
                    if isinstance(ref, str) and ref:
                        part["image_url"], _ = await self._process(ref, stats)
        return stats

    async def normalize_anthropic(self, body: dict[str, Any]) -> VisionStats:
        """Anthropic Messages API image blocks (in place)."""
        stats = VisionStats()
        for msg in body.get("messages") or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                src = part.get("source") or {}
                if src.get("type") == "base64":
                    ref = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                elif src.get("type") == "url":
                    ref = src.get("url", "")
                else:
                    continue
                url, fmt = await self._process(ref, stats)
                m = _DATA_URI.match(url)
                part["source"] = {"type": "base64", "media_type": f"image/{fmt}", "data": m.group(4) if m else ""}
        return stats
