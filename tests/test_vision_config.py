import asyncio
import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from winrunner.config import LoadParams, ModelProfile, SettingsStore
from winrunner.templates import analyze, render
from winrunner.vision import ImageError, ImageNormalizer, convert_image, sniff


def enc(fmt, size=(64, 48), mode="RGB", **kw):
    b = io.BytesIO()
    Image.new(mode, size, (10, 200, 30) if mode == "RGB" else None).save(b, fmt, **kw)
    return b.getvalue()


@pytest.mark.parametrize("fmt,expect_same", [("PNG", True), ("JPEG", True), ("BMP", True), ("GIF", True), ("WEBP", False), ("TIFF", False)])
def test_convert_only_when_needed(fmt, expect_same):
    data = enc(fmt)
    out, new_fmt, note = convert_image(data)
    assert (out is data) == expect_same
    assert new_fmt in ("png", "jpeg", "bmp", "gif")
    assert bool(note) != expect_same


def test_exif_rotation_applied():
    ex = Image.Exif()
    ex[0x0112] = 6
    b = io.BytesIO()
    Image.new("RGB", (80, 40)).save(b, "JPEG", exif=ex.tobytes())
    out, fmt, note = convert_image(b.getvalue())
    assert "EXIF" in note and Image.open(io.BytesIO(out)).size == (40, 80)


def test_downscale_and_alpha():
    b = io.BytesIO()
    Image.new("RGBA", (1000, 500), (0, 0, 0, 0)).save(b, "WEBP")
    out, fmt, note = convert_image(b.getvalue(), max_edge=256)
    im = Image.open(io.BytesIO(out))
    assert fmt == "png" and im.mode == "RGBA" and max(im.size) == 256


def test_normalize_chat_variants():
    async def run():
        async with httpx.AsyncClient() as c:
            n = ImageNormalizer(c, fetch_remote=False)
            webp = "data:image/webp;base64," + base64.b64encode(enc("WEBP")).decode()
            png = base64.b64encode(enc("PNG")).decode()
            msgs = [
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": webp}]},
                {"role": "user", "content": [{"type": "input_image", "image_url": {"url": f"data:image/png;base64,{png}"}}]},
                {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}]},
                {"role": "user", "content": "b", "images": [png]},
            ]
            st = await n.normalize_chat(msgs)
            assert st.images == 4 and st.converted == ["webp -> png/jpeg"]
            parts = [p for m in msgs for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
            assert len(parts) == 4 and all(p["image_url"]["url"].startswith("data:image/") for p in parts)
            assert parts[0]["image_url"]["url"].startswith("data:image/jpeg")
            with pytest.raises(ImageError):
                await n.normalize_chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}])
    asyncio.run(run())


def test_sniff():
    assert sniff(enc("PNG")) == "png" and sniff(enc("WEBP")) == "webp" and sniff(b"xxxx") == "unknown"


def test_settings_roundtrip_and_replace(tmp_path: Path):
    s = SettingsStore(tmp_path / "settings.json")
    assert s.settings.server.port == 5070 and s.settings.defaults.context_length == 65536
    s.update({"ui": {"custom_themes": {"a": {"--acc": "#fff"}, "b": {"--acc": "#000"}}}})
    s.update({"ui": {"custom_themes": {"b": {"--acc": "#000"}}}})  # replaced, not merged
    assert list(s.settings.ui.custom_themes) == ["b"]
    s.update({"server": {"jit_loading": False}})
    s2 = SettingsStore(tmp_path / "settings.json")
    assert s2.settings.server.jit_loading is False and s2.settings.server.port == 5070
    with pytest.raises(Exception):
        s.update({"server": {"port": 70000}})


def test_settings_corrupt_file_backed_up(tmp_path: Path):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    s = SettingsStore(p)
    assert s.settings.server.port == 5070
    assert any(x.name.startswith("settings.invalid-") for x in tmp_path.iterdir())


def test_effective_params_layering(tmp_path: Path):
    s = SettingsStore(tmp_path / "settings.json")
    s.update({"defaults": {"ubatch_size": 1024}})
    s.set_profile("/m/a.gguf", ModelProfile(load={"context_length": 16384, "bogus": 1}))
    p = s.effective_load_params("/m/a.gguf", {"flash_attn": "on"})
    assert (p.context_length, p.ubatch_size, p.flash_attn) == (16384, 1024, "on")
    with pytest.raises(Exception):
        LoadParams(kv_cache_type="q3_z")


def test_template_analysis_and_render():
    t = ("{%- if tools %}{{ '<|im_start|>system\\n' }}{%- endif %}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")
    a = analyze(t)
    assert a["family"] == "ChatML" and a["reasoning"]
    out = render(t, [{"role": "user", "content": "hi"}])
    assert out.endswith("<|im_start|>assistant\n<think>\n") and "<|im_start|>user\nhi<|im_end|>" in out
    assert analyze("{{ '<|start_header_id|>' }}")["family"] == "Llama 3"
    assert json.dumps(analyze(""))
