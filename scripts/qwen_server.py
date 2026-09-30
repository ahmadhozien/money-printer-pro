"""
Minimal Qwen-Image HTTP server for MoneyPrinterV2.

Self-contained on purpose: copy this single file to whatever box owns the GPU
(a rented Runpod/Vast instance, a desktop on the LAN, or localhost) and run it.
It imports nothing from src/, so there is no repo to sync on the far end.

    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers transformers accelerate safetensors pillow bitsandbytes
    python qwen_server.py

Environment:
    QWEN_MODEL      HF model id (default Qwen/Qwen-Image-2512)
    QWEN_API_KEY    shared token; REQUIRED unless bound to loopback
    QWEN_HOST       bind address (default 127.0.0.1)
    QWEN_PORT       bind port (default 8189)
    QWEN_QUANT      none | 4bit (default: auto by VRAM, see pick_quant_mode)
    QWEN_COMPILE    1 = torch.compile the transformer; ~2.4x faster once warm,
                    but the first render pays several minutes of compilation.
                    Worth it for a batch, not for a one-off.

Protocol:
    GET  /health   -> {"status": "ok", "model": ...}
    POST /generate -> raw PNG bytes (Content-Type: image/png)
        {"prompt": str, "width": int, "height": int,
         "steps": int, "negative_prompt": str}
"""

import hmac
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen-Image-2512")
API_KEY = os.environ.get("QWEN_API_KEY", "")
HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_PORT", "8189"))
QUANT = os.environ.get("QWEN_QUANT", "").strip().lower()
COMPILE = os.environ.get("QWEN_COMPILE", "") == "1"

# Qwen-Image's trained resolutions. Requests are snapped to this set: arbitrary
# sizes both degrade output and let a caller ask for a ruinous allocation.
ALLOWED_DIMENSIONS = {
    (1328, 1328), (1664, 928), (928, 1664), (1472, 1104),
    (1104, 1472), (1584, 1056), (1056, 1584),
}
DEFAULT_DIMENSIONS = (928, 1664)
MAX_PROMPT_CHARS = 4000

# Recommended by the model card; suppresses the artefacts Qwen is prone to.
DEFAULT_NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, deformed fingers, "
    "oversaturated, waxy skin, featureless face, overly smooth, AI-looking, "
    "cluttered composition, blurry text, distorted text"
)

pipe = None


def pick_quant_mode(vram_gb: float) -> str:
    """
    Chooses a precision for the detected card. Qwen-Image is ~41GB in bf16 and
    ~15GB in 4-bit nf4, so anything short of an 80GB card wants quantizing:
    a resident 4-bit model beats an offloaded bf16 one by a wide margin,
    because offload is bound by PCIe rather than by the GPU.

    Args:
        vram_gb (float): total VRAM of the visible device

    Returns:
        mode (str): "none" or "4bit"
    """
    if QUANT in ("none", "4bit"):
        return QUANT
    return "none" if vram_gb >= 48 else "4bit"


def load_pipeline():
    """
    Loads the Qwen-Image pipeline once at startup, quantizing and offloading
    only as far as the detected card actually requires.

    Returns:
        pipe: the ready diffusers pipeline
    """
    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available - this will run on CPU and take hours "
            "per image. Install a CUDA build of torch.",
            file=sys.stderr,
        )
        print(f"Loading {MODEL_ID} (float32, CPU)...")
        return DiffusionPipeline.from_pretrained(MODEL_ID, dtype=torch.float32).to("cpu")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    mode = pick_quant_mode(vram_gb)
    name = torch.cuda.get_device_name(0)
    print(f"{name}, {vram_gb:.0f}GB VRAM -> precision: {mode}")

    kwargs = {"dtype": torch.bfloat16}
    if mode == "4bit":
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )

    print(f"Loading {MODEL_ID}... first run downloads the weights (~40GB).")
    loaded = DiffusionPipeline.from_pretrained(MODEL_ID, device_map="cuda", **kwargs)

    # Only fall back to offload when even the quantized model will not fit.
    if vram_gb < 16:
        loaded.enable_model_cpu_offload()
        print("Model CPU offload enabled (card too small to hold the weights).")

    if COMPILE:
        print("Compiling transformer; the first render will be slow...")
        loaded.transformer = torch.compile(loaded.transformer)

    print(f"Ready. VRAM reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.1f}GB")
    return loaded


def render(prompt: str, width: int, height: int, steps: int, negative_prompt: str) -> bytes:
    """
    Renders one image and encodes it as PNG.

    Args:
        prompt (str): text prompt
        width (int): image width
        height (int): image height
        steps (int): sampling steps
        negative_prompt (str): things to suppress

    Returns:
        png (bytes): encoded PNG payload
    """
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=4.0,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        return hmac.compare_digest(token, API_KEY)

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, {"status": "ok", "model": MODEL_ID, "ready": pipe is not None})

    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        if not self._authorized():
            self._send_json(401, {"error": "invalid or missing bearer token"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "bad content length"})
            return

        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"error": "empty or oversized body"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "body is not valid JSON"})
            return

        prompt = str(payload.get("prompt", "")).strip()[:MAX_PROMPT_CHARS]
        if not prompt:
            self._send_json(400, {"error": "prompt is required"})
            return

        try:
            width = int(payload.get("width", DEFAULT_DIMENSIONS[0]))
            height = int(payload.get("height", DEFAULT_DIMENSIONS[1]))
            steps = int(payload.get("steps", 30))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "width, height and steps must be integers"})
            return

        if (width, height) not in ALLOWED_DIMENSIONS:
            width, height = DEFAULT_DIMENSIONS
        steps = max(1, min(steps, 100))

        negative_prompt = str(payload.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)[:MAX_PROMPT_CHARS]

        started = time.time()
        try:
            png = render(prompt, width, height, steps, negative_prompt)
        except Exception as exc:
            print(f"render failed: {exc}", file=sys.stderr)
            self._send_json(500, {"error": f"render failed: {exc}"})
            return

        print(f"rendered {width}x{height} in {time.time() - started:.1f}s ({steps} steps)")
        self._send(200, png, "image/png")

    def log_message(self, fmt, *args):
        pass  # request logging is noise; render timings are printed instead


def main():
    global pipe

    if not API_KEY and HOST not in ("127.0.0.1", "localhost"):
        sys.exit(
            f"Refusing to bind {HOST} without QWEN_API_KEY: an unauthenticated "
            "GPU endpoint on a public interface is open to anyone who finds it."
        )

    pipe = load_pipeline()
    print(f"Qwen-Image server ready on http://{HOST}:{PORT}")
    # ponytail: threaded server, but one GPU renders one image at a time anyway.
    # Add a real queue only if you ever front multiple GPUs with this.
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
