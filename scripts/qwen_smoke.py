"""
Renders one real image through the configured Qwen-Image server and reports
what it cost in wall-clock time. Run this after standing up a pod, before
committing a whole video run to it.

    python scripts/qwen_smoke.py
    python scripts/qwen_smoke.py --steps 20 --rate 0.34
"""

import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import (  # noqa: E402
    get_nanobanana2_aspect_ratio,
    get_qwen_api_base_url,
    get_qwen_api_key,
    get_qwen_dimensions,
    get_qwen_request_timeout,
    get_qwen_steps,
    get_max_image_prompts,
)

DEFAULT_PROMPT = (
    "Cinematic wide shot of a lone figure walking through a sunlit desert "
    "canyon, warm golden hour light, volumetric dust, shallow depth of field"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=None, help="override qwen_steps")
    parser.add_argument("--rate", type=float, default=0.0, help="GPU $/hour, to estimate cost per video")
    parser.add_argument("--out", default="qwen_smoke.png")
    args = parser.parse_args()

    base_url = get_qwen_api_base_url()
    steps = args.steps or get_qwen_steps()
    width, height = get_qwen_dimensions(get_nanobanana2_aspect_ratio())

    print(f"server : {base_url}")
    print(f"render : {width}x{height}, {steps} steps")

    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        health.raise_for_status()
        print(f"health : {health.json()}")
    except requests.RequestException as exc:
        sys.exit(
            f"Cannot reach {base_url}: {exc}\n"
            "If the server is on a pod, check the SSH tunnel is still up."
        )

    headers = {}
    api_key = get_qwen_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.time()
    try:
        response = requests.post(
            f"{base_url}/generate",
            headers=headers,
            json={"prompt": args.prompt, "width": width, "height": height, "steps": steps},
            timeout=get_qwen_request_timeout(),
        )
    except requests.RequestException as exc:
        sys.exit(f"Render request failed: {exc}")

    elapsed = time.time() - started

    if response.status_code != 200:
        sys.exit(f"Server returned {response.status_code}: {response.text[:500]}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        sys.exit(f"Expected an image, got {content_type}: {response.text[:500]}")

    with open(args.out, "wb") as handle:
        handle.write(response.content)

    size_kb = len(response.content) / 1024
    print(f"\nOK  {elapsed:.1f}s  {size_kb:.0f}KB -> {args.out}")

    if args.rate:
        scenes = get_max_image_prompts() or 20
        per_image = args.rate * elapsed / 3600
        print(
            f"at ${args.rate:.2f}/hr: ${per_image:.4f}/image, "
            f"${per_image * scenes:.2f} for a {scenes}-scene video "
            f"({elapsed * scenes / 60:.0f} min of GPU time)"
        )
        print("compare: $0.039/image, $%.2f/video on Nano Banana 2" % (0.039 * scenes))


if __name__ == "__main__":
    main()
