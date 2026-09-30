"""
Self-check for the Qwen-Image wire contract. Runs without a GPU: the real
request handler is exercised against a stubbed renderer, so it catches auth,
validation and content-type regressions that would otherwise only show up
mid-render on a rented box.

    python scripts/test_qwen_local.py
"""

import os
import sys
import threading
from http.server import ThreadingHTTPServer

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import qwen_server
from config import get_qwen_dimensions

# 1x1 PNG; standing in for whatever the pipeline would have produced.
STUB_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)

calls = []


def stub_render(prompt, width, height, steps, negative_prompt):
    calls.append({"prompt": prompt, "width": width, "height": height, "steps": steps})
    return STUB_PNG


def main():
    qwen_server.render = stub_render
    qwen_server.API_KEY = "test-token"

    server = ThreadingHTTPServer(("127.0.0.1", 0), qwen_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    auth = {"Authorization": "Bearer test-token"}

    try:
        # Dimension table agrees with what the server will accept, so the
        # client can never ask for a size the server silently rewrites.
        assert get_qwen_dimensions("9:16") == (928, 1664)
        assert get_qwen_dimensions("16:9") == (1664, 928)
        assert get_qwen_dimensions("nonsense") == (928, 1664), "unknown ratio must fall back to portrait"
        for dimensions in (get_qwen_dimensions(r) for r in ("1:1", "16:9", "9:16", "4:3", "3:4")):
            assert dimensions in qwen_server.ALLOWED_DIMENSIONS, f"{dimensions} rejected by server"

        assert requests.get(f"{base}/health").json()["status"] == "ok"

        # Auth is enforced whenever a token is configured.
        assert requests.post(f"{base}/generate", json={"prompt": "x"}).status_code == 401
        assert requests.post(
            f"{base}/generate", json={"prompt": "x"}, headers={"Authorization": "Bearer wrong"}
        ).status_code == 401

        assert requests.post(f"{base}/generate", json={"prompt": "  "}, headers=auth).status_code == 400
        assert requests.post(f"{base}/generate", data="{not json", headers=auth).status_code == 400
        assert requests.post(f"{base}/nope", json={"prompt": "x"}, headers=auth).status_code == 404

        # Happy path returns raw PNG bytes, which is what _persist_image writes.
        ok = requests.post(
            f"{base}/generate",
            json={"prompt": "a cat", "width": 928, "height": 1664, "steps": 30},
            headers=auth,
        )
        assert ok.status_code == 200, ok.status_code
        assert ok.headers["Content-Type"] == "image/png"
        assert ok.content == STUB_PNG
        assert ok.content[:8] == b"\x89PNG\r\n\x1a\n", "client checks the magic bytes implicitly via PIL"

        # Off-grid sizes snap instead of allocating whatever was asked for.
        requests.post(
            f"{base}/generate",
            json={"prompt": "a cat", "width": 99999, "height": 99999, "steps": 500},
            headers=auth,
        )
        assert calls[-1]["width"] == 928 and calls[-1]["height"] == 1664, calls[-1]
        assert calls[-1]["steps"] == 100, "steps must clamp to 100"
    finally:
        server.shutdown()

    print(f"OK - {len(calls)} renders dispatched, contract holds")


if __name__ == "__main__":
    main()
