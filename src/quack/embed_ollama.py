"""Ollama embedding provider wrapper.

Reads text from stdin and prints one JSON array of floats, matching the
``embed.command`` contract. It talks to the local Ollama server and defaults to
``nomic-embed-text``.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "nomic-embed-text"
# Try both IPv4 and IPv6 loopback — on Linux 'localhost' may resolve to ::1
# while Ollama binds 0.0.0.0, or vice versa depending on system config.
_EMBED_URLS = [
    "http://127.0.0.1:11434/api/embed",
    "http://[::1]:11434/api/embed",
]
DEFAULT_URL = _EMBED_URLS[0]


def embed(text: str, model: str = DEFAULT_MODEL, url: str | None = None) -> list[float]:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    urls = [url] if url is not None else _EMBED_URLS
    data = None
    last_err: Exception | None = None
    for try_url in urls:
        req = urllib.request.Request(
            try_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.URLError as e:
            last_err = e
    if data is None:
        tried = ", ".join(urls)
        raise RuntimeError(
            f"Could not reach Ollama (tried: {tried}). "
            f"Is Ollama running? Try: ollama serve && ollama pull {model}"
        ) from last_err
    vectors = data.get("embeddings") or data.get("embedding")
    if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
        return [float(x) for x in vectors[0]]
    if isinstance(vectors, list):
        return [float(x) for x in vectors]
    raise RuntimeError("Ollama did not return an embedding vector.")


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    text = sys.stdin.read()
    print(json.dumps(embed(text, model=model)))


if __name__ == "__main__":
    main()
