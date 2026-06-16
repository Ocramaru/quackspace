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
DEFAULT_URL = "http://localhost:11434/api/embed"


def embed(text: str, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL) -> list[float]:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. Start Ollama and run "
            f"`ollama pull {model}`, or choose the built-in embedder."
        ) from e
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
