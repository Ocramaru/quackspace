"""Built-in free embedding provider for ``quack embed``.

This is intentionally small and dependency-free. It creates a deterministic
hashed text vector from words and character n-grams, then L2-normalizes it. It
is not a replacement for a real embedding model, but it gives every QuackSpace
install a useful local default before users choose a stronger provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys

DIM = 256
WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def embed(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    for token in tokens:
        _add(vec, token, 1.0)
        if "/" in token or "." in token:
            for part in re.split(r"[./:-]+", token):
                if part:
                    _add(vec, part, 0.7)
        padded = f" {token} "
        for i in range(max(0, len(padded) - 2)):
            _add(vec, padded[i : i + 3], 0.35)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def _add(vec: list[float], feature: str, weight: float) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big")
    sign = 1.0 if raw & 1 else -1.0
    vec[(raw >> 1) % len(vec)] += sign * weight


def main() -> None:
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    print(json.dumps(embed(text)))


if __name__ == "__main__":
    main()
