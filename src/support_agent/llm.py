"""Thin OpenAI wrapper with a content-addressed disk cache.

Every call is keyed on (model, messages, params). Reruns of the pipeline hit
the cache and cost nothing, which is what makes the README reproducible in
minutes. Delete data/cache/ to force fresh calls.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .config import CACHE, ROOT

load_dotenv(ROOT / ".env")

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_path(kind: str, key: str) -> Path:
    p = CACHE / kind / key[:2]
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def chat_json(model: str, system: str, user: str, temperature: float | None = 0.0, **params: Any) -> dict[str, Any]:
    """Call a chat model and parse a JSON object from its reply. Cached."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload = {"model": model, "messages": messages, "temperature": temperature, **params}
    path = _cache_path("chat", _key(payload))
    if path.exists():
        return json.loads(path.read_text())["parsed"]
    kwargs: dict[str, Any] = dict(model=model, messages=messages, response_format={"type": "json_object"}, **params)
    if temperature is not None and not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    for attempt in range(8):
        try:
            r = client().chat.completions.create(**kwargs)
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(30, 1.5**attempt + random.random()))
    content = r.choices[0].message.content or "{}"
    parsed = json.loads(content)
    path.write_text(json.dumps({"parsed": parsed, "usage": r.usage.model_dump(), "raw": content}))
    return parsed


def embed(texts: list[str], model: str, batch: int = 512) -> np.ndarray:
    """Embed a list of strings. Cached per text so partial reruns are cheap."""
    out: list[np.ndarray | None] = [None] * len(texts)
    todo: list[int] = []
    for i, t in enumerate(texts):
        path = _cache_path("embed", _key({"model": model, "text": t}))
        if path.exists():
            out[i] = np.array(json.loads(path.read_text()), dtype=np.float32)
        else:
            todo.append(i)
    for s in range(0, len(todo), batch):
        idx = todo[s : s + batch]
        r = client().embeddings.create(model=model, input=[texts[i] for i in idx])
        for i, d in zip(idx, r.data):
            vec = np.array(d.embedding, dtype=np.float32)
            out[i] = vec
            _cache_path("embed", _key({"model": model, "text": texts[i]})).write_text(json.dumps(d.embedding))
    return np.vstack(out)  # type: ignore[arg-type]
