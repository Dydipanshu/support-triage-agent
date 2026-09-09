"""Embedding index over historical Delta threads for grounded reply drafting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EMBED_MODEL, PROCESSED, BRAND
from .llm import embed


@dataclass
class Retriever:
    threads: pd.DataFrame
    matrix: np.ndarray

    @classmethod
    def build(cls, threads: pd.DataFrame, exclude_ids: set[int] | None = None) -> "Retriever":
        if exclude_ids:
            threads = threads[~threads.thread_id.isin(exclude_ids)]
        threads = threads.reset_index(drop=True)
        m = embed(threads.customer_text.tolist(), EMBED_MODEL)
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        return cls(threads=threads, matrix=m)

    def query(self, text: str, k: int = 5) -> pd.DataFrame:
        q = embed([text], EMBED_MODEL)[0]
        q = q / np.linalg.norm(q)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        out = self.threads.iloc[top][["thread_id", "customer_text", "brand_reply"]].copy()
        out["similarity"] = sims[top]
        return out


def load_threads() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{BRAND}_threads.parquet")
