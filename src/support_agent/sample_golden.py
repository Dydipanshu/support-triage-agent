"""Draw the golden-set candidates with stratification by a cheap weak label.

Random sampling would give ~40% praise and almost no baggage or loyalty cases.
So 2,000 random threads are weakly labelled with gpt-4.1-nano, then 220
candidates are drawn with a floor per weak label. The weak label is thrown
away after sampling; every golden label is assigned by hand afterwards.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from .config import GOLDEN, PROCESSED, BRAND, SEED
from .llm import chat_json
from .taxonomy import INTENTS, taxonomy_prompt_block

WEAK_MODEL = "gpt-4.1-nano"
POOL = 2000
TARGET = 220
FLOOR = 18

SYSTEM = f"""Classify the customer's message into one intent.\n{taxonomy_prompt_block()}\nReturn JSON: {{"intent": "<label>"}}"""


def main() -> None:
    threads = pd.read_parquet(PROCESSED / f"{BRAND}_threads.parquet")
    pool = threads.sample(POOL, random_state=SEED).copy()
    def one(text: str) -> str:
        lab = chat_json(WEAK_MODEL, SYSTEM, text).get("intent", "other")
        return lab if lab in INTENTS else "other"

    with ThreadPoolExecutor(8) as ex:
        labels = list(tqdm(ex.map(one, pool.customer_text.tolist()), total=len(pool), desc="weak labels"))
    pool["weak_intent"] = labels
    print(pool.weak_intent.value_counts())
    GOLDEN.mkdir(parents=True, exist_ok=True)
    pool[["thread_id", "customer_text", "weak_intent"]].to_csv(GOLDEN / "weak_pool.csv", index=False)

    rng_state = SEED
    picks = []
    for lab, grp in pool.groupby("weak_intent"):
        picks.append(grp.sample(min(FLOOR, len(grp)), random_state=rng_state))
    chosen = pd.concat(picks)
    rest = pool.drop(chosen.index)
    extra = rest.sample(TARGET - len(chosen), random_state=rng_state)
    golden = pd.concat([chosen, extra]).sample(frac=1, random_state=rng_state)
    golden = golden[["thread_id", "created_at", "customer_text", "brand_reply", "weak_intent"]]
    GOLDEN.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN / "candidates.csv", index=False)
    print(len(golden), "candidates ->", GOLDEN / "candidates.csv")
    print(golden.weak_intent.value_counts())


if __name__ == "__main__":
    main()
