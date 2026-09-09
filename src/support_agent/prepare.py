"""Extract one brand's first-contact support threads from the raw twcs.csv.

Output: data/processed/<brand>_threads.parquet with one row per thread that
starts with a customer tweet and receives at least one brand reply.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/twcs.csv")
OUT_DIR = Path("data/processed")

URL_RE = re.compile(r"https?://\S+")
SIG_RE = re.compile(r"\*[A-Z]{2,4}\b")
PART_RE = re.compile(r"\b(\d)\s*/\s*\d\b")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_brand_reply(text: str) -> str:
    text = SIG_RE.sub("", text)
    text = PART_RE.sub("", text)
    text = text.replace("...", " ")
    return clean_text(text)


def clean_brand_reply(text: str) -> str:
    text = SIG_RE.sub("", text)
    text = PART_RE.sub("", text)
    text = text.replace("...", " ")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = URL_RE.sub("<url>", text)
    text = MENTION_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return WS_RE.sub(" ", text).strip()


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(
        RAW,
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "inbound": "str",
            "created_at": "str",
            "text": "str",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "float64",
        },
    )
    df["inbound"] = df["inbound"] == "True"
    return df


def _part(text: str) -> int:
    m = PART_RE.search(text)
    return int(m.group(1)) if m else 0


def build_threads(df: pd.DataFrame, brand: str, max_turns: int = 8) -> pd.DataFrame:
    brand_ids = set(df.tweet_id[df.author_id == brand].tolist())
    roots = df[df.inbound & df.in_response_to_tweet_id.isna() & df.response_tweet_id.notna()]
    first_reply = roots.response_tweet_id.str.split(",").str[0].astype("int64")
    roots = roots[first_reply.isin(brand_ids)]

    by_id = dict(
        zip(
            df.tweet_id.tolist(),
            zip(df.author_id.tolist(), df.inbound.tolist(), df.created_at.tolist(), df.text.tolist(), df.response_tweet_id.tolist()),
        )
    )

    rows = []
    for r in roots.itertuples(index=False):
        direct_ids = [int(x) for x in r.response_tweet_id.split(",")]
        direct = [(i, by_id[i]) for i in direct_ids if i in by_id and by_id[i][0] == brand]
        if not direct:
            continue
        direct.sort(key=lambda it: it[0])
        reply_raw = " ".join(t[3] for _, t in direct)
        turns = [("customer", r.text, r.created_at), ("brand", reply_raw, direct[0][1][2])]
        seen = {r.tweet_id, *(i for i, _ in direct)}
        nxt = None
        for _, t in reversed(direct):
            if isinstance(t[4], str):
                nxt = int(t[4].split(",")[0])
                break
        while nxt is not None and nxt in by_id and len(turns) < max_turns and nxt not in seen:
            seen.add(nxt)
            t = by_id[nxt]
            turns.append(("customer" if t[1] else "brand", t[3], t[2]))
            nxt = int(t[4].split(",")[0]) if isinstance(t[4], str) else None
        rows.append(
            {
                "thread_id": int(r.tweet_id),
                "created_at": r.created_at,
                "customer_text_raw": r.text,
                "customer_text": clean_text(r.text),
                "brand_reply_raw": reply_raw,
                "brand_reply": clean_brand_reply(reply_raw),
                "n_turns": len(turns),
                "thread": [
                    {"role": a, "text": clean_brand_reply(b) if a == "brand" else clean_text(b), "created_at": c}
                    for a, b, c in turns
                ],
            }
        )
    out = pd.DataFrame(rows)
    out = out[out.customer_text.str.len() >= 15]
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="Delta")
    args = ap.parse_args()
    df = load_raw()
    threads = build_threads(df, args.brand)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.brand}_threads.parquet"
    threads.to_parquet(out, index=False)
    print(f"{args.brand}: {len(threads)} threads -> {out}")


if __name__ == "__main__":
    main()
