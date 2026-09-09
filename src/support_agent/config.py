from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
GOLDEN = DATA / "golden"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"

BRAND = "Delta"
AGENT_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-5.6-sol"
EMBED_MODEL = "text-embedding-3-small"
SEED = 42
