"""Run the agent on one message from the command line.

Usage: python -m src.support_agent.demo "left my wallet on DL1234 from ATL, help"
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from .agent import run_agent, to_row
from .config import GOLDEN
from .retrieval import Retriever, load_threads


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print(__doc__)
        sys.exit(1)
    threads = load_threads()
    golden_path = GOLDEN / "golden.csv"
    exclude = set(pd.read_csv(golden_path).thread_id) if golden_path.exists() else set()
    retriever = Retriever.build(threads, exclude_ids=exclude)
    out = run_agent(0, text, retriever)
    print(json.dumps(to_row(out), indent=2))


if __name__ == "__main__":
    main()
