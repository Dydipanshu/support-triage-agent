"""Inter-annotator agreement between the golden labels and a blind second pass.

Reads data/golden/second_pass_labels.csv and reports Cohen's kappa for intent
and for the escalation decision, plus the disagreements for inspection.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import GOLDEN, RESULTS


def main() -> None:
    gold = pd.read_csv(GOLDEN / "golden.csv")
    second = pd.read_csv(GOLDEN / "second_pass_labels.csv")
    df = gold.merge(second, on="thread_id", suffixes=("_a", "_b"))
    for col in ("escalate_a", "escalate_b"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    res = {
        "n": int(len(df)),
        "intent_kappa": float(cohen_kappa_score(df.intent_a, df.intent_b)),
        "intent_agreement": float((df.intent_a == df.intent_b).mean()),
        "escalate_kappa": float(cohen_kappa_score(df.escalate_a, df.escalate_b)),
        "escalate_agreement": float((df.escalate_a == df.escalate_b).mean()),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "label_agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    dis = df[df.intent_a != df.intent_b][["thread_id", "customer_text", "intent_a", "intent_b"]]
    dis.to_csv(RESULTS / "label_disagreements.csv", index=False)
    pd.set_option("display.max_colwidth", 120)
    print(dis.to_string(index=False))


if __name__ == "__main__":
    main()
