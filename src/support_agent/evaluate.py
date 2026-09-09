"""Evaluation harness: runs the agent and both baselines on the golden set and
writes metrics, judge scores, and per-example outputs to results/.

Usage: python -m src.support_agent.evaluate [--limit N] [--no-judge]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .agent import run_agent, to_row
from .baselines import SimpleBaseline, TrivialBaseline
from .config import GOLDEN, PROCESSED, RESULTS, BRAND, SEED
from .judge import judge_reply
from .retrieval import Retriever
from .taxonomy import INTENTS

LABELS = list(INTENTS)


def load_golden() -> pd.DataFrame:
    g = pd.read_csv(GOLDEN / "golden.csv")
    g["escalate"] = g["escalate"].astype(str).str.lower().isin(["true", "1", "yes"])
    return g


def intent_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def escalation_metrics(y_true: pd.Series, decision: pd.Series) -> dict:
    y_pred = decision == "escalate"
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auto_rate = float((~y_pred).mean())
    unsafe_auto = int(((~y_pred) & y_true).sum())
    return {
        "escalate_precision": p,
        "escalate_recall": r,
        "escalate_f1": f,
        "auto_rate": auto_rate,
        "unsafe_auto_count": unsafe_auto,
        "unsafe_auto_rate": unsafe_auto / max(1, int(y_true.sum())),
    }


def run_judge(golden: pd.DataFrame, preds: pd.DataFrame, name: str, workers: int = 8) -> pd.DataFrame:
    def one(i: int) -> dict:
        g = golden.iloc[i]
        p = preds.iloc[i]
        j = judge_reply(g.customer_text, g.intent, p.reply, g.brand_reply)
        j["thread_id"] = int(g.thread_id)
        return j

    with ThreadPoolExecutor(workers) as ex:
        rows = list(tqdm(ex.map(one, range(len(golden))), total=len(golden), desc=f"judge {name}"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    golden = load_golden()
    if args.limit:
        golden = golden.head(args.limit)
    threads = pd.read_parquet(PROCESSED / f"{BRAND}_threads.parquet")
    golden_ids = set(golden.thread_id)

    history = threads[~threads.thread_id.isin(golden_ids)]
    retriever = Retriever.build(history)

    pool = pd.read_csv(GOLDEN / "weak_pool.csv")
    pool = pool[~pool.thread_id.isin(golden_ids)]
    majority = golden.intent.value_counts().idxmax()
    trivial = TrivialBaseline(majority)
    simple = SimpleBaseline(pool.customer_text.tolist(), pool.weak_intent.tolist(), history)

    texts = golden.customer_text.tolist()
    preds = {"trivial": trivial.predict(texts), "simple": simple.predict(texts)}

    def run_one(i: int) -> dict:
        return to_row(run_agent(int(golden.thread_id.iloc[i]), texts[i], retriever))

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(run_one, range(len(golden))), total=len(golden), desc="agent"))
    preds["agent"] = pd.DataFrame(rows)

    summary: dict = {"n": len(golden), "systems": {}}
    for name, p in preds.items():
        m = {"intent": intent_metrics(golden.intent, p.intent), "escalation": escalation_metrics(golden.escalate, p.decision)}
        if not args.no_judge:
            j = run_judge(golden, p, name, args.workers)
            j.to_csv(RESULTS / f"judge_{name}.csv", index=False)
            m["judge"] = {k: float(j[k].mean()) for k in ["grounded", "correct_next_step", "safe", "tone", "overall"]}
            m["judge"]["share_overall_3"] = float((j.overall == 3).mean())
            m["judge"]["share_overall_1"] = float((j.overall == 1).mean())
        summary["systems"][name] = m
        out = golden[["thread_id", "customer_text", "intent", "escalate", "escalation_reason", "brand_reply"]].copy()
        out.columns = ["thread_id", "customer_text", "gold_intent", "gold_escalate", "gold_reason", "historical_reply"]
        out = pd.concat([out.reset_index(drop=True), p.add_prefix("pred_").reset_index(drop=True)], axis=1)
        if not args.no_judge:
            out = pd.concat([out, j.drop(columns=["thread_id"]).add_prefix("judge_")], axis=1)
        out.to_csv(RESULTS / f"predictions_{name}.csv", index=False)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)
    print(open(RESULTS / "summary.md").read())


def write_markdown(summary: dict) -> None:
    lines = [f"# Results on golden set (n={summary['n']})", ""]
    lines.append("| system | intent acc | macro F1 | esc. recall | esc. precision | auto rate | unsafe auto | judge overall | share post-as-is |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in summary["systems"].items():
        i, e = m["intent"], m["escalation"]
        j = m.get("judge", {})
        lines.append(
            f"| {name} | {i['accuracy']:.3f} | {i['macro_f1']:.3f} | {e['escalate_recall']:.3f} | {e['escalate_precision']:.3f} | "
            f"{e['auto_rate']:.3f} | {e['unsafe_auto_count']} | {j.get('overall', float('nan')):.2f} | {j.get('share_overall_3', float('nan')):.3f} |"
        )
    lines.append("")
    for name, m in summary["systems"].items():
        lines.append(f"## {name}: per-intent F1")
        lines.append("| intent | precision | recall | f1 | support |")
        lines.append("|---|---|---|---|---|")
        for lab in LABELS:
            r = m["intent"]["report"].get(lab, {})
            lines.append(f"| {lab} | {r.get('precision', 0):.2f} | {r.get('recall', 0):.2f} | {r.get('f1-score', 0):.2f} | {int(r.get('support', 0))} |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
