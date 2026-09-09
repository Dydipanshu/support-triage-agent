"""Judge-vs-human agreement on reply quality.

Reads data/golden/human_reply_scores.csv (thread_id, system, human_overall)
and compares against the judge's overall score for the same replies.
Reports Spearman rho, Cohen's kappa (quadratic weights), exact agreement,
and within-one agreement.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .config import GOLDEN, RESULTS


def main() -> None:
    human = pd.read_csv(GOLDEN / "human_reply_scores.csv")
    frames = []
    for system in human.system.unique():
        j = pd.read_csv(RESULTS / f"judge_{system}.csv")[["thread_id", "overall"]].rename(columns={"overall": "judge_overall"})
        h = human[human.system == system]
        frames.append(h.merge(j, on="thread_id", how="inner"))
    df = pd.concat(frames, ignore_index=True)
    rho, p = spearmanr(df.human_overall, df.judge_overall)
    kappa = cohen_kappa_score(df.human_overall, df.judge_overall, weights="quadratic")
    exact = float((df.human_overall == df.judge_overall).mean())
    within1 = float(((df.human_overall - df.judge_overall).abs() <= 1).mean())
    res = {"n": int(len(df)), "spearman_rho": float(rho), "spearman_p": float(p), "kappa_quadratic": float(kappa), "exact_agreement": exact, "within_one": within1,
           "human_mean": float(df.human_overall.mean()), "judge_mean": float(df.judge_overall.mean()),
           "confusion": pd.crosstab(df.human_overall, df.judge_overall).to_dict()}
    (RESULTS / "judge_agreement.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))
    print(pd.crosstab(df.human_overall, df.judge_overall, rownames=["human"], colnames=["judge"]))


if __name__ == "__main__":
    main()
