"""Two baselines the agent must beat.

trivial: majority intent, always escalate, one canned reply.
simple:  TF-IDF + logistic regression intent (trained on weak labels from the
         sampling pool, never on golden), nearest-neighbour historical reply
         by TF-IDF cosine, keyword rules for escalation.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .taxonomy import ESCALATE_INTENTS

CANNED = "We're sorry to hear this. Please follow/DM your confirmation number so we can look into it."

ESCALATE_KW = re.compile(
    r"\b(refund|compensat|voucher|reimburs|lawyer|lawsuit|sue\b|discriminat|disabil|wheelchair|medical|hospital|"
    r"police|assault|harass|racis|policy|allowed|how much|how many|can i bring|do you (offer|allow|have))",
    re.I,
)


class TrivialBaseline:
    def __init__(self, majority_intent: str):
        self.majority_intent = majority_intent

    def predict(self, texts: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "intent": [self.majority_intent] * len(texts),
                "reply": [CANNED] * len(texts),
                "decision": ["escalate"] * len(texts),
                "reason": ["trivial baseline always escalates"] * len(texts),
            }
        )


class SimpleBaseline:
    def __init__(self, train_texts: list[str], train_labels: list[str], history: pd.DataFrame):
        self.clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"),
        )
        self.clf.fit(train_texts, train_labels)
        self.history = history.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.hist_matrix = self.vec.fit_transform(self.history.customer_text)

    def predict(self, texts: list[str]) -> pd.DataFrame:
        intents = self.clf.predict(texts)
        q = self.vec.transform(texts)
        sims = (q @ self.hist_matrix.T).toarray()
        nn = sims.argmax(axis=1)
        replies = self.history.brand_reply.values[nn]
        decisions, reasons = [], []
        for t, it in zip(texts, intents):
            kw = ESCALATE_KW.search(t)
            if it in ESCALATE_INTENTS:
                decisions.append("escalate"); reasons.append(f"intent {it} needs a human")
            elif kw:
                decisions.append("escalate"); reasons.append(f"keyword '{kw.group(0)}'")
            else:
                decisions.append("auto"); reasons.append("no risk keyword")
        return pd.DataFrame({"intent": intents, "reply": replies, "decision": decisions, "reason": reasons, "nn_similarity": sims.max(axis=1)})
