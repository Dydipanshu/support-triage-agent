"""LLM-as-judge for reply quality, scored against a fixed rubric.

The judge model is a different family from the generator to reduce
self-preference. Every score is cached, so re-running the harness is free.
"""
from __future__ import annotations

from .config import JUDGE_MODEL
from .llm import chat_json

RUBRIC = """Score a Delta Twitter support reply on four criteria, each 1 (bad), 2 (acceptable), 3 (good).

1. grounded: Does the reply follow how Delta actually handles this kind of message (acknowledge, apologise once, ask for a DM with confirmation number, point to the right form or desk)? 3 = matches Delta's real handling pattern; 1 = invents a process or promises something Delta agents never promise publicly.
2. correct_next_step: Does it move the customer toward resolution of THEIR specific problem? 3 = the right next step for this message; 2 = generic but not wrong; 1 = wrong or missing next step, or addresses a different problem.
3. safe: Does it avoid hallucinated facts, fees, policies, flight details, compensation promises, and asking for private data in public? 3 = nothing risky; 2 = a minor unverifiable claim; 1 = a clear fabrication or a promise.
4. tone: Warm, brief, on-brand for Delta on Twitter, under 280 characters, no over-apologising. 3 = good; 2 = slightly off; 1 = rude, robotic, or far too long.

Also give overall: 1 = would embarrass Delta if posted, 2 = fine to post with light edits, 3 = post as is.

Return JSON: {"grounded": n, "correct_next_step": n, "safe": n, "tone": n, "overall": n, "critique": "<one sentence naming the biggest weakness>"}"""


def judge_reply(customer_text: str, intent: str, reply: str, historical_reply: str, model: str = JUDGE_MODEL) -> dict:
    user = (
        f"Customer message:\n{customer_text}\n\nIntent: {intent}\n\n"
        f"Reply to score:\n{reply}\n\n"
        f"For reference only, Delta's actual reply to this exact message (may itself be imperfect):\n{historical_reply}"
    )
    out = chat_json(model, RUBRIC, user, temperature=None, reasoning_effort="low", max_completion_tokens=400)
    keys = ["grounded", "correct_next_step", "safe", "tone", "overall"]
    res = {k: int(out.get(k, 1)) for k in keys}
    res["critique"] = str(out.get("critique", ""))
    return res
