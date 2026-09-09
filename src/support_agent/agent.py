"""The Delta support agent: intent classification, grounded reply, escalation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import pandas as pd

from .config import AGENT_MODEL
from .llm import chat_json
from .retrieval import Retriever
from .taxonomy import INTENTS, ESCALATE_INTENTS, taxonomy_prompt_block

PII_RE = re.compile(
    r"(?i:confirmation|conf|pnr|record locator|ticket|booking|reservation|account|skymiles)\b.{0,25}?(?<![A-Za-z0-9])#?([A-Z0-9]{6}|\d{9,13})(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])#?(?!DL\d)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}(?![A-Za-z0-9])"
    r"|\b\d{10,13}\b|\b\d{3}-\d{3}-\d{4}\b"
)
COMPENSATION_RE = re.compile(r"\b(refund|compensat\w*|reimburs\w*|charged (my card |me )?twice|double charged|why am i being charged|charging me|pay(ing)? \$?\d+ to change|change fee|voucher|hotel)\b", re.I)
SERIOUS_RE = re.compile(
    r"\b(lawsuit|lawyer|sue\b|sued|attorney|DOT complaint|discriminat\w*|racis\w*|assault\w*|harass\w*|"
    r"injur\w*|hospital\w*|surgery|medical|wheelchair|disabilit\w*|ems\b|paramedic|police|pregnan\w*|"
    r"afraid|unsafe|threat\w*)",
    re.I,
)
HOSTILE_RE = re.compile(r"\b(?:livid|contact me now|fuck\w*|wtf|bullshit)\b", re.I)

CLASSIFY_SYSTEM = f"""You are the intent classifier for Delta Air Lines' Twitter support desk.
Classify the customer's first message into exactly one intent from this list.

{taxonomy_prompt_block()}

Rules:
- Pick the intent that describes what the customer needs Delta to DO, not the topic word.
- If the message is a mixture, pick the most actionable intent.
- If the message is not a support request (a joke, marketing tag, unrelated), use "other".
Return JSON: {{"intent": "<label>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}}"""

REPLY_SYSTEM = """You are a Delta Air Lines Twitter support agent writing the first public reply to a customer.

Voice rules (learned from Delta's real replies):
- Under 280 characters. Warm, brief, first person ("I'd be glad to look into this").
- Apologize once when the customer had a bad experience; never over-apologize.
- Never promise compensation, refunds, waivers, or rebooking outcomes. Never quote policy numbers or fees you are not sure of.
- Never ask the customer to post personal data publicly. When account details are needed, ask them to DM the confirmation number and their full name.
- If the customer left an item on a plane, ask for the arrival airport and flight number and mention the lost-item form.
- For praise, thank them by name if given and keep it to one or two sentences.
- Do not invent facts about the customer's flight.
- Never write the literal token <url>. Historical replies show <url> where a link was; instead name the destination in words, for example "the lost item form on delta.com" or "DM us through our profile link".

You are given historical Delta replies to similar messages. Ground your draft in HOW those replies handled the situation. Do not copy them verbatim.
Return JSON: {"reply": "<text>", "grounding_used": "<one sentence on which historical pattern you followed>"}"""

ESCALATE_SYSTEM = f"""You decide whether a drafted public Twitter reply from Delta support can be posted automatically or must go to a human first.

Definition of "auto": the reply is a low-risk acknowledgement. It thanks, apologises once, or asks the customer to DM their confirmation number. It states no facts about policy, fees, schedules, or the customer's booking, and promises nothing. Delta handles most first replies this way, so "auto" is the default for praise, delays, complaints, bag problems, booking changes, account problems, and support-access complaints.

Escalate ONLY when one of these named triggers applies. Use the trigger name as the first word of the reason.
- needs_factual_answer: the customer asks a general question whose useful answer is a policy, fee, schedule, allowance, or product fact (intent {sorted(ESCALATE_INTENTS)}). A DM request would not answer it.
- compensation_or_refund: the customer disputes a charge, asks for money back, compensation, a hotel, or a promised credit that has not arrived.
- safety_medical_legal_discrimination: a medical situation, accessibility or mobility need, safety incident, legal or regulator threat, discrimination or harassment claim, or a traveler afraid for their safety.
- hostile_or_threatening: abusive language aimed at Delta, a demand like "contact me NOW", or a reputational attack that a bot must not answer.
- pii_posted_publicly: the message contains a confirmation number, ticket number, or account number in public.
- unclear_or_non_english: the message is not in English, is a joke or link with no ask, or is too vague to know what a correct reply would be. A plain "can you help me?" is NOT unclear; a bot can ask for details.

Return JSON: {{"decision": "auto" | "escalate", "reason": "<trigger_name>: <one short sentence>"}}"""


@dataclass
class AgentOutput:
    thread_id: int
    intent: str
    intent_confidence: float
    intent_rationale: str
    reply: str
    grounding_used: str
    decision: str
    reason: str
    rule_triggers: list[str]
    retrieved_ids: list[int]


def rule_triggers(text: str, intent: str, confidence: float) -> list[str]:
    """Deterministic escalation floor. Fires independently of the LLM decision."""
    triggers = []
    if intent in ESCALATE_INTENTS:
        triggers.append("needs_factual_answer")
    if intent == "other":
        triggers.append("unclear_or_non_english")
    if COMPENSATION_RE.search(text):
        triggers.append("compensation_or_refund")
    if SERIOUS_RE.search(text):
        triggers.append("safety_medical_legal_discrimination")
    if HOSTILE_RE.search(text):
        triggers.append("hostile_or_threatening")
    if PII_RE.search(text):
        triggers.append("pii_posted_publicly")
    if confidence < 0.6:
        triggers.append("low_confidence")
    return triggers


def classify(text: str, model: str = AGENT_MODEL) -> dict:
    out = chat_json(model, CLASSIFY_SYSTEM, f"Customer message:\n{text}")
    intent = out.get("intent", "other")
    if intent not in INTENTS:
        intent = "other"
    return {
        "intent": intent,
        "confidence": float(out.get("confidence", 0.5)),
        "rationale": str(out.get("rationale", "")),
    }


def draft_reply(text: str, intent: str, examples: pd.DataFrame, model: str = AGENT_MODEL) -> dict:
    shots = "\n\n".join(
        f"Historical customer message: {r.customer_text}\nDelta's reply: {r.brand_reply}" for r in examples.itertuples()
    )
    user = f"Intent: {intent}\n\nCustomer message:\n{text}\n\nSimilar historical cases:\n{shots}"
    out = chat_json(model, REPLY_SYSTEM, user)
    return {"reply": str(out.get("reply", "")).strip(), "grounding_used": str(out.get("grounding_used", ""))}


def decide(text: str, intent: str, confidence: float, reply: str, triggers: list[str], model: str = AGENT_MODEL) -> dict:
    user = (
        f"Customer message:\n{text}\n\nClassified intent: {intent} (confidence {confidence:.2f})\n"
        f"Rule triggers fired: {triggers or 'none'}\n\nDrafted reply:\n{reply}"
    )
    out = chat_json(model, ESCALATE_SYSTEM, user)
    decision = out.get("decision", "escalate")
    if decision not in {"auto", "escalate"}:
        decision = "escalate"
    reason = str(out.get("reason", ""))
    if triggers and decision == "auto":
        decision = "escalate"
        reason = f"Rule override: {', '.join(triggers)}. Model said: {reason}"
    return {"decision": decision, "reason": reason}


def run_agent(thread_id: int, text: str, retriever: Retriever, k: int = 5, model: str = AGENT_MODEL) -> AgentOutput:
    c = classify(text, model)
    examples = retriever.query(text, k=k)
    d = draft_reply(text, c["intent"], examples, model)
    triggers = rule_triggers(text, c["intent"], c["confidence"])
    e = decide(text, c["intent"], c["confidence"], d["reply"], triggers, model)
    return AgentOutput(
        thread_id=thread_id,
        intent=c["intent"],
        intent_confidence=c["confidence"],
        intent_rationale=c["rationale"],
        reply=d["reply"],
        grounding_used=d["grounding_used"],
        decision=e["decision"],
        reason=e["reason"],
        rule_triggers=triggers,
        retrieved_ids=examples.thread_id.tolist(),
    )


def to_row(o: AgentOutput) -> dict:
    r = asdict(o)
    r["rule_triggers"] = json.dumps(r["rule_triggers"])
    r["retrieved_ids"] = json.dumps(r["retrieved_ids"])
    return r
