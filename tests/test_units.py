"""Fast, offline tests for the deterministic parts of the pipeline."""
import pandas as pd

from src.support_agent.agent import rule_triggers
from src.support_agent.baselines import SimpleBaseline, TrivialBaseline
from src.support_agent.prepare import clean_brand_reply, clean_text
from src.support_agent.taxonomy import INTENTS, ESCALATE_INTENTS, taxonomy_prompt_block


def test_clean_text_strips_mentions_urls_and_entities():
    assert clean_text("@Delta help &amp; thanks https://t.co/x") == "help & thanks <url>"


def test_clean_brand_reply_strips_signature_and_part_markers():
    assert clean_brand_reply("Hi Sam, please DM us. *TBT 1/2") == "Hi Sam, please DM us."


def test_taxonomy_is_well_formed():
    assert "other" in INTENTS
    assert ESCALATE_INTENTS <= set(INTENTS)
    block = taxonomy_prompt_block()
    for name in INTENTS:
        assert f'"{name}"' in block


def test_rule_triggers_fire_on_risk_signals():
    assert "safety_medical_legal_discrimination" in rule_triggers("I will sue you, worst airline", "service_complaint_or_feedback", 0.9)
    assert "pii_posted_publicly" in rule_triggers("my confirmation is HJPAFC", "booking_or_seat_change", 0.9)
    assert "pii_posted_publicly" not in rule_triggers("flight DL1234 to ATL delayed", "flight_disruption", 0.9)
    assert "low_confidence" in rule_triggers("hmm", "other", 0.3)
    assert "needs_factual_answer" in rule_triggers("do you allow candles?", "policy_or_info_question", 0.9)
    assert "compensation_or_refund" in rule_triggers("you charged my card twice", "booking_or_seat_change", 0.9)
    assert "unclear_or_non_english" in rule_triggers("May have compassion on me!", "other", 0.9)
    assert rule_triggers("Thanks for a great flight!", "praise_or_share", 0.95) == []


def test_trivial_baseline_shape():
    p = TrivialBaseline("praise_or_share").predict(["a", "b"])
    assert list(p.intent) == ["praise_or_share"] * 2
    assert set(p.decision) == {"escalate"}


def test_simple_baseline_trains_and_predicts():
    texts = ["thanks great flight", "lost my bag", "flight delayed again", "do you allow pets", "miles missing from account"] * 4
    labels = ["praise_or_share", "baggage_or_lost_item", "flight_disruption", "policy_or_info_question", "loyalty_or_account"] * 4
    hist = pd.DataFrame({"customer_text": texts, "brand_reply": ["r"] * len(texts)})
    b = SimpleBaseline(texts, labels, hist)
    p = b.predict(["my bag is lost", "do you allow dogs"])
    assert p.intent.iloc[0] == "baggage_or_lost_item"
    assert p.decision.iloc[1] == "escalate"
