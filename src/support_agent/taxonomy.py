"""Intent taxonomy for Delta's Twitter support desk.

Derived by hand from a 300-message random sample of first-contact tweets to
Delta (data/processed/taxonomy_read_sample.csv). The labels describe what the
customer needs Delta to DO, because that is what drives the reply and the
escalation decision. Topic words alone (for example "seat") are not intents.
"""
from __future__ import annotations

INTENTS: dict[str, dict[str, str]] = {
    "praise_or_share": {
        "definition": "Compliment, thanks, kudos for staff, or a positive/neutral share (photo, hashtag, milestone) with no problem to solve.",
        "boundary": "A message that thanks Delta but still asks for something is NOT praise; label the ask.",
        "example": "Shoutout to the flight attendant who moved me closer to the front so I made my connection!",
    },
    "flight_disruption": {
        "definition": "The customer is in or facing a delay, cancellation, tarmac wait, gate change, missed or at-risk connection, and wants status, a reason, or help.",
        "boundary": "Venting about a past delay with no live trip and no ask is service_complaint_or_feedback.",
        "example": "DL1496 still on tarmac, will likely miss connection to DL4820. Any other options to get home?",
    },
    "booking_or_seat_change": {
        "definition": "Wants to change, cancel, rebook, stand by, upgrade, pick a seat, fix a name, or disputes a change fee on a specific reservation.",
        "boundary": "Requires touching a reservation. A generic 'what is your change policy' is policy_or_info_question.",
        "example": "I purchased Comfort+ for our trip to LAX. Hubby is 6'5. Possible to get an aisle seat?",
    },
    "baggage_or_lost_item": {
        "definition": "Delayed, lost, damaged, or mis-tagged checked bag, baggage fee dispute on a trip, or an item left on a plane or in the airport.",
        "boundary": "A general question about baggage allowance is policy_or_info_question.",
        "example": "I left my jacket on Flight 1833 yesterday (SLC to CVG). Who can I contact?",
    },
    "loyalty_or_account": {
        "definition": "SkyMiles or Medallion status, missing miles, merging accounts, Amex card benefits, TSA PreCheck or Known Traveler Number on a profile, login problems.",
        "boundary": "About MY account or MY profile. Upgrade requests on a specific flight are booking_or_seat_change. A general question about how the programme works (earning rates, posting times) is policy_or_info_question.",
        "example": "Seems as though miles from my LHR > ATH leg have disappeared from my account. Can you advise?",
    },
    "policy_or_info_question": {
        "definition": "A general question answerable from public policy or schedule facts: allowances, waivers, routes, pets, lounges, arrival times, fees in general.",
        "boundary": "No specific reservation or account needs to be touched to answer it. Includes general loyalty-programme rules and in-flight product facts.",
        "example": "Does Basic Economy include any checked bags?",
    },
    "service_complaint_or_feedback": {
        "definition": "Complaint about staff behaviour, cabin, food, cleanliness, WiFi, boarding process, or a suggestion. The customer is not asking for an action on a reservation.",
        "boundary": "If the complaint carries a live trip problem or a refund/compensation demand, prefer the actionable intent.",
        "example": "Please add baby changing stations to all plane lavatories. Recent flight ATL>DEN had none!",
    },
    "support_access_issue": {
        "definition": "Cannot reach Delta: long phone hold, no callback, app or website error, chat or form broken, asking how to contact someone.",
        "boundary": "If the message states the underlying problem clearly, label that problem instead (for example a seat change).",
        "example": "Is there ever a time to call when the hold time isn't 2 hours?",
    },
    "other": {
        "definition": "Not a support request Delta can act on: unrelated, spam, marketing, non-English with no clear ask, or too vague to classify.",
        "boundary": "Use sparingly. 'I need help' with no detail is other, not support_access_issue.",
        "example": "May have compassion on me!",
    },
}

ESCALATE_INTENTS: set[str] = {"policy_or_info_question"}

ESCALATION_REASONS: list[str] = [
    "none",
    "needs_factual_answer",
    "compensation_or_refund",
    "safety_medical_legal_discrimination",
    "hostile_or_threatening",
    "pii_posted_publicly",
    "unclear_or_non_english",
]


def taxonomy_prompt_block() -> str:
    lines = []
    for name, d in INTENTS.items():
        lines.append(f'- "{name}": {d["definition"]} Boundary: {d["boundary"]} Example: "{d["example"]}"')
    return "\n".join(lines)
