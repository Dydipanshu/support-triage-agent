# Golden evaluation set

`golden.csv` holds 220 first-contact customer tweets to Delta with hand-assigned labels.

## Columns

| column | meaning |
|---|---|
| `thread_id` | tweet id of the customer's opening message in `twcs.csv` |
| `customer_text` | the message after cleaning (mentions removed, URLs replaced by `<url>`) |
| `brand_reply` | Delta's real first reply, for reference and for the judge. It is not a gold reply; many are mediocre |
| `intent` | one of the nine labels in `src/support_agent/taxonomy.py` |
| `escalate` | `True` when a public reply must be reviewed by a human before posting |
| `escalation_reason` | one of the fixed reason codes, `none` when `escalate` is `False` |

## How the sample was drawn

1. From the 23,608 Delta threads, 2,000 were drawn at random (seed 42).
2. Each was weakly labelled with `gpt-4.1-nano` using the taxonomy prompt. A random draw would be about 40 percent praise and under 4 percent baggage, so the weak label is used only to stratify.
3. 18 candidates were drawn per weak label (162), then 58 more at random, for 220 total.
4. The weak label was discarded. Every intent and escalation label below was assigned by reading the message.

The weak labeller and the agent share a model family, so the stratification is not independent of the system under test. This affects which messages appear, not what their labels are. See the report's "misleading" section.

## Labelling rules

Intent follows the definitions and boundary notes in `taxonomy.py`. The two rules that mattered most:

- Label what the customer needs Delta to do, not the topic. "Why am I charged a bag fee on HJPAFC" is a baggage fee dispute, not a policy question.
- A general question about how something works (allowances, programme rules, schedules) is `policy_or_info_question` even when it mentions the customer's own trip. It becomes `booking_or_seat_change` or `loyalty_or_account` only when a specific reservation or account must be touched.

Escalation follows one question: can a low-risk acknowledgement (thank, apologise once, ask for a DM) be posted without a human reading the message first? It is `True` when any of these hold:

| reason | rule |
|---|---|
| `needs_factual_answer` | the useful answer is a fact about policy, fees, schedules, or products. A DM request does not answer it |
| `compensation_or_refund` | a charge is disputed, or money, a hotel, or a promised credit is requested |
| `safety_medical_legal_discrimination` | medical, accessibility, safety, legal or regulator threat, discrimination, or a traveler afraid for their safety |
| `hostile_or_threatening` | abuse aimed at Delta, "contact me NOW", or a reputational attack |
| `pii_posted_publicly` | a confirmation, ticket, or account number in the public tweet |
| `unclear_or_non_english` | non-English, a joke or link with no ask, or too vague to know what a correct reply would be |

Judgement calls made during labelling, kept for consistency:

- Live disruptions (stuck on tarmac, cancelled today) are `auto`. Delta's real practice is a public apology plus a DM request, and that is exactly what the bot posts. The human takes over in the DM.
- Damaged-bag reports are `auto` unless money is explicitly demanded or a claim is already in dispute.
- "Can you help me?" with no detail is `other` and `auto`. The bot can ask what is wrong. A joke or a bare link is `other` and escalated.
- Profanity in a positive message (one case) is not hostile.

## Who labelled

All 220 labels were assigned in one pass by the author, with an AI coding assistant used to display batches and write the CSV.

`second_pass_input.csv` and `second_pass_labels.csv` are a blind second pass over 80 of the 220 by a fresh AI assistant instance that was given only `taxonomy.py`, the rules above, and the tweets. It never saw `golden.csv`. `make label-agreement` computes the agreement (intent kappa 0.94, escalation kappa 0.79). This measures whether the written rules are followable, not human agreement; see the report.

`human_scoring_blind.csv` and `human_reply_scores.csv` are the 60 replies scored by hand for the judge agreement study, shown shuffled and without system names or judge scores.
