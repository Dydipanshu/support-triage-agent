# Delta support agent

An AI support agent for Delta Air Lines' Twitter desk, built and evaluated on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. Given a customer's opening tweet it:

1. classifies the intent into one of nine action-oriented labels,
2. drafts a reply grounded in how Delta historically resolved similar messages, and
3. decides whether the reply can be auto-posted or must go to a human, with a named reason.

The system is small on purpose. Most of the work is in the evaluation: a 220-example hand-labelled golden set, two baselines, a rubric-based LLM judge, and a judge-vs-human agreement study.

Headline results are in [REPORT.md](REPORT.md). The design decisions are in [DECISIONS.md](DECISIONS.md).

## Reproduce the headline results (under 15 minutes)

Requirements: Python 3.11+, [uv](https://github.com/astral-sh/uv), an OpenAI API key.

```bash
git clone <this repo> && cd hiver-support-agent
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]" scipy
echo 'OPENAI_API_KEY=sk-...' > .env

# 1. Raw data (516 MB). Kaggle needs credentials; this Hugging Face mirror does not.
mkdir -p data/raw
curl -L -o data/raw/twcs.csv https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv

# 2. Delta threads (about 50 seconds)
make data

# 3. Evaluate agent and both baselines on the golden set, with the judge
make eval          # writes results/summary.md and results/predictions_*.csv
make agreement     # judge-vs-human agreement, writes results/judge_agreement.json
make label-agreement  # blind second-pass label agreement
make test          # offline unit tests
```

Every OpenAI call is cached under `data/cache/` by content hash. The chat cache (15 MB) is committed, so `make eval` on a fresh clone makes no chat or judge calls. The embedding cache is not committed (737 MB); the first `make eval` embeds the 23k Delta threads once, about two minutes and a few cents. Delete `data/cache/` to regenerate everything from scratch (about 3 USD, 10 to 15 minutes at default rate limits).

`data/processed/Delta_threads.parquet` (11 MB) is also committed, so step 1 and `make data` can be skipped entirely.

`make golden` regenerates the golden-set candidates. Do not run it unless you intend to relabel; it is here for transparency.

Try the agent on a new message:

```bash
python -m src.support_agent.demo "left my wallet on DL1234 from ATL last night, who do I call?"
```

## Layout

```
src/support_agent/
  prepare.py        raw twcs.csv -> data/processed/Delta_threads.parquet
  taxonomy.py       the nine intents, boundaries, escalation reasons
  sample_golden.py  stratified draw of golden candidates
  retrieval.py      embedding index over historical threads
  agent.py          classify -> retrieve -> draft -> decide
  baselines.py      trivial and simple baselines
  judge.py          LLM-as-judge rubric
  evaluate.py       harness: metrics, judge, per-example outputs
  agreement.py      judge vs human agreement
  llm.py            cached OpenAI client
data/golden/        golden.csv, README.md (how it was built), human_reply_scores.csv
results/            summary.md, summary.json, predictions_*.csv, judge_*.csv
tests/              offline unit tests
```

## Models

| role | model | why |
|---|---|---|
| agent (classify, draft, decide) | `gpt-4.1-mini` | cheap, no reasoning tokens, temperature 0 |
| retrieval | `text-embedding-3-small` | 23k threads embed for cents |
| weak labels for stratification | `gpt-4.1-nano` | thrown away after sampling |
| judge | `gpt-5.6-sol`, low reasoning | different family from the generator |

## What was borrowed

- Dataset: Stuart Axelbrooke, "Customer Support on Twitter", Kaggle, via the `SunidhiSriram/twcs` Hugging Face mirror.
- LLM-as-judge with a fixed rubric and a reference reply follows the pattern in Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023).
- Quadratic-weighted Cohen's kappa for ordinal agreement is from scikit-learn.
- Everything else was written for this submission.
