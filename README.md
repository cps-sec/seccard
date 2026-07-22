# Discovering the Discoverable: Security Evaluation and Defense of A2A Agent Cards in the Wild

**Research artifact.** This repository accompanies the paper and contains the
datasets and code needed to reproduce its measurement and defense results for
Agent-to-Agent (A2A) agent cards.

> Anonymized for review. No author, affiliation, or institutional information is
> included here.

---

## Overview

The artifact provides:

- **A four-layer pre-invocation defense pipeline** for A2A agent cards —
  structural validation (Layer 1), deterministic NLP rules (Layer 2A), a hardened
  LLM-as-judge (Layer 2B), and a tier-gated trust model — plus a cross-model study.
- **The 310-card in-the-wild corpus** (real cards discovered by an Internet-scale
  crawl), including raw card content and tier/health assessments.
- **The 260-card synthetic adversarial dataset** spanning the ten attack categories.

It reproduces the conformance rubric (Table 1), the layered verdicts and tier
assignments, the LLM-as-judge evaluation (Tables 5–8), and the nine-model
cross-study (Table 5).

## Repository layout

```
.
├── README.md                 # this file
├── LICENSE                   # MIT (code in artifacts/)
├── LICENSE-data.txt          # CC BY 4.0 (data/)
├── requirements.txt          # Python dependencies
├── .env.example              # API-key template (copy to .env)
├── data/
│   ├── README.md
│   ├── in_the_wild_corpus/
│   │   └── in_the_wild_corpus_310.json    # 310 real cards: raw text + tier results
│   └── adversarial_dataset/               # 260 synthetic cards + dictionary
│       ├── README.md
│       ├── METADATA_DICTIONARY.md
│       └── batch_*.jsonl                  # 10 files, 260 cards
└── artifacts/
    ├── README.md                          # per-script command reference
    ├── validate_agent_cards.py            # Layer 1 — conformance rubric
    ├── nlp_features.py                     # Layer 2A — deterministic NLP rules
    ├── inspect_eval.py                     # Layer 2B — LLM-as-judge evaluation
    ├── a2a_card_health_validator.py        # Tier 0/1/2 trust validator
    ├── cross_model_eval.py                 # nine-model cross-study
    └── adversarial_card_generator.py       # regenerate the adversarial dataset
```

## Setup

Requires **Python 3.12**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **API keys** (for the LLM-as-judge and cross-model study): `cp .env.example .env`
  and fill in the keys you need (`ANTHROPIC_API_KEY`, optionally `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`). `.env` is gitignored.
- **NLTK VADER** lexicon downloads automatically on first use of `nlp_features.py`.
- **Optional — SiEBERT** neural sentiment backend: uncomment `transformers` and
  `torch` in `requirements.txt`.
- **Optional — open-weight models** in the cross-model study run via a local
  [Ollama](https://ollama.com) server (no API key).

## Data

- **`data/in_the_wild_corpus/in_the_wild_corpus_310.json`** — the 310 real A2A
  cards. Each record carries the **raw card content** (`card`) *and* its
  **tier/health assessment** (`tier`, `structural_score`, `final_score`,
  `tier_0/1/2_checks`, `blockers`, …). Top level: `{"total": 310, "results": [...]}`.
  Licensed **CC BY 4.0**.
- **`data/adversarial_dataset/`** — the **260 synthetic adversarial cards** (10
  `batch_<category>.jsonl` files) with the full field dictionary
  (`METADATA_DICTIONARY.md`). Licensed **CC BY 4.0**.

## Artifacts

See **`artifacts/README.md`** for the complete command reference and parameter
descriptions. Summary:

| Script | Reproduces | Prerequisite |
|---|---|---|
| `validate_agent_cards.py` | Layer 1 conformance rubric (Table 1) | none |
| `a2a_card_health_validator.py` | Tier 0/1/2 trust validator | none (network only for Rekor) |
| `nlp_features.py` | Layer 2A NLP rules | none (VADER) / HF+torch (SiEBERT) |
| `inspect_eval.py` | Layer 2B LLM-as-judge (Tables 5–8) | `ANTHROPIC_API_KEY` |
| `cross_model_eval.py` | nine-model cross-study (Table 5) | API keys / Ollama |
| `adversarial_card_generator.py` | regenerates the 260-card adversarial dataset | `ANTHROPIC_API_KEY` |

## Reproducing results

Commands are run from the repository root; scripts resolve `data/` relative to
their own location, so the working directory does not matter.

**Offline (no API key), on the shipped 310-card corpus:**
```bash
# Tier 0/1/2 assessment of a single card
python artifacts/a2a_card_health_validator.py --card-file <card.json> --skip-rekor

# Layer 1 — conformance scoring (writes validated.json)
python artifacts/validate_agent_cards.py \
    -i data/in_the_wild_corpus/in_the_wild_corpus_310.json --threshold 3 -o validated.json

# Layer 2A — deterministic NLP features on the validated cards (VADER, offline)
python artifacts/nlp_features.py --input validated.json --sentiment-backend vader
```
`nlp_features.py` consumes the `validate_agent_cards.py` output (its `valid_results`),
so run the two in that order.

**LLM-as-judge (needs `ANTHROPIC_API_KEY`):**
```bash
# Runs the judge on the shipped 310 wild cards
python artifacts/inspect_eval.py wild --limit 5

# Adversarial evaluation on the shipped 260 cards
python artifacts/inspect_eval.py adversarial --dry_run   # validate wiring, no API
python artifacts/inspect_eval.py adversarial --limit 5
```

**Cross-model study (needs API keys and/or Ollama):**
```bash
python artifacts/cross_model_eval.py --models haiku --limit 5
```

**On request:** the per-layer verdict files (`cc_probe_nlp_features.json`,
`cc_probe_intent_v2.json`) plus human labels for exact wild-corpus ground truth
(place them in `data/wild_verdicts/`). Without them, `inspect_eval.py wild` still
runs the judge on every card and reports its verdicts.

## License

- **Code** (`artifacts/`): MIT — see [`LICENSE`](LICENSE).
- **Data** (`data/`): CC BY 4.0 — see [`LICENSE-data.txt`](LICENSE-data.txt).

## Ethics and responsible disclosure

In-the-wild findings were handled through coordinated disclosure with the affected
operators prior to release. The adversarial cards are synthetic attack templates
released for defensive research and artifact evaluation.

## Citation

Citation details are omitted in this anonymized artifact and will be added upon
de-anonymization.
