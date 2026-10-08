# We Have All the Cards: Securing Agent Cards at the Operational Trust Boundary of the Agent2Agent Protocol

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
- **Two adversarial datasets**, both spanning the same ten attack categories:
  - **Dataset A — 660 cards, seeded.** 600 attacks grafted onto 60 *real* benign
    cards drawn from the wild corpus, shipped alongside those 60 untouched seeds.
    Because a variant differs from its seed only at the injected field, A is a
    controlled comparison, and its benign seeds give the evaluation a
    **false-positive rate**.
  - **Dataset B — 260 cards, synthetic.** Every card invented by the generator
    around one of four personas. No benign cards, so no false-positive measure.
- **Nine-model cross-study results for both datasets** under the same judge
  prompt, with per-card predictions and provenance records.

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
│   │   └── in_the_wild_corpus_310.json   # 310 real cards: raw text + tier results
│   ├── adversarial_dataset_a/            # 660 seeded cards (600 attacks + 60 seeds)
│   │   ├── README.md
│   │   ├── ELIGIBILITY_REPORT.md         # seed-pool filter funnel
│   │   ├── seed_pool.json                # the 60 seeds + sha256 of each card
│   │   ├── generation_plan.jsonl         # 600 rows, written before any API call
│   │   └── batch_*.jsonl                 # 11 files (10 categories + benign_seed)
│   └── adversarial_dataset_b/            # 260 synthetic cards + dictionary
│       ├── README.md
│       ├── METADATA_DICTIONARY.md
│       ├── GENERATION_PROMPT.md
│       └── batch_*.jsonl                 # 10 files, 260 cards
└── artifacts/
    ├── README.md                         # per-script command reference
    ├── validate_agent_cards.py           # Layer 1 — conformance rubric
    ├── nlp_features.py                   # Layer 2A — deterministic NLP rules
    ├── evaluate_card_intent.py           # Layer 2B prompts (v1/v2/v3, longform)
    ├── inspect_eval.py                   # Layer 2B — LLM-as-judge evaluation
    ├── a2a_card_health_validator.py      # Tier 0/1/2 trust validator
    ├── trust_labeling.py                 # Trusted/Untrusted labeling
    ├── cross_model_eval.py               # nine-model study, dataset B
    ├── cross_model_eval_b.py             # nine-model study, either dataset
    ├── seed_selection_b.py               # dataset A — seed pool selection
    ├── patch_ops_b.py                    # dataset A — patch vocabulary
    ├── adversarial_card_generator_b.py   # dataset A — generator
    ├── verify_dataset_b.py               # dataset A — verification suite
    ├── adversarial_card_generator.py     # dataset B — generator
    ├── make_dataset_comparison.py        # A vs B composition table
    └── cross_model_results/              # results, provenance, LaTeX tables
```

> **Naming.** `adversarial_dataset_a/` is the **seeded** 660-card corpus and
> `adversarial_dataset_b/` the **synthetic** 260-card one, matching the paper.
> Script filenames carrying a `_b` suffix (`cross_model_eval_b.py`,
> `verify_dataset_b.py`, …) are an artifact of development order and do **not**
> indicate which dataset they operate on — `cross_model_eval_b.py --dataset A|B`
> selects the corpus, using the paper's letters.

## Setup

Requires **Python 3.12**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **API keys** (for the LLM-as-judge and cross-model study): `cp .env.example .env`
  and fill in the keys you need (`ANTHROPIC_API_KEY`, optionally `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`). `.env` is gitignored.
- **`openai` must be `<3` if you install it at all.** `inspect_ai` 0.3.220 hangs
  forever against `openai` 3.x — silently, with no error and no request issued —
  and that provider path serves both the OpenAI models *and* the local `ollama/*`
  ones, so a 3.x install stalls most of the cross-model grid. It is an optional
  dependency; `requirements.txt` carries the bound (`openai>=2.26,<3`) in its
  optional block, commented out. Uncomment rather than `pip install openai`.
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
- **`data/adversarial_dataset_a/`** — the **660-card seeded** dataset: 600 attacks
  (60 per category) plus the 60 benign seed cards they were grafted onto. Records
  carry `in_the_wild` (1 = seed) and `parent_card` (the seed's wild card id), so
  every attack is traceable to its backbone. See its `README.md` for the schema,
  the six-operation patch protocol, and the known caveats.
- **`data/adversarial_dataset_b/`** — the **260 synthetic** adversarial cards (10
  `batch_<category>.jsonl` files) with the full field dictionary
  (`METADATA_DICTIONARY.md`) and the generator prompt (`GENERATION_PROMPT.md`).

All data is licensed **CC BY 4.0**.

## Artifacts

See **`artifacts/README.md`** for the complete command reference. Summary:

| Script | Reproduces | Prerequisite |
|---|---|---|
| `validate_agent_cards.py` | Layer 1 conformance rubric | none |
| `a2a_card_health_validator.py` | Tier 0/1/2 trust validator | none (network only for Rekor) |
| `nlp_features.py` | Layer 2A NLP rules | none (VADER) / HF+torch (SiEBERT) |
| `inspect_eval.py` | Layer 2B LLM-as-judge | `ANTHROPIC_API_KEY` |
| `cross_model_eval_b.py` | nine-model cross-study, **either dataset** | API keys / Ollama |
| `cross_model_eval.py` | the original dataset-B-only cross-study | API keys / Ollama |
| `seed_selection_b.py` | dataset A seed pool (needs the annotator labels, see below) | none |
| `adversarial_card_generator_b.py` | regenerates dataset A | `ANTHROPIC_API_KEY` |
| `verify_dataset_b.py` | dataset A acceptance checks | none |
| `adversarial_card_generator.py` | regenerates dataset B | `ANTHROPIC_API_KEY` |
| `make_dataset_comparison.py` | the A vs B composition table | none |

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

**Offline — verify the shipped datasets (no API key):**
```bash
# Confirm which corpus a driver will read, and its sha256 fingerprint
python artifacts/cross_model_eval_b.py --dataset A --dataset-check
python artifacts/cross_model_eval_b.py --dataset B --dataset-check

# Full acceptance suite on dataset A: schema, id uniqueness, seed fidelity
# (each attack differs from its seed only at the declared patch paths),
# per-category properties, balance, structural scores, judge visibility
python artifacts/verify_dataset_b.py \
    --dataset data/adversarial_dataset_a \
    --seeds   data/adversarial_dataset_a/seed_pool.json

# Composition table comparing the two datasets
python artifacts/make_dataset_comparison.py

# Rebuild both LaTeX tables from the shipped results — no API calls
python artifacts/cross_model_eval_b.py --dataset A --latex-only
python artifacts/cross_model_eval_b.py --dataset B --latex-only
```

**LLM-as-judge (needs `ANTHROPIC_API_KEY`):**
```bash
python artifacts/inspect_eval.py wild --limit 5
python artifacts/inspect_eval.py adversarial --dry_run   # validate wiring, no API
python artifacts/inspect_eval.py adversarial --limit 5
```

**Cross-model study (needs API keys and/or Ollama):**
```bash
# Stratified smoke test: 2 cards per category, one local model, no API cost
python artifacts/cross_model_eval_b.py --dataset A --models llama31_8b \
       --sample-per-category 2

# Full grid for either dataset
python artifacts/cross_model_eval_b.py --dataset A
python artifacts/cross_model_eval_b.py --dataset B
```

Every run records a **sha256 fingerprint of the dataset it read** in its results
JSON, and refuses to start if a previously completed model used a different one.
`--recompute` rebuilds all metrics from saved Inspect logs and diffs them against
the stored results — use it to confirm a change is behaviour-preserving.

## Results

`artifacts/cross_model_results/` ships, for both datasets under the same judge
prompt (`v2-longform`):

| file | contents |
|---|---|
| `cross_model_results_a.json` | per-model metrics **and per-card predictions**, dataset A |
| `cross_model_results_b.json` | the same for dataset B |
| `provenance_a.json`, `provenance_b.json` | dataset fingerprint, system-prompt hash, models verified |
| `cross_model_table_a.tex`, `cross_model_table_b.tex` | the LaTeX tables |
| `dataset_comparison_table.tex` | A vs B composition |

Raw Inspect `.eval` logs (~75 MB) are **not** included; the per-card predictions in
the results JSON carry the information needed to audit any individual verdict.

## Known limitations

- **`seed_selection_b.py` cannot run from this artifact as shipped.** It needs the
  human annotator labels for the 310-card corpus to choose BENIGN seeds, and those
  are not included here. The resulting seed pool is shipped instead
  (`data/adversarial_dataset_a/seed_pool.json`, with a sha256 per seed card), so
  the selection is auditable even though it is not re-runnable.
- **Dataset A's 660 records are not independent.** Eleven share each seed
  backbone, so confidence intervals and significance tests must cluster on
  `parent_card`; the effective n per category is 60 distinct cards. Dataset B's
  260 are approximately independent.
- **Overall-recall rows are not directly comparable between the datasets** even
  under the same prompt: A weights categories uniformly (60 each) while B weights
  them 35/25/15. Compare per-category rows, or re-weight.
- 26 of dataset A's `context_pollution` cards exceed the judge's 6,144-character
  window by design; that truncation is the attack, not a defect.

## License

- **Code** (`artifacts/`): MIT — see [`LICENSE`](LICENSE).
- **Data** (`data/`): CC BY 4.0 — see [`LICENSE-data.txt`](LICENSE-data.txt).

## Ethics and responsible disclosure

In-the-wild findings were handled through coordinated disclosure with the affected
operators prior to release.

The two adversarial datasets differ in how they were produced, and the distinction
matters. **Dataset B is wholly synthetic** — invented personas, invented
organisations — and is released as attack templates for defensive research.
**Dataset A is seeded from real agent cards** observed in the wild and retains
each seed's name, URL and provider. Its adversarial variants are therefore
fabricated malicious cards bearing the identity of real services.

Three points on that choice. The seed cards are already-public documents, served
without authentication from the RFC 8615 well-known path and republished in this
artifact's wild corpus. Nothing in either dataset was ever deployed, hosted, or
submitted to any agent registry; the cards exist only as evaluation inputs. And
retaining real identity is what makes dataset A a controlled comparison — the
attack variant and its benign seed differ only at the injected field, which is
precisely what lets a verdict change be attributed to the attack rather than to
the card's provenance or style.

Researchers reusing dataset A should treat its cards as what they are: synthetic
attacks on real names, not evidence of wrongdoing by the named services. None of
the 60 seed cards was malicious — every one is a human-labelled BENIGN card.

## Citation

Citation details are omitted in this anonymized artifact and will be added upon
de-anonymization.
