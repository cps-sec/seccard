# Data

This directory holds the datasets for the study. All data is released under
**CC BY 4.0**.

## `in_the_wild_corpus/`

`in_the_wild_corpus_310.json` — the **310 real A2A agent cards** discovered by an
Internet-scale crawl (36.1M registered domains). Each record contains **both**:

- the **raw card content** (`card`: `name`, `description`, `url`, `provider`,
  `capabilities`, `defaultInputModes`/`defaultOutputModes`, `skills`, …), and
- the **tier / health assessment** (`tier`, `structural_score`, `final_score`,
  `tier_0/1/2_checks`, `blockers`, `advisories`, `signatures_present`,
  `signer_identity`, `rekor_integrated_time`, …), and
- the **pipeline verdicts and trust label**: `l2a_verdict`, `l2b_v1_verdict`,
  `l2b_v2_verdict` (each BENIGN/SUSPICIOUS/MALICIOUS), `ensemble_verdict` (2-of-3
  majority), `reached_tier0` (bool, `structural_score >= 8`), and `trust`
  (`"Trusted"`/`"Untrusted"`).

**Trust policy:** a card is `Trusted` iff `reached_tier0` **and**
`ensemble_verdict == "BENIGN"` — the only cards eligible for automatic orchestrator
consumption. Reproduce the labels and summary with
`artifacts/trust_labeling.py` (see `trust_summary.md`).

Top level: `{"total": 310, "results": [ ... ]}`. This is the ground-truth corpus for
the wild-corpus evaluation.

## `adversarial_dataset/`

The **260 synthetic adversarial cards** (10 attack categories), released in full:
10 `batch_<category>.jsonl` files plus the field dictionary
(`METADATA_DICTIONARY.md`). See `adversarial_dataset/README.md`.

## Responsible disclosure

In-the-wild findings were handled through coordinated disclosure with the affected
operators prior to release. The adversarial cards are synthetic attack templates
intended for defensive research and artifact evaluation.
