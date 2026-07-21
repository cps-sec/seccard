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
  `signer_identity`, `rekor_integrated_time`, …).

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
