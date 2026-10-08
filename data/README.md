# Data

> `adversarial_dataset_a/` is the **seeded** 660-card corpus and
> `adversarial_dataset_b/` the **synthetic** 260-card one, matching the paper.

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

## `adversarial_dataset_a/` — 660 seeded cards

600 attacks grafted onto **60 real benign cards** from the wild corpus, shipped
alongside those 60 untouched seeds. Eleven `batch_*.jsonl` files: ten attack
categories at 60 cards each, plus `batch_benign_seed.jsonl`.

Beyond the fields dataset B carries, each record adds:

| field | meaning |
|---|---|
| `in_the_wild` | 1 for a seed card, 0 for a generated attack |
| `parent_card` | the seed's wild-corpus card id (1–310); 0 for a seed |
| `parent_domain` | the seed's domain |
| `patch_op`, `patch_paths` | the operation applied and every JSON path it touched |
| `position_drawn` vs `variation.position` | records when the position fallback fired |
| `structural_score_seed/_patched` | conformance before and after the injection |

`patch_paths` is the verification invariant: an attack card must differ from its
seed at those paths **and nowhere else**. `verify_dataset_b.py` checks this for all
600. Also shipped: `seed_pool.json` (the 60 seeds with a sha256 per card),
`ELIGIBILITY_REPORT.md` (the filter funnel, 123 eligible → 60 sampled) and
`generation_plan.jsonl` (all 600 rows, written before any API call).

Labels: 60 BENIGN / 300 SUSPICIOUS / 300 MALICIOUS. Note the BENIGN cards are
**real observed cards**, not synthetic — they are the controls that give the
evaluation a false-positive rate.

## `adversarial_dataset_b/` — 260 synthetic cards

The **260 synthetic adversarial cards** (10 attack categories), released in full:
10 `batch_<category>.jsonl` files plus the field dictionary
(`METADATA_DICTIONARY.md`). See `adversarial_dataset/README.md`.

## Responsible disclosure

In-the-wild findings were handled through coordinated disclosure with the affected
operators prior to release. The adversarial cards are synthetic attack templates
intended for defensive research and artifact evaluation.
