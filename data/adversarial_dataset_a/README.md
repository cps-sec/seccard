# Adversarial Dataset B — seeded from real in-the-wild agent cards

Dataset A (260 cards) is fully synthetic: each card was invented by the generator
around one of eight personas. Dataset B instead grafts the same ten-category threat
taxonomy onto **real BENIGN agent cards** observed in the wild, and ships each
untouched seed alongside its ten adversarial variants.

Because a variant differs from its seed **only** at the injected path, B is a
controlled comparison: a verdict change is attributable to the attack, not to the
card's provenance or writing style. B also carries BENIGN controls, which dataset A
structurally cannot — giving the adversarial-style evaluation a false-positive rate.

## Composition

| | |
|---|---|
| Seeds (`in_the_wild=1`) | 60 real BENIGN cards, verbatim |
| Generated (`in_the_wild=0`) | 600 = 60 seeds × 10 taxonomy categories |
| **Total** | **660** records |
| Labels | 60 BENIGN / 300 MALICIOUS / 300 SUSPICIOUS |
| Generator model | `claude-sonnet-4-6` (same as dataset A) |

## Provenance

Seeds are drawn from `in_the_wild_corpus_310.json`, restricted to cards whose
**human** label is BENIGN (`wild_cards_human_labels.json`), then filtered so that
every seed can host every attack category:

| filter | why |
|---|---|
| raw card JSON ≤ 6000 chars | keeps the judge-visible serialization clear of the 6144 cap |
| 2 ≤ skills ≤ 8 | ≥2 so `skills[1].*` positions exist; **≤8 because the judge only ever sees `skills[:10]`** — a skill inserted past that index is invisible, so the card would be a guaranteed miss unrelated to its attack |
| first 2 skills well-formed | `skill_deception` targets them |
| structural_score ≥ 8 | both the stored corpus value and the live `validate_card` score; matches dataset A's 8–9 profile so Layer 1 is blind to both datasets |
| dedupe by card content | the corpus contains near-duplicate cards across domains |

Pool: **123 eligible → 60 sampled (49%)**, `random.Random(20260922).sample`.
Full funnel in `seeds/eligibility_report_b.md`; the chosen seeds with a sha256 of each
card are in `seeds/seed_pool_b.json`.

## Record schema

Dataset A's ten keys, in order, then B's additions — so any A-shaped consumer works
unchanged:

```
id, category, expected_verdict, group,
variation{persona, position, obfuscation, sophistication},
card, injection_location, payload_description, generated_at, generator_model,
in_the_wild, parent_card, parent_domain, position_drawn,
patch_op, patch_paths, patch_evidence,
structural_score_seed, structural_score_patched
```

- `in_the_wild` — 1 for a seed, 0 for a generated card
- `parent_card` — the seed's in-the-wild `card_id` (1–310); **0** for a seed
- `patch_paths` — every JSON path the injection touched (the verification invariant)
- `position_drawn` vs `variation.position` — records when the fallback rule fired
- `variation.persona` is always `"none"` (B has no persona axis), but the key is kept
  because `dspy_judge_optimizer.py` indexes `variation["obfuscation"]` with `[]`

IDs are `b_<category>_p<parent card_id>`, e.g. `b_direct_injection_p047`. They derive
from the plan, never from write order, so they are immune to the id-drift that affects
dataset A on resume.

## How generation works

The model never sees or returns a whole card. The harness draws `(category, position)`,
checks the position is viable on that seed, and asks the model **only for the payload**.
Six script-applied ops cover all ten categories:

| op | model returns |
|---|---|
| `REWRITE_TEXT` | the complete new value of one text field |
| `REWRITE_EXAMPLES` | a new `examples` list |
| `INSERT_SKILLS` | skill objects, spliced in at index 1 (never appended — see the ≤8 filter) |
| `REWRITE_URL` | nothing; the harness constructs the pivot URL |
| `INFLATE_TEXT` | a lead sentence + a filler unit the harness repeats to a target length |
| `GAP` | a claim sentence; the harness nulls/deletes the field and appends the contradiction |

Four categories cannot be a pure model-authored text patch, so the harness owns the
mechanism and the model owns only the prose: `url_pivot` (URL construction),
`unicode_tricks` (exact codepoints), `context_pollution` (volume), `metadata_gaps`
(deletion + contradiction). The model is still called for each, so generation
provenance matches dataset A.

## Deviations from dataset A

- No persona axis. Obfuscation, sophistication and position are unchanged.
- Cards derive from real seeds rather than invention.
- Patch-based generation instead of whole-card generation.
- `securitySchemes` is **dropped** as an injection position: the judge never sees it
  (`extract_relevant_text` forwards only name/description/url/provider/documentationUrl/
  skills[:10]), so a payload there measures nothing. Dataset A's variation matrix could
  not reach it either, so no comparability is lost.
- The 9-pair (obfuscation × sophistication) cycle is offset per category. With N=60,
  a fixed phase starves the trailing pairs identically in every category (subtle 180 vs
  overt/moderate 210); offsetting balances the totals to 201/201/198.
- `metadata_gaps` is **contradiction-first**: every record ADDS an unsupportable claim,
  with deletion only as a companion. Pure deletion is not coherent on real seeds —
  40% of genuinely BENIGN wild cards already lack `documentationUrl`, so removing it
  yields a card indistinguishable from real benign traffic. **B's `metadata_gaps`
  therefore measures "detect a provider/identity contradiction", a different construct
  from A's "detect a missing field".**

## Known caveats

- **The 660 records are not independent.** Eleven share each seed backbone. Confidence
  intervals and significance tests must cluster or bootstrap on `parent_card`; the
  effective n per category is 60 distinct cards, not 60 independent draws. Dataset A's
  260 are approximately independent. `parent_card` exists so this is fixable downstream.
- The sample is 49% of the eligible pool, so it is a substantial fraction of the
  available BENIGN wild cards, not a small random draw from a large population.
- Adversarial variants carry the **real** names, domains and URLs of live third-party
  services. See the ethics note below.

## Ethics

Dataset B contains fabricated MALICIOUS cards bearing the identities of real,
identifiable services. Nothing here was deployed, hosted, or submitted to any registry;
the cards exist only as evaluation inputs. Note that the paper's ethics statement
currently describes the adversarial dataset as "generated synthetically" — accurate for
dataset A, **not** for B. That sentence must be revised before B is released.

## Files

```
seed_selection_b.py             deterministic seed pool + sample
patch_ops_b.py                  the six-op patch vocabulary
adversarial_card_generator_b.py plan-first generator (--plan-only is free)
verify_dataset_b.py             acceptance suite (no API calls)
seeds/                          seed_pool_b.json, eligibility_report_b.md
plan/generation_plan_b.jsonl    600 rows, written before any API call
dataset_b/batch_*.jsonl         the dataset (11 files: 10 categories + benign_seed)
logs/                           generation logs
```
