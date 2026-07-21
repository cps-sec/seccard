# Adversarial Dataset

The 260-card synthetic adversarial dataset spanning the ten attack categories,
released in full under **CC BY 4.0**.

## Contents

- **`batch_<category>.jsonl`** (10 files, 260 cards) — one JSON object per line.
- **`METADATA_DICTIONARY.md`** — the complete field dictionary: every record- and
  card-level field, all enumerated values, the category → verdict → count mapping,
  and the generation matrix (categories × obfuscation × sophistication × personas ×
  injection position).

## Record format

Each line is a record with `id`, `category`, `expected_verdict`
(`MALICIOUS`/`SUSPICIOUS`), `group` (`judge_primary`/`rule_primary`/`mixed`),
`variation` (obfuscation/sophistication/persona/position), the full `card` object,
`injection_location`, and `payload_description`. See `METADATA_DICTIONARY.md` for the
authoritative schema.

## Responsible use

These cards are attack templates intended for defensive research and
artifact evaluation. Please use them only to build and evaluate defenses.
