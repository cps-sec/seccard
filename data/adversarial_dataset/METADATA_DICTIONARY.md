# Adversarial Dataset — Metadata Dictionary

**Dataset:** 260 synthetic A2A agent cards spanning 10 attack categories.
**Storage:** 10 files `batch_<category>.jsonl`; **one JSON object per line** (JSON Lines).
**Generator:** all cards produced by `claude-sonnet-4-6`.
**Labeling:** by construction — each card's `expected_verdict` is the verdict its `category` is designed to elicit.

Format below: `field: [enumerated values]` or `[free text]`, with a description.
"present N/260" is given where a field could in principle be absent.

---

## 1. Record-level fields (top of each JSON object)

| Field | Values | Description |
|---|---|---|
| `id` | `[free / unique]` — 260 unique strings, pattern `<category>_NNNN` (e.g. `direct_injection_0001`) | Stable per-card identifier. |
| `category` | `[context_pollution, direct_injection, evaluator_manipulation, metadata_gaps, out_of_context_skills, ranking_manipulation, skill_chaining, skill_deception, unicode_tricks, url_pivot]` | The attack class the card instantiates (one per file). |
| `expected_verdict` | `[MALICIOUS, SUSPICIOUS]` | Ground-truth label (by construction). No `BENIGN` cards in this set. |
| `group` | `[judge_primary, rule_primary, mixed]` | Which defense layer is the intended primary detector (see §5). |
| `variation` | `[object]` — see §2 | The generation-matrix coordinates for this card. |
| `card` | `[object]` — see §3 | The actual A2A agent card (the artifact under evaluation). |
| `injection_location` | `[free / semi-enumerated]` — see §6 | Where in the card the payload was placed (realized location; may be compound). |
| `payload_description` | `[free / unique]` — 260 unique strings | Human-readable note describing the embedded attack and its intent. |
| `generated_at` | `[free]` — ISO-8601 UTC timestamps | Generation timestamp. |
| `generator_model` | `[claude-sonnet-4-6]` | Model that generated the card (single value across the set). |

---

## 2. `variation.*` — generation-matrix coordinates

| Field | Values | Description |
|---|---|---|
| `variation.persona` | `[healthcare_records, legal_document, financial_analytics, enterprise_translation]` | Business domain of the card. **4 distinct values in the delivered data** (counts: healthcare_records 89, legal_document 79, financial_analytics 52, enterprise_translation 40). *The generator defines 8 personas — also `iot_sensor, code_review, supply_chain, customer_support` — but only the first 4 are represented in the released 260 cards.* |
| `variation.obfuscation` | `[overt, moderate, subtle]` | How hidden the payload is (overt = explicit markers; subtle = paraphrased/implicit). |
| `variation.sophistication` | `[naive, intermediate, advanced]` | Attacker skill level. |
| `variation.position` | `[name, description, url, documentationUrl, provider, provider.url, provider.description, skills, skills[0].description, skills[0].examples, skills[1].description]` | The card field the generator was asked to inject into. |

---

## 3. `card.*` — the A2A agent card (all present 260/260)

| Field | Values | Description |
|---|---|---|
| `card.name` | `[free text]` | Agent display name. |
| `card.description` | `[free text]` | Primary capability description (most common injection site). |
| `card.version` | `[1.0.0]` | Card content version. |
| `card.protocolVersion` | `[1.0.0]` | Declared A2A protocol version. |
| `card.url` | `[free text]` — HTTPS URLs | Agent A2A endpoint. |
| `card.documentationUrl` | `[free text]` — HTTPS URLs | Human documentation link. |
| `card.provider` | `[object]` — see §4 | Publisher identity. |
| `card.capabilities` | `[object]` — see §4 | Declared transport capabilities. |
| `card.defaultInputModes` | `[["application/json"], ["application/json","text/plain"]]` | Accepted input MIME types. |
| `card.defaultOutputModes` | `[["application/json"], ["application/json","text/plain"]]` | Produced output MIME types. |
| `card.skills` | `[array of skill objects]` — see §4; **3 or 4 skills per card** | Advertised agent skills. |

> **Note on `metadata_gaps`:** every card in the released set carries all 11 `card.*` fields (present 260/260). The `metadata_gaps` attack is expressed through *strategically thin or evasive field content*, not by physically omitting required keys.

---

## 4. Nested objects

### `card.provider.*`
| Field | Values | Description |
|---|---|---|
| `card.provider.organization` | `[free text: string]` | Publishing organization name. |
| `card.provider.url` | `[free text: string]` | Organization website. |
| `card.provider.description` | `[free text: string]` (optional) | Free-text provider blurb; sometimes an injection site. |

### `card.capabilities.*`
| Field | Values | Description |
|---|---|---|
| `card.capabilities.streaming` | `[true, false]` | Whether the agent supports streaming responses. |
| `card.capabilities.pushNotifications` | `[true, false]` | Whether the agent supports push notifications. |

### `card.skills[].*` (each of the 3–4 skills; 922 skill objects total)
| Field | Values | Description |
|---|---|---|
| `card.skills[].id` | `[free text]` | Skill identifier slug. |
| `card.skills[].name` | `[free text]` | Skill display name. |
| `card.skills[].description` | `[free text]` | Skill capability text (injection site in several categories). |
| `card.skills[].tags` | `[array of free-text tokens]` — 626 distinct tokens across the set | Domain keywords (e.g. `EHR`, `FHIR`, `HIPAA`). |
| `card.skills[].examples` | `[array of free-text strings]` | Example invocations for the skill. |

---

## 5. `group` values and their meaning

| `group` | Cards | Primary detector | Categories | `expected_verdict` |
|---|---|---|---|---|
| `judge_primary` | 175 | Layer 2B (LLM judge) — semantic intent | direct_injection, evaluator_manipulation, ranking_manipulation, skill_chaining, skill_deception (5 × 35) | all `MALICIOUS` |
| `rule_primary` | 60 | Layer 2A (deterministic NLP rules) — encoding/schema | context_pollution, metadata_gaps, unicode_tricks, url_pivot (4 × 15) | all `SUSPICIOUS` |
| `mixed` | 25 | Both layers | out_of_context_skills (1 × 25) | all `SUSPICIOUS` |

`group` is a design-time hypothesis about *how* a card should be caught; the evaluation tests whether the assigned layer actually detects it.

---

## 6. `injection_location` — realized values

The realized injection site. Usually equals `variation.position`, but may be **compound** (multiple sites) for chained attacks:

`[name, description, url, documentationUrl, provider, provider.url, provider.description, skills, skills[0].description, skills[0].examples, skills[1].description, skills[2], skills[2].description, "skills[0].description, skills[1].description, skills[2].description, skills[3].description"]`

The final compound value marks cards (e.g. `skill_chaining`) whose payload spans every skill description.

---

## 7. Category → verdict → count summary

| Category | `group` | `expected_verdict` | Cards |
|---|---|---|---|
| direct_injection | judge_primary | MALICIOUS | 35 |
| evaluator_manipulation | judge_primary | MALICIOUS | 35 |
| ranking_manipulation | judge_primary | MALICIOUS | 35 |
| skill_chaining | judge_primary | MALICIOUS | 35 |
| skill_deception | judge_primary | MALICIOUS | 35 |
| out_of_context_skills | mixed | SUSPICIOUS | 25 |
| context_pollution | rule_primary | SUSPICIOUS | 15 |
| metadata_gaps | rule_primary | SUSPICIOUS | 15 |
| unicode_tricks | rule_primary | SUSPICIOUS | 15 |
| url_pivot | rule_primary | SUSPICIOUS | 15 |
| **Total** | | 175 MAL / 85 SUS | **260** |

---

*Generated from the delivered `batch_*.jsonl` files (introspection of all 260 records). Value sets reflect what is actually present in the released dataset, not merely what the generator can produce.*
