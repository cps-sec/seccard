# Adversarial dataset — Trusted/Untrusted summary

Trust policy: **Trusted iff structural_score >= 8 (Tier 0) AND ensemble = BENIGN**; otherwise Untrusted. Only Trusted cards are allowed for automatic orchestrator consumption.

| Metric | Count |
|---|---|
| Total cards | 260 |
| Reached Tier 0 (score ≥ 8) | 260 |
| **Trusted** (Tier 0 ∧ ensemble BENIGN) | **21** |
| **Untrusted** | **239** |
| &nbsp;&nbsp;· did not reach Tier 0 | 0 |
| &nbsp;&nbsp;· reached Tier 0 but ensemble flagged | 239 |

Per-layer verdict classification (all 260 cards):

| Layer | BENIGN | SUSPICIOUS | MALICIOUS |
|---|---|---|---|
| L2A (NLP rules) | 158 | 72 | 30 |
| L2B v1 (LLM judge, prompt v1) | 26 | 99 | 135 |
| L2B v2 (LLM judge, prompt v2) | 8 | 95 | 157 |
| Ensemble (2-of-3 majority) | 21 | 107 | 132 |

> These are attack cards, so a `Trusted` verdict is a **false negative** (an attack that would be auto-consumed). The ensemble withholds 239/260 (91.9%) of attacks from automatic consumption.
