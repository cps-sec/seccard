# In-the-wild corpus — Trusted/Untrusted summary

Trust policy: **Trusted iff structural_score >= 8 (Tier 0) AND ensemble = BENIGN**; otherwise Untrusted. Only Trusted cards are allowed for automatic orchestrator consumption.

| Metric | Count |
|---|---|
| Total cards | 310 |
| Reached Tier 0 (score ≥ 8) | 207 |
| **Trusted** (Tier 0 ∧ ensemble BENIGN) | **172** |
| **Untrusted** | **138** |
| &nbsp;&nbsp;· did not reach Tier 0 | 103 |
| &nbsp;&nbsp;· reached Tier 0 but ensemble flagged | 35 |

Per-layer verdict classification (all 310 cards):

| Layer | BENIGN | SUSPICIOUS | MALICIOUS |
|---|---|---|---|
| L2A (NLP rules) | 202 | 105 | 3 |
| L2B v1 (LLM judge, prompt v1) | 265 | 44 | 1 |
| L2B v2 (LLM judge, prompt v2) | 146 | 156 | 8 |
| Ensemble (2-of-3 majority) | 228 | 80 | 2 |

> Baked `ensemble_verdict` / `trust` fields match the recomputation for all 310 cards.
