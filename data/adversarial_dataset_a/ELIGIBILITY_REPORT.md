# Seed eligibility — adversarial dataset B

RNG seed `20260922` · pool **123** · sampled **60** (49% of the pool)

| filter stage | remaining |
|---|---|
| human_label == BENIGN | 282 |
| raw card JSON <= 6000 | 260 |
| >= 2 skills | 187 |
| <= 8 skills (judge sees skills[:10]) | 177 |
| first 2 skills well-formed | 175 |
| stored structural_score >= 8 | 143 |
| live validate_card >= 8 | 129 |
| dedupe by card content | 123 |

## Field coverage in the sampled seeds

| field | seeds |
|---|---|
| `has_url` | 52/60 |
| `has_documentationUrl` | 38/60 |
| `has_provider` | 55/60 |
| `has_provider_url` | 55/60 |
| `has_skill0_examples` | 41/60 |
