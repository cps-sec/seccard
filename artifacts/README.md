# Artifacts — command reference

Five Python scripts implementing the paper's defense pipeline and evaluation.
Run them from the repository root as `python artifacts/<script>.py …`; each script
resolves `data/` relative to its own location, so the working directory does not
matter. Install dependencies first: `pip install -r ../requirements.txt`.

Dependency graph (all scripts live here, so imports resolve):
`a2a_card_health_validator.py` → `validate_agent_cards.py`, `nlp_features.py`;
`cross_model_eval.py` → `inspect_eval.py`.

---

## `validate_agent_cards.py` — Layer 1: conformance rubric

Scores each card 0–10 against the A2A schema (required fields, optional fields,
false-positive penalty) and classifies it valid/invalid at a threshold. Reproduces
the Table 1 conformance statistics.

| Parameter | Type / default | Description |
|---|---|---|
| `-i, --input` | path (**required**) | Input JSON in crawl format `{"results": [{"domain","url","card"}, …]}`. |
| `-o, --output` | path (def `<input>_validated.json`) | Validated-results output JSON. |
| `--csv` | path (def `<input>_valid.csv`) | CSV of valid domains + scores. |
| `--threshold` | int (def `8`) | Minimum 0–10 score to classify a card as valid. |
| `-v, --verbose` | flag | Debug logging. |

```bash
python artifacts/validate_agent_cards.py -i data/in_the_wild_corpus/in_the_wild_corpus_310.json --threshold 3
```

---

## `nlp_features.py` — Layer 2A: deterministic NLP rules

Extracts the millisecond-scale rule features (imperative/system-framing/
classifier-addressing regexes, promotional-pressure composite, Unicode and URL
checks, description↔skill coherence) and maps them to BENIGN/SUSPICIOUS/MALICIOUS.

Input is the **`validate_agent_cards.py` output** (it reads the `valid_results`
list), so run Layer 1 first.

| Parameter | Type / default | Description |
|---|---|---|
| `--input` | path (def `cards.json`) | Validated-cards JSON produced by `validate_agent_cards.py`. |
| `--orig` | path (def `agent_cards.json`) | Original crawler output for provenance (optional). |
| `--output` | path (def `nlp_features.json`) | Per-card feature/verdict output. |
| `--csv` | path (def `nlp_features.csv`) | CSV summary. |
| `--sentiment-backend` | `vader` \| `siebert` (def `vader`) | Sentiment gate backend. `vader` is offline; `siebert` needs `transformers`+`torch`. |
| `--sentiment-threshold` | float (def `0.85`) | Promotional-pressure gate threshold. |
| `-v, --verbose` | flag | Debug logging. |

```bash
python artifacts/validate_agent_cards.py -i data/in_the_wild_corpus/in_the_wild_corpus_310.json \
    --threshold 3 -o validated.json
python artifacts/nlp_features.py --input validated.json --sentiment-backend vader
```

---

## `a2a_card_health_validator.py` — Tier 0/1/2 trust validator

Applies the hierarchical trust gates: Tier 0 (schema compliance), Tier 1 (semantic
hygiene, invokes Layer 2A), Tier 2 (cryptographic trust — JWS ES256, x5c chain,
Fulcio issuer, Rekor transparency log). Single-card or dataset mode.

| Parameter | Type / default | Description |
|---|---|---|
| `--card-file` | path (def `agent-card.json`) | Score a single card JSON (single-card mode). |
| `--input` | path (def `cc_probe_validated.json`) | Validated-cards JSON (dataset mode). |
| `--orig` | path (def `agent_cards.json`) | Original crawler output (dataset mode, optional). |
| `--output` | path (def `health.json`) | Output JSON (dataset mode). |
| `--csv` | path (def `health.csv`) | Output CSV (dataset mode). |
| `--dataset` | flag | Force dataset mode. |
| `--skip-rekor` | flag | Skip Rekor network calls (faster; safe for unsigned cards). |
| `-v, --verbose` | flag | Debug logging. |

```bash
python artifacts/a2a_card_health_validator.py --card-file <card.json> --skip-rekor
```

---

## `inspect_eval.py` — Layer 2B: LLM-as-judge evaluation

Runs the hardened LLM judge (UK AISI Inspect harness) and scores **security recall**
(MALICIOUS→MALICIOUS; SUSPICIOUS→{SUSPICIOUS,MALICIOUS}; BENIGN excluded).
Reproduces Tables 5–8. Requires `ANTHROPIC_API_KEY` (or an Ollama model id).

| Parameter | Type / default | Description |
|---|---|---|
| `task` | `adversarial` \| `wild` \| `all` (**positional**) | Which evaluation to run. |
| `--model` | str (def `anthropic/claude-haiku-4-5-20251001`) | Inspect model id (e.g. `ollama/llama3.1:8b`). |
| `--limit` | int (def none) | Cap the number of samples (smoke tests). |
| `--category` | str (def none) | Filter adversarial samples to one category (e.g. `ranking_manipulation`). |
| `--dry_run` | flag | Print dataset stats and exit without calling the API. |
| `--log_dir` | path (def `./inspect_logs`) | Inspect log directory. |
| `--prompt_version` | `v2` \| `v3` (def `v2`) | System-prompt version. |

- **`wild`** runs on the shipped `data/in_the_wild_corpus/in_the_wild_corpus_310.json`
  (raw card text). Exact ground truth is optional — place
  `cc_probe_nlp_features.json`, `cc_probe_intent_v2.json`, and
  `wild_cards_human_labels.json` (available on request) in `data/wild_verdicts/`.
- **`adversarial`** runs on the 260 cards shipped in `data/adversarial_dataset/`.

```bash
python artifacts/inspect_eval.py wild --limit 5
python artifacts/inspect_eval.py adversarial --dry_run
```

---

## `cross_model_eval.py` — nine-model cross-study

Replicates the adversarial evaluation across nine models (closed-source via API,
open-weight via Ollama) and writes `artifacts/cross_model_results/`. Reproduces
Table 5. Loads `.env` from the repo root; evaluates the 260 adversarial cards in
`data/adversarial_dataset/`.

| Parameter | Type / default | Description |
|---|---|---|
| `--models` | one or more of the model keys (def: all) | Subset to run: `haiku opus48 gpt5mini gpt55 gemini25flash gemini31flashlite phi4 mistral_nemo llama31_8b`. |
| `--limit` | int (def none) | Cap samples per model (smoke tests). |
| `--log_dir` | path (def `artifacts/cross_model_results/inspect_logs`) | Inspect log directory. |

```bash
python artifacts/cross_model_eval.py --models haiku --limit 5
```

Requires `ANTHROPIC_API_KEY` (Claude), `OPENAI_API_KEY` (GPT), `GOOGLE_API_KEY`
(Gemini) as applicable; open-weight models (`phi4`, `mistral_nemo`, `llama31_8b`)
run through a local Ollama server.

---

## `adversarial_card_generator.py` — regenerate the adversarial dataset

Produces the 260-card adversarial dataset (`claude-sonnet-4-6`) into
`data/adversarial_dataset/` as `batch_<category>.jsonl`, checkpoint/resume aware. The
exact system prompt, user-message template, and taxonomy are documented in
`data/adversarial_dataset/GENERATION_PROMPT.md`. Needs `ANTHROPIC_API_KEY`.

| Parameter | Description |
|---|---|
| `--batch CATEGORY` | Generate a single category batch (resume-aware). |
| `--all` | Generate all pending category batches sequentially. |
| `--status` | Show per-category completion status. |
| `--validate CATEGORY` | Validate an existing batch against the Tier-0 structural checker. |

```bash
python artifacts/adversarial_card_generator.py --status
python artifacts/adversarial_card_generator.py --all
```
