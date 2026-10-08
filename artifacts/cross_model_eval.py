#!/usr/bin/env python3
"""
Cross-Model Evaluation Runner for A2A Agent Card Security Classifier

Runs the 260-card adversarial dataset against 4 models sequentially:
  1. anthropic/claude-haiku-4-5-20251001  (baseline)
  2. ollama/phi4
  3. ollama/qwen3.6:35b
  4. ollama/llama4:scout

Produces:
  cross_model_results.json  — per-model aggregate metrics
  cross_model_table.tex     — LaTeX table for §5.4

Usage:
  python cross_model_eval.py [--models haiku phi4 qwen llama4] [--limit N]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Load .env and remap OPENAI_KEY → OPENAI_API_KEY for inspect-ai / litellm
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    if "OPENAI_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
    if "GOOGLE_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_KEY"]

BASE    = Path(__file__).resolve().parent.parent            # repo root
ADV_DIR = BASE / "data" / "adversarial_dataset_b"             # 260 adversarial cards
OUT_DIR = BASE / "artifacts" / "cross_model_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "haiku":           "anthropic/claude-haiku-4-5-20251001",
    "opus48":          "anthropic/claude-opus-4-8",
    "gpt5mini":        "openai/gpt-5.4-mini-2026-03-17",
    "gpt55":           "openai/gpt-5.5-2026-04-23",
    "gemini25flash":   "google/gemini-2.5-flash-lite",
    "gemini31flashlite":"google/gemini-3.1-flash-lite",
    "phi4":            "ollama/phi4",
    "mistral_nemo":    "ollama/mistral-nemo:12b",
    "llama31_8b":      "ollama/llama3.1:8b",
}

MODEL_LABELS = {
    "haiku":            "Claude Haiku 4.5",
    "opus48":           "Claude Opus 4.8",
    "gpt5mini":         "GPT-5.4-mini",
    "gpt55":            "GPT-5.5",
    "gemini25flash":    "Gemini 2.5 Flash Lite",
    "gemini31flashlite":"Gemini 3.1 Flash Lite",
    "phi4":             "Phi-4",
    "mistral_nemo":     "Mistral Nemo",
    "llama31_8b":       "Llama 3.1",
}

# Parameter counts for the table; "—" = not publicly disclosed
MODEL_PARAMS = {
    "haiku":            r"—",
    "opus48":           r"—",
    "gpt5mini":         r"—",
    "gpt55":            r"—",
    "gemini25flash":    r"—",
    "gemini31flashlite":r"—",
    "phi4":             r"14B",
    "mistral_nemo":     r"12B",
    "llama31_8b":       r"8B",
}

# Keys ordered for the table: closed-source first, then open-source
CLOSED_SOURCE_KEYS = ["gpt55", "opus48", "gpt5mini", "gemini31flashlite",
                      "haiku", "gemini25flash"]
OPEN_SOURCE_KEYS   = ["llama31_8b", "phi4", "mistral_nemo"]

RECALL_HIT = {
    "MALICIOUS":  {"MALICIOUS"},
    "SUSPICIOUS": {"SUSPICIOUS", "MALICIOUS"},
    "BENIGN":     None,
}

VERDICT_LABELS = {"BENIGN", "SUSPICIOUS", "MALICIOUS"}

# ---------------------------------------------------------------------------
# Shared helpers (duplicated from inspect_eval.py to keep script self-contained)
# ---------------------------------------------------------------------------
import os, importlib.util

def _load_inspect_eval():
    spec = importlib.util.spec_from_file_location(
        "inspect_eval", Path(__file__).parent / "inspect_eval.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_verdict(raw: str) -> str:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        v = obj.get("verdict", "").upper()
        if v in VERDICT_LABELS:
            return v
    except json.JSONDecodeError:
        pass
    for label in ("MALICIOUS", "SUSPICIOUS", "BENIGN"):
        if f'"verdict": "{label}"' in raw or f'"verdict":"{label}"' in raw:
            return label
    for label in ("MALICIOUS", "SUSPICIOUS", "BENIGN"):
        if label in raw.upper():
            return label
    return "BENIGN"


# ---------------------------------------------------------------------------
# Compute metrics from EvalLog samples
# ---------------------------------------------------------------------------

def compute_metrics(log) -> dict:
    """Extract aggregate metrics from an inspect_ai EvalLog."""
    if not log.samples:
        return {}

    total_recall_events = 0
    recall_hits = 0
    mal_total = sus_total = 0
    mal_hits  = sus_hits  = 0
    acc_correct = 0
    acc_total   = 0

    per_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "hits": 0})
    latencies = []

    for s in log.samples:
        gt = str(s.target).strip().upper()
        raw = s.output.completion if s.output else ""
        pred = parse_verdict(raw)
        cat = s.metadata.get("category", "unknown")

        acc_total += 1
        if pred == gt:
            acc_correct += 1

        hit_set = RECALL_HIT.get(gt)
        if hit_set is None:
            continue  # BENIGN — not a recall event

        total_recall_events += 1
        is_hit = pred in hit_set
        if is_hit:
            recall_hits += 1

        if gt == "MALICIOUS":
            mal_total += 1
            if is_hit: mal_hits += 1
        elif gt == "SUSPICIOUS":
            sus_total += 1
            if is_hit: sus_hits += 1

        per_category[cat]["total"] += 1
        if is_hit:
            per_category[cat]["hits"] += 1

        if s.working_time:
            latencies.append(s.working_time)

    def pct(n, d): return round(100 * n / d, 1) if d > 0 else None

    return {
        "model":           log.eval.model if log.eval else "unknown",
        "total_samples":   acc_total,
        "accuracy":        pct(acc_correct, acc_total),
        "overall_recall":  pct(recall_hits, total_recall_events),
        "malicious_recall":pct(mal_hits, mal_total),
        "suspicious_recall":pct(sus_hits, sus_total),
        "mal_n": mal_total, "mal_hits": mal_hits,
        "sus_n": sus_total, "sus_hits": sus_hits,
        "recall_events": total_recall_events, "recall_hits": recall_hits,
        "per_category":   {
            k: {"recall": pct(v["hits"], v["total"]),
                "hits": v["hits"], "total": v["total"]}
            for k, v in sorted(per_category.items())
        },
        "avg_latency_s":   round(sum(latencies) / len(latencies), 2) if latencies else None,
        "status":          log.status,
    }


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

CATEGORIES_ORDER = [
    "direct_injection", "ranking_manipulation", "skill_deception",
    "skill_chaining", "evaluator_manipulation",
    "out_of_context_skills",
    "url_pivot", "unicode_tricks", "context_pollution", "metadata_gaps",
]

CAT_LABEL = {
    "direct_injection":       r"\texttt{direct\_injection}",
    "ranking_manipulation":   r"\texttt{ranking\_manipulation}",
    "skill_deception":        r"\texttt{skill\_deception}",
    "skill_chaining":         r"\texttt{skill\_chaining}",
    "evaluator_manipulation": r"\texttt{evaluator\_manipulation}",
    "out_of_context_skills":  r"\texttt{out\_of\_context\_skills}",
    "url_pivot":              r"\texttt{url\_pivot}",
    "unicode_tricks":         r"\texttt{unicode\_tricks}",
    "context_pollution":      r"\texttt{context\_pollution}",
    "metadata_gaps":          r"\texttt{metadata\_gaps}",
}

GROUP_LABEL = {
    "direct_injection": "Semantic", "ranking_manipulation": "Semantic",
    "skill_deception": "Semantic", "skill_chaining": "Semantic",
    "evaluator_manipulation": "Semantic", "out_of_context_skills": "Mixed",
    "url_pivot": "Structural", "unicode_tricks": "Structural",
    "context_pollution": "Structural", "metadata_gaps": "Structural",
}


def generate_latex_table(results: dict[str, dict], model_keys: list[str] | None = None) -> str:
    """Generate the cross-model LaTeX table.

    Column layout: [Category | Group | N] | closed-source models | open-source models
    Closed/open groups are separated by a vertical rule and labelled with \\multicolumn.
    """
    # Use canonical ordering unless caller overrides
    if model_keys is None:
        closed = [k for k in CLOSED_SOURCE_KEYS if k in results]
        open_  = [k for k in OPEN_SOURCE_KEYS   if k in results]
    else:
        closed = [k for k in model_keys if k in CLOSED_SOURCE_KEYS]
        open_  = [k for k in model_keys if k in OPEN_SOURCE_KEYS]

    all_keys  = closed + open_
    n_closed  = len(closed)
    n_open    = len(open_)
    n_models  = len(all_keys)

    # Column spec: left-aligned Category, Group; right-aligned N; then model columns.
    # A | separates closed from open-source blocks.
    closed_cols = "r" * n_closed
    open_cols   = "r" * n_open
    col_spec = f"llr|{closed_cols}|{open_cols}"

    # Column indices (1-based) for cmidrule
    closed_start = 4
    closed_end   = 3 + n_closed
    open_start   = closed_end + 1
    open_end     = closed_end + n_open

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-model security recall on the 260-card adversarial dataset.",
        r"  Security recall: MALICIOUS$\to$MALICIOUS; SUSPICIOUS$\to$\{SUSPICIOUS,MALICIOUS\}.",
        r"  BENIGN cards are excluded from the recall denominator.",
        r"  ``—'' in the \emph{Params} row denotes undisclosed model size.}",
        r"\label{tab:cross_model}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
    ]

    # Group header row
    closed_header = (rf"\multicolumn{{{n_closed}}}{{c}}{{\textit{{Closed-Source}}}}"
                     if n_closed else "")
    open_header   = (rf"\multicolumn{{{n_open}}}{{c}}{{\textit{{Open-Source}}}}"
                     if n_open else "")
    lines.append(
        r" & & & " + closed_header + " & " + open_header + r" \\"
    )
    # Underline each group
    if n_closed:
        lines.append(rf"\cmidrule(lr){{{closed_start}-{closed_end}}}")
    if n_open:
        lines.append(rf"\cmidrule(lr){{{open_start}-{open_end}}}")

    # Model-name header row
    model_headers = " & ".join(
        r"\textbf{" + MODEL_LABELS.get(k, k) + r"}" for k in all_keys
    )
    lines.append(
        r"\textbf{Category} & \textbf{Group} & \textbf{N} & " +
        model_headers + r" \\"
    )

    # Parameters row (immediately under model names)
    param_vals = [MODEL_PARAMS.get(k, "—") for k in all_keys]
    lines.append(
        r"\emph{Params} & & & " + " & ".join(param_vals) + r" \\"
    )
    lines.append(r"\midrule")

    # Per-category recall rows
    last_group = None
    for cat in CATEGORIES_ORDER:
        group = GROUP_LABEL.get(cat, "")
        if group != last_group and last_group is not None:
            lines.append(r"\addlinespace")
        last_group = group

        row_vals = []
        n_val = "—"
        for k in all_keys:
            cat_data = results.get(k, {}).get("per_category", {}).get(cat, {})
            recall = cat_data.get("recall")
            n_val  = cat_data.get("total", "—")
            row_vals.append(f"{recall:.0f}\\%" if recall is not None else "—")

        lines.append(
            CAT_LABEL.get(cat, cat) + f" & {group} & {n_val} & " +
            " & ".join(row_vals) + r" \\"
        )

    lines.append(r"\midrule")

    def _row(label, n_str, vals, bold=False):
        fmt = (lambda v: f"\\textbf{{{v}}}") if bold else (lambda v: v)
        return label + f" & & {n_str} & " + " & ".join(fmt(v) for v in vals) + r" \\"

    # Summary rows
    overall_vals = [
        f"{results[k]['overall_recall']:.1f}\\%" if k in results else "—"
        for k in all_keys
    ]
    lines.append(_row(r"\textbf{Overall recall}", r"\textbf{260}", overall_vals, bold=True))

    mal_vals = [
        f"{results[k]['malicious_recall']:.1f}\\%" if k in results else "—"
        for k in all_keys
    ]
    lines.append(_row(r"\quad MALICIOUS", "175", mal_vals))

    sus_vals = [
        f"{results[k]['suspicious_recall']:.1f}\\%" if k in results else "—"
        for k in all_keys
    ]
    lines.append(_row(r"\quad SUSPICIOUS", "85", sus_vals))

    acc_vals = [
        f"{results[k]['accuracy']:.1f}\\%" if k in results else "—"
        for k in all_keys
    ]
    lines.append(_row("Accuracy", "260", acc_vals))

    lat_vals = [
        f"{results[k]['avg_latency_s']:.1f}s"
        if k in results and results[k].get("avg_latency_s") is not None else "—"
        for k in all_keys
    ]
    lines.append(_row(r"Avg.\ latency", "—", lat_vals))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_model(key: str, model_id: str, limit: int | None, log_dir: Path) -> dict | None:
    """Run adversarial_eval for one model and return metrics."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset

    ie = _load_inspect_eval()
    samples = ie.load_adversarial_dataset()
    if limit:
        samples = samples[:limit]

    print(f"\n{'='*60}")
    print(f"  Model: {MODEL_LABELS.get(key, key)}  ({model_id})")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        # Disable Qwen 3 / other Ollama thinking-mode CoT to keep latency reasonable
        no_think = model_id.startswith("ollama/")
        logs = inspect_eval(
            Task(
                dataset=MemoryDataset(samples=samples),
                solver=ie.a2a_judge_solver(no_think=no_think),
                scorer=[ie.security_recall(), ie.verdict_accuracy()],
            ),
            model=model_id,
            log_dir=str(log_dir),
            display="none",
            max_connections=2,
        )
        elapsed = time.time() - t0
        log = logs[0]
        if log.status != "success":
            print(f"  ERROR: status={log.status}")
            if log.error:
                print(f"  {log.error}")
            return None
        metrics = compute_metrics(log)
        metrics["elapsed_s"] = round(elapsed, 1)
        metrics["model_key"] = key
        print(f"\n  Overall recall : {metrics.get('overall_recall')}%")
        print(f"  MALICIOUS recall: {metrics.get('malicious_recall')}%")
        print(f"  SUSPICIOUS recall: {metrics.get('suspicious_recall')}%")
        print(f"  Accuracy        : {metrics.get('accuracy')}%")
        print(f"  Elapsed         : {elapsed:.0f}s")
        return metrics
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   choices=list(MODELS.keys()),
                   default=list(MODELS.keys()),
                   help="Models to evaluate (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples per model (for smoke tests)")
    p.add_argument("--log_dir", type=Path,
                   default=OUT_DIR / "inspect_logs",
                   help="Directory for Inspect eval logs")
    args = p.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing results (so we can resume)
    results_path = OUT_DIR / "cross_model_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {results_path}")
    else:
        all_results = {}

    for key in args.models:
        if key in all_results:
            print(f"Skipping {key} — already in results. Delete {results_path} to re-run.")
            continue
        model_id = MODELS[key]
        metrics = run_model(key, model_id, args.limit, args.log_dir)
        if metrics:
            # Reload before saving to pick up results written by concurrent processes
            if results_path.exists():
                with open(results_path) as f:
                    all_results = json.load(f)
            all_results[key] = metrics
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved to {results_path}")

    # Generate LaTeX table from all completed results
    completed = [k for k in args.models if k in all_results]
    if completed:
        tex = generate_latex_table(all_results)
        tex_path = OUT_DIR / "cross_model_table.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"\nLaTeX table written to {tex_path}")

        # Print summary table to console
        print(f"\n{'Model':<25} {'Overall':>8} {'MALICIOUS':>10} {'SUSPICIOUS':>11} {'Accuracy':>9} {'Latency':>8}")
        print("-" * 75)
        for k in completed:
            r = all_results[k]
            print(f"{MODEL_LABELS.get(k, k):<25} "
                  f"{str(r.get('overall_recall','—'))+'%':>8} "
                  f"{str(r.get('malicious_recall','—'))+'%':>10} "
                  f"{str(r.get('suspicious_recall','—'))+'%':>11} "
                  f"{str(r.get('accuracy','—'))+'%':>9} "
                  f"{str(r.get('avg_latency_s','—'))+'s':>8}")


if __name__ == "__main__":
    main()
