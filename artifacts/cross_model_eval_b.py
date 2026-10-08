#!/usr/bin/env python3
"""
cross_model_eval_b.py — 9-model cross-model evaluation on adversarial dataset B.

Counterpart to cross_model_eval.py (dataset A, 260 synthetic cards). Dataset B is 660
cards: 600 attacks grafted onto 60 REAL benign seed cards, plus those 60 seeds
themselves. The seeds are BENIGN controls, so B's table carries a false-positive rate —
something dataset A structurally cannot have.

WHY THIS FILE EXISTS RATHER THAN A FLAG ON cross_model_eval.py
--------------------------------------------------------------
cross_model_eval.run_model() calls ie.load_adversarial_dataset(), which globs the
module-level ADV_DIR inside inspect_eval.py (= dataset A). cross_model_eval.py's own
ADV_DIR constant is never used for loading. A driver that merely set its own constant
would silently evaluate dataset A across all 9 models and emit a plausible-looking
table. This file therefore ships its own loader and never calls that function, and it
records a dataset fingerprint in every result so the substitution is detectable after
the fact.

Usage:
    python cross_model_eval_b.py --dataset-check                    # free, no API
    python cross_model_eval_b.py --models llama31_8b --sample-per-category 2
    python cross_model_eval_b.py                                    # full 9-model grid
    python cross_model_eval_b.py --latex-only
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                   # repo root
APPS = HERE          # all scripts live in artifacts/
# Dataset profiles. Each carries the invariants that prove the right corpus was
# loaded; the checks are as strict for A as for B, just parameterised.
# Dataset profiles. The letters here are the PAPER's: Dataset A is the seeded
# 660-card corpus, Dataset B is the 260-card synthetic one. Each profile carries
# the invariants that prove the right corpus was loaded.
DATASETS = {
    "A": {"dir": BASE / "data" / "adversarial_dataset_a",
          "n": 660, "id_prefix": "b_", "has_controls": True,
          "labels": {"BENIGN": 60, "SUSPICIOUS": 300, "MALICIOUS": 300},
          "n_categories": 11,
          "desc": "adversarial dataset A (600 attacks grafted onto 60 real benign seeds)"},
    "B": {"dir": BASE / "data" / "adversarial_dataset_b",
          "n": 260, "id_prefix": None, "has_controls": False,
          "labels": {"SUSPICIOUS": 85, "MALICIOUS": 175},
          "n_categories": 10,
          "desc": "adversarial dataset B (260 synthetic cards)"},
}
DATASET_KEY = "A"
DATASET_DIR = DATASETS["A"]["dir"]
OUT_DIR = HERE / "cross_model_results"
TABLE_LABEL = "tab:cross_model_a"
TABLE_FILE = "cross_model_table_a.tex"

PROMPT_VERSION = "v2-longform"
MAX_CONNECTIONS = 2          # match dataset A so the latency column is comparable

ATTACK_N, SEED_N, TOTAL_N = 600, 60, 660
CLASSES = ["BENIGN", "SUSPICIOUS", "MALICIOUS"]
FLAGGED = {"SUSPICIOUS", "MALICIOUS"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def guard_env() -> None:
    """inspect_ai 0.3.220 hangs forever (silently, no request issued) against
    openai>=3. Refuse to start rather than burn hours on a stalled grid."""
    from importlib.metadata import version
    o = version("openai")
    if int(o.split(".")[0]) >= 3:
        sys.exit(f"openai {o} is installed; inspect_ai 0.3.220 hangs against it.\n"
                 f"Install openai<3 (see requirements.txt)")


# ---------------------------------------------------------------------------
# Dataset B — own loader, plus proof of which dataset was actually read
# ---------------------------------------------------------------------------

def dataset_fingerprint(dataset_dir: Path | None = None) -> dict:
    # Resolved at CALL time, not definition time: a default of `DATASET_DIR` is
    # bound once at import and would keep pointing at the original dataset after
    # --dataset reassigns the global, stamping the wrong fingerprint on results.
    dataset_dir = dataset_dir or DATASET_DIR
    files = sorted(dataset_dir.glob("batch_*.jsonl"))
    h = hashlib.sha256()
    per_file = {}
    for f in files:
        b = f.read_bytes()
        h.update(f.name.encode()); h.update(b)
        per_file[f.name] = sum(1 for line in b.decode("utf-8").splitlines() if line.strip())
    return {"path": str(dataset_dir), "sha256": h.hexdigest(),
            "files": per_file, "n_files": len(files),
            "n_records": sum(per_file.values())}


def dataset_dir_msg(files) -> str:
    return str(files[0].parent) if files else "<empty>"


def load_dataset_b(ie, sample_per_category: int | None = None) -> list:
    """Never call ie.load_adversarial_dataset() — that reads dataset A."""
    from inspect_ai.dataset import Sample

    files = sorted(DATASET_DIR.glob("batch_*.jsonl"))
    if not files:
        sys.exit(f"no batch_*.jsonl under {DATASET_DIR}")
    by_cat = collections.defaultdict(list)
    for f in files:
        for line in f.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                by_cat[rec["category"]].append(rec)

    recs = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        recs += rows[:sample_per_category] if sample_per_category else rows

    # Reject a non-B dataset up front with a clear message. Without this the loader
    # still fails -- but only incidentally, on a KeyError for a field dataset A lacks.
    # An explicit check keeps the protection if A's schema ever changes.
    prof = DATASETS[DATASET_KEY]
    if prof["id_prefix"]:
        for rec in recs[:5]:
            missing = {"in_the_wild", "parent_card", "patch_paths"} - set(rec)
            if missing or not str(rec.get("id", "")).startswith(prof["id_prefix"]):
                sys.exit(f"{dataset_dir_msg(files)} is not {prof['desc']}\n"
                         f"  id={rec.get('id')!r}  missing fields: {sorted(missing)}")
    else:
        for rec in recs[:5]:
            if "in_the_wild" in rec:
                sys.exit(f"{dataset_dir_msg(files)} carries 'in_the_wild' — that is "
                         f"dataset B, but --dataset A was requested")

    samples = []
    for rec in recs:
        samples.append(Sample(
            id=rec["id"],
            input=ie.build_user_message(ie.extract_relevant_text(rec["card"])),
            target=rec["expected_verdict"],
            metadata={
                "card_id":        rec["id"],
                "category":       rec["category"],
                "group":          rec.get("group", ""),
                "obfuscation":    rec["variation"].get("obfuscation", ""),
                "sophistication": rec["variation"].get("sophistication", ""),
                "in_the_wild":    rec.get("in_the_wild", 0),   # A has no controls
                "parent_card":    rec.get("parent_card", 0),
                "dataset":        f"adversarial_{DATASET_KEY.lower()}",
            },
        ))

    # Proof we loaded B, not A. A's ids look like "direct_injection_0001".
    if prof["id_prefix"]:
        bad = [s.id for s in samples if not str(s.id).startswith(prof["id_prefix"])]
        if bad:
            sys.exit(f"ids must start with {prof['id_prefix']!r}; found {bad[:3]}")
    cats = {s.metadata["category"] for s in samples}
    if prof["has_controls"] and "benign_seed" not in cats:
        sys.exit("no benign_seed records — dataset B must carry its BENIGN controls")
    if not prof["has_controls"] and "benign_seed" in cats:
        sys.exit("benign_seed present but the profile expects no controls")
    if sample_per_category is None:
        n_seed = sum(1 for s in samples if s.metadata["in_the_wild"] == 1)
        counts = collections.Counter(str(s.target) for s in samples)
        exp_seed = prof["labels"].get("BENIGN", 0)
        if len(samples) != prof["n"] or n_seed != exp_seed:
            sys.exit(f"expected {prof['n']} samples / {exp_seed} controls, "
                     f"got {len(samples)} / {n_seed}")
        if counts != collections.Counter(prof["labels"]):
            sys.exit(f"unexpected label distribution: {dict(counts)} "
                     f"(expected {prof['labels']})")
        if len(cats) != prof["n_categories"]:
            sys.exit(f"expected {prof['n_categories']} categories, got {len(cats)}")
    if len({s.id for s in samples}) != len(samples):
        sys.exit("duplicate sample ids")
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_b(log, ie, cme) -> dict:
    """Forked from cross_model_eval.compute_metrics, which `continue`s on BENIGN
    targets before touching recall, per_category and latency (:152-158). On dataset B
    that would silently discard all 60 controls — exactly the metric B exists for."""
    rows = []
    for s in log.samples:
        raw = s.output.completion if s.output else ""
        parsed = ie._parse_verdict(raw)
        md = s.metadata or {}
        rows.append({
            "id": s.id, "gt": str(s.target).strip().upper(),
            "pred": parsed or "BENIGN", "parseable": parsed is not None,
            "category": md.get("category", "unknown"),
            "in_the_wild": md.get("in_the_wild", 0),
            "parent_card": md.get("parent_card", 0),
            "latency": s.working_time,
        })

    attacks = [r for r in rows if r["in_the_wild"] == 0]
    seeds = [r for r in rows if r["in_the_wild"] == 1]

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    # ---- attacks -------------------------------------------------------
    per_cat = collections.defaultdict(lambda: {"hits": 0, "total": 0})
    hits = mal_n = mal_h = sus_n = sus_h = exact = 0
    for r in attacks:
        hit_set = cme.RECALL_HIT.get(r["gt"])
        if hit_set is None:            # should not occur: attacks are never BENIGN
            continue
        is_hit = r["pred"] in hit_set
        hits += is_hit
        if r["gt"] == "MALICIOUS":
            mal_n += 1; mal_h += is_hit
        else:
            sus_n += 1; sus_h += is_hit
        per_cat[r["category"]]["total"] += 1
        per_cat[r["category"]]["hits"] += is_hit
        exact += (r["pred"] == r["gt"])

    # ---- seeds (the BENIGN controls) -----------------------------------
    s_correct = sum(1 for r in seeds if r["pred"] == "BENIGN")
    s_sus = sum(1 for r in seeds if r["pred"] == "SUSPICIOUS")
    s_mal = sum(1 for r in seeds if r["pred"] == "MALICIOUS")

    # ---- paired: does the model flag the child yet clear its own parent? --
    seed_pred = {r["parent_card"]: r["pred"] for r in seeds} if seeds else {}
    seed_pred_by_id = {}
    for r in seeds:
        # seed ids look like b_benign_seed_pNNN -> recover NNN
        try:
            seed_pred_by_id[int(str(r["id"]).rsplit("_p", 1)[1])] = r["pred"]
        except (IndexError, ValueError):
            pass
    paired_ok = paired_total = 0
    for r in attacks:
        p = seed_pred_by_id.get(r["parent_card"])
        if p is None:
            continue
        paired_total += 1
        hit_set = cme.RECALL_HIT.get(r["gt"]) or set()
        if r["pred"] in hit_set and p == "BENIGN":
            paired_ok += 1

    lat = [r["latency"] for r in rows if r["latency"]]
    return {
        "model": log.eval.model if log.eval else "unknown",
        "prompt_version": PROMPT_VERSION,
        "status": log.status,
        "total_samples": len(rows),
        "attacks": {
            "n": len(attacks), "recall": pct(hits, len(attacks)),
            "mal_n": mal_n, "mal_hits": mal_h, "malicious_recall": pct(mal_h, mal_n),
            "sus_n": sus_n, "sus_hits": sus_h, "suspicious_recall": pct(sus_h, sus_n),
            "accuracy_exact": pct(exact, len(attacks)),
            "per_category": {c: {"recall": pct(v["hits"], v["total"]),
                                 "hits": v["hits"], "total": v["total"]}
                             for c, v in sorted(per_cat.items())},
        },
        "seeds": {
            "n": len(seeds), "correct_benign": s_correct,
            "accuracy": pct(s_correct, len(seeds)),
            "fpr": pct(len(seeds) - s_correct, len(seeds)),
            "to_suspicious": s_sus, "to_malicious": s_mal,
        },
        "overall": {
            "accuracy_all": pct(sum(1 for r in rows if r["pred"] == r["gt"]), len(rows)),
            "unparseable_n": sum(1 for r in rows if not r["parseable"]),
            "pred_counts": dict(collections.Counter(r["pred"] for r in rows)),
        },
        "paired": {"n": paired_total, "caught_child_cleared_parent": paired_ok,
                   "pct": pct(paired_ok, paired_total)},
        "avg_latency_s": round(statistics.mean(lat), 2) if lat else None,
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        "predictions": [{"id": r["id"], "category": r["category"], "gt": r["gt"],
                         "pred": r["pred"], "parseable": r["parseable"],
                         "in_the_wild": r["in_the_wild"],
                         "parent_card": r["parent_card"]} for r in rows],
    }


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

MODEL_HEADER_TEX = {
    "gpt55": r"GPT-5.5", "opus48": r"\shortstack{Claude\\Opus 4.8}",
    "gpt5mini": r"\shortstack{GPT-5.4\\mini}",
    "gemini31flashlite": r"\shortstack{Gemini 3.1\\Flash Lite}",
    "haiku": r"\shortstack{Claude\\Haiku 4.5}",
    "gemini25flash": r"\shortstack{Gemini 2.5\\Flash Lite}",
    "llama31_8b": r"\shortstack{Llama\\3.1}", "phi4": r"Phi-4",
    "mistral_nemo": r"\shortstack{Mistral\\Nemo}",
}


def generate_latex_table_b(results: dict, cme) -> str:
    closed = [k for k in cme.CLOSED_SOURCE_KEYS if k in results]
    open_ = [k for k in cme.OPEN_SOURCE_KEYS if k in results]
    keys = closed + open_
    if not keys:
        raise RuntimeError("no results to tabulate")

    def derive(getter, what):
        vals = {getter(results[k]) for k in keys}
        if len(vals) != 1:
            raise RuntimeError(f"models disagree on {what}: {vals} — partial run?")
        return str(vals.pop())

    n_att = derive(lambda m: m["attacks"]["n"], "attack count")
    n_mal = derive(lambda m: m["attacks"]["mal_n"], "MALICIOUS count")
    n_sus = derive(lambda m: m["attacks"]["sus_n"], "SUSPICIOUS count")
    n_seed = derive(lambda m: m["seeds"]["n"], "seed count")
    has_controls = int(n_seed) > 0

    nc, no = len(closed), len(open_)
    # guard against "llr||r" when one source group is empty (e.g. locals-only run)
    groups = [g for g in ("r" * nc, "r" * no) if g]
    spec = "llr|" + "|".join(groups)
    ncols = 3 + nc + no

    def f(v, dp=0, suffix=r"\%"):
        return "—" if v is None or v != v else f"{v:.{dp}f}{suffix}"

    L = [
        r"% Auto-generated by cross_model_eval_b.py — do not hand-edit.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Cross-model security recall on the " + str(int(n_att) + int(n_seed)) +
        r"-card " + DATASETS[DATASET_KEY]["desc"].split(" (")[0] + r",",
    ] + ([
        r"  in which " + n_att + r" attacks are grafted onto " + n_seed +
        r" real benign agent cards that are themselves included as controls.",
    ] if has_controls else [
        r"  a fully synthetic corpus with no benign cards.",
    ]) + [
        r"  Security recall: MALICIOUS$\to$MALICIOUS; SUSPICIOUS$\to$\{SUSPICIOUS,MALICIOUS\}.",
    ] + ([
        r"  The \emph{Controls} block reports behaviour on the " + n_seed +
        r" untouched seed cards; seed accuracy and seed false-positive rate are",
        r"  complements, shown separately for readability.",
    ] if has_controls else [
        r"  There are no BENIGN cards, so this table admits no false-positive rate.",
    ]) + [
        r"  Judged with Prompt \texttt{" + PROMPT_VERSION.replace("_", r"\_") + r"}.",
        r"  ``—'' in the \emph{Params} row denotes undisclosed model size.}",
        r"\label{" + TABLE_LABEL + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
    ]
    if nc and no:
        L.append(r" & & & \multicolumn{" + str(nc) + r"}{c}{\textit{Closed-Source}} & "
                 r"\multicolumn{" + str(no) + r"}{c}{\textit{Open-Source}} \\")
        L.append(r"\cmidrule(lr){4-" + str(3 + nc) + "}")
        L.append(r"\cmidrule(lr){" + str(4 + nc) + "-" + str(3 + nc + no) + "}")
    L.append(r"\textbf{Category} & \textbf{Group} & \textbf{N} & " +
             " & ".join(r"\textbf{" + MODEL_HEADER_TEX.get(k, cme.MODEL_LABELS.get(k, k))
                        + "}" for k in keys) + r" \\")
    L.append(r"\emph{Params} & & & " +
             " & ".join(cme.MODEL_PARAMS.get(k, "—") for k in keys) + r" \\")
    L.append(r"\midrule")

    prev_group = None
    for cat in cme.CATEGORIES_ORDER:
        grp = cme.GROUP_LABEL.get(cat, "")
        if prev_group is not None and grp != prev_group:
            L.append(r"\addlinespace")
        prev_group = grp
        totals = {results[k]["attacks"]["per_category"].get(cat, {}).get("total")
                  for k in keys}
        totals.discard(None)
        n_val = str(totals.pop()) if len(totals) == 1 else "—"
        cells = [f(results[k]["attacks"]["per_category"].get(cat, {}).get("recall"))
                 for k in keys]
        L.append(f"{cme.CAT_LABEL.get(cat, cat)} & {grp} & {n_val} & "
                 + " & ".join(cells) + r" \\")

    def row(label, n, getter, bold=False, dp=1, suffix=r"\%"):
        cells = [f(getter(results[k]), dp, suffix) for k in keys]
        if bold:
            label = r"\textbf{" + label + "}"; n = r"\textbf{" + n + "}"
            cells = [r"\textbf{" + c + "}" for c in cells]
        return f"{label} & & {n} & " + " & ".join(cells) + r" \\"

    L.append(r"\midrule")
    L.append(row(r"Overall recall", n_att, lambda m: m["attacks"]["recall"], bold=True))
    L.append(row(r"\quad MALICIOUS", n_mal, lambda m: m["attacks"]["malicious_recall"]))
    L.append(row(r"\quad SUSPICIOUS", n_sus, lambda m: m["attacks"]["suspicious_recall"]))
    L.append(row(r"Accuracy (attacks)", n_att, lambda m: m["attacks"]["accuracy_exact"]))

    L.append(r"\midrule")
    if has_controls:
        L.append(r"\multicolumn{" + str(ncols) +
                 r"}{l}{\textit{Controls --- " + n_seed + r" benign seed cards}} \\")
        L.append(row(r"Seed accuracy", n_seed, lambda m: m["seeds"]["accuracy"]))
        L.append(row(r"Seed false-positive rate", n_seed, lambda m: m["seeds"]["fpr"]))
    L.append(row(r"Avg.\ latency", "—", lambda m: m["avg_latency_s"], suffix="s"))

    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def results_path() -> Path:
    # Must vary by dataset: both datasets share one results directory in this repo,
    # and a fixed filename would make the second run overwrite the first.
    return OUT_DIR / f"cross_model_results_{DATASET_KEY.lower()}.json"


def load_results() -> dict:
    p = results_path()
    return json.loads(p.read_text()) if p.exists() else {}


def atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def run_model_b(key, model_id, ie, cme, samples, log_dir, fp) -> dict | None:
    from inspect_ai import Task, eval as inspect_eval_fn
    from inspect_ai.dataset import MemoryDataset

    log_dir.mkdir(parents=True, exist_ok=True)
    no_think = model_id.startswith("ollama/")
    print(f"  {key}: {model_id}{'  [no_think]' if no_think else ''}  "
          f"{len(samples)} samples  prompt={PROMPT_VERSION}")
    t0 = time.time()
    logs = inspect_eval_fn(
        Task(dataset=MemoryDataset(samples=samples),
             solver=ie.a2a_judge_solver(no_think=no_think,
                                        prompt_version=PROMPT_VERSION),
             scorer=[ie.security_recall(), ie.verdict_accuracy()]),
        model=model_id, log_dir=str(log_dir), display="none",
        max_connections=MAX_CONNECTIONS)
    elapsed = time.time() - t0
    log = logs[0]
    if log.status != "success":
        print(f"    FAILED: {log.status} {getattr(log, 'error', '')}")
        return None

    # Post-hoc proof that the model judged the cards we loaded. Comparing the log's
    # id SET against the loaded samples is stronger than a prefix test and works for
    # any dataset profile (dataset A's ids carry no distinguishing prefix).
    seen = {str(s.id) for s in log.samples}
    want = {str(s.id) for s in samples}
    if seen != want:
        only_log, only_want = sorted(seen - want)[:3], sorted(want - seen)[:3]
        sys.exit(f"{key}: log id set != loaded sample id set "
                 f"(log-only {only_log}, missing {only_want})")

    m = compute_metrics_b(log, ie, cme)
    m["model_key"] = key
    m["elapsed_s"] = round(elapsed, 1)
    m["dataset_fingerprint"] = fp
    m["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return m


def main() -> None:
    global OUT_DIR, PROMPT_VERSION, DATASET_KEY, DATASET_DIR, TABLE_LABEL, TABLE_FILE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cme = _load("cme", HERE / "cross_model_eval.py")
    ap.add_argument("--models", nargs="+", choices=list(cme.MODELS),
                    default=list(cme.MODELS))
    ap.add_argument("--sample-per-category", type=int,
                    help="take N records per category (stratified smoke test)")
    ap.add_argument("--prompt_version", default=PROMPT_VERSION)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="B")
    ap.add_argument("--label", help="LaTeX label for the generated table")
    ap.add_argument("--table-file", help="filename for the generated .tex")
    ap.add_argument("--dataset-check", action="store_true",
                    help="verify the dataset and exit; no API calls")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute metrics from saved .eval logs and diff against "
                         "the stored results; no API calls")
    ap.add_argument("--latex-only", action="store_true")
    ap.add_argument("--force", nargs="+", default=[])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    DATASET_KEY = args.dataset
    DATASET_DIR = DATASETS[DATASET_KEY]["dir"]
    OUT_DIR = args.out
    PROMPT_VERSION = args.prompt_version
    TABLE_LABEL = args.label or f"tab:cross_model_{DATASET_KEY.lower()}"
    TABLE_FILE = args.table_file or f"cross_model_table_{DATASET_KEY.lower()}.tex"

    fp = dataset_fingerprint()
    print(f"dataset {DATASET_KEY}: {fp['path']}")
    print(f"  files {fp['n_files']}  records {fp['n_records']}  "
          f"sha256 {fp['sha256'][:16]}...")

    if args.latex_only:
        results = load_results()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / TABLE_FILE).write_text(
            generate_latex_table_b(results, cme))
        print(f"Wrote {OUT_DIR/TABLE_FILE}")
        return

    if args.recompute:
        # Recompute every model's metrics from its saved .eval log and diff against
        # what is stored. Used to prove a refactor is behaviour-preserving.
        import glob as _glob
        ie = _load("ie", HERE / "inspect_eval.py")
        from inspect_ai.log import read_eval_log
        stored = load_results()
        if not stored:
            sys.exit(f"no results at {results_path()}")
        diffs = 0
        for key in sorted(stored):
            logs = sorted(_glob.glob(str(OUT_DIR / "inspect_logs" / key / "*.eval")))
            if not logs:
                print(f"  {key:22} no log — skipped"); continue
            fresh = compute_metrics_b(read_eval_log(logs[-1]), ie, cme)
            old_m, new_m = stored[key], fresh
            fields = [("attacks.recall", lambda m: m["attacks"]["recall"]),
                      ("attacks.mal", lambda m: m["attacks"]["malicious_recall"]),
                      ("attacks.sus", lambda m: m["attacks"]["suspicious_recall"]),
                      ("attacks.acc", lambda m: m["attacks"]["accuracy_exact"]),
                      ("seeds.accuracy", lambda m: m["seeds"]["accuracy"]),
                      ("seeds.fpr", lambda m: m["seeds"]["fpr"]),
                      ("total_samples", lambda m: m["total_samples"])]
            bad = [n for n, g in fields if g(old_m) != g(new_m)]
            cat_bad = [c for c in old_m["attacks"]["per_category"]
                       if old_m["attacks"]["per_category"][c]["recall"]
                       != new_m["attacks"]["per_category"].get(c, {}).get("recall")]
            status = "IDENTICAL" if not bad and not cat_bad else f"DIFFERS {bad+cat_bad}"
            if bad or cat_bad:
                diffs += 1
            print(f"  {key:22} {status}")
        print(f"\n{diffs} model(s) differ from stored metrics")
        sys.exit(1 if diffs else 0)

    guard_env()
    ie = _load("ie", HERE / "inspect_eval.py")
    # register the longform prompt family (it lives in evaluate_card_intent.py)
    ec = _load("ec", HERE / "evaluate_card_intent.py")
    for k, v in ec.PROMPT_VERSIONS.items():
        ie.PROMPT_VERSIONS.setdefault(k, v)
    if PROMPT_VERSION not in ie.PROMPT_VERSIONS:
        sys.exit(f"unknown prompt {PROMPT_VERSION}; have {sorted(ie.PROMPT_VERSIONS)}")
    print(f"  prompt {PROMPT_VERSION} "
          f"({len(ie.PROMPT_VERSIONS[PROMPT_VERSION]):,} chars)")

    samples = load_dataset_b(ie, args.sample_per_category)
    cats = collections.Counter(s.metadata["category"] for s in samples)
    labels = collections.Counter(str(s.target) for s in samples)
    print(f"  loaded {len(samples)} samples across {len(cats)} categories")
    print(f"  labels {dict(labels)}")
    print(f"  ids e.g. {[str(s.id) for s in samples[:3]]}")
    if args.dataset_check:
        print("\nDataset check passed.")
        return

    results = load_results()
    for key in args.models:
        prev = results.get(key)
        if prev and key not in args.force:
            print(f"\n[{key}] cached — skipping")
            continue
        if prev is None:
            for k2, m2 in results.items():
                if m2.get("dataset_fingerprint", {}).get("sha256") != fp["sha256"]:
                    sys.exit(f"dataset changed since {k2} ran — refusing to mix results")
        print(f"\n[{key}] " + "-" * 56)
        try:
            m = run_model_b(key, cme.MODELS[key], ie, cme, samples,
                            OUT_DIR / "inspect_logs" / key, fp)
        except Exception as e:                                       # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue
        if not m:
            continue
        results = load_results(); results[key] = m
        atomic_write(results_path(), results)
        a, s = m["attacks"], m["seeds"]
        print(f"    recall {a['recall']}%  (MAL {a['malicious_recall']}%  "
              f"SUS {a['suspicious_recall']}%)  acc {a['accuracy_exact']}%  |  "
              f"seed acc {s['accuracy']}%  seed FPR {s['fpr']}%  "
              f"({m['elapsed_s']/60:.1f} min)")

    results = load_results()
    if results:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / TABLE_FILE).write_text(
                generate_latex_table_b(results, cme))
            print(f"\nWrote {OUT_DIR/TABLE_FILE}")
        except RuntimeError as e:
            print(f"\n(table not written: {e})")
        print(f"\n{'model':22} {'recall':>8} {'MAL':>7} {'SUS':>7} {'acc':>7} "
              f"{'seedAcc':>8} {'seedFPR':>8} {'lat':>7}")
        print("-" * 80)
        for k in [x for x in cme.CLOSED_SOURCE_KEYS + cme.OPEN_SOURCE_KEYS
                  if x in results]:
            m = results[k]; a, s = m["attacks"], m["seeds"]
            g = lambda v: "—" if v is None else f"{v:.1f}"
            print(f"{k:22} {g(a['recall']):>8} {g(a['malicious_recall']):>7} "
                  f"{g(a['suspicious_recall']):>7} {g(a['accuracy_exact']):>7} "
                  f"{g(s['accuracy']):>8} {g(s['fpr']):>8} "
                  f"{g(m['avg_latency_s']):>7}")


if __name__ == "__main__":
    main()
