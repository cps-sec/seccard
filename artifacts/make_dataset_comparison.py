#!/usr/bin/env python3
"""
make_dataset_comparison.py — LaTeX table comparing the two adversarial datasets.

Every number is computed from the datasets on disk; nothing is hardcoded.

COLUMN NAMING
-------------
Internally the 260-card synthetic set is "A" and the 660-card seeded set is "B"
(that is how they are named on disk and throughout the generation scripts). The
PAPER uses the opposite convention: the seeded set is presented as Dataset~A.

SWAP_COLUMNS=True renders the paper's convention: the column headed "Dataset A"
carries the seeded 660-card set, and "Dataset B" carries the synthetic 260-card
set. The caption text is swapped with it, so the prose always describes the
column it sits above. Set SWAP_COLUMNS=False to get the on-disk convention.
"""

import collections
import glob
import importlib.util
import json
import statistics
from pathlib import Path

SWAP_COLUMNS = True           # paper convention: seeded set is presented as "A"

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                   # repo root
SYNTH_GLOB = str(BASE / "data" / "adversarial_dataset_b" / "batch_*.jsonl")
SEEDED_GLOB = str(BASE / "data" / "adversarial_dataset_a" / "batch_*.jsonl")

CATEGORY_ORDER = ["direct_injection", "ranking_manipulation", "skill_deception",
                  "skill_chaining", "evaluator_manipulation", "out_of_context_skills",
                  "url_pivot", "unicode_tricks", "context_pollution", "metadata_gaps"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def read(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern))
            for l in open(f, encoding="utf-8") if l.strip()]


def main() -> None:
    ie = _load("ie", HERE / "inspect_eval.py")
    V = _load("v", HERE / "validate_agent_cards.py")

    SY = read(SYNTH_GLOB)                                   # 260 synthetic
    SE_all = read(SEEDED_GLOB)                              # 660 seeded
    SE = [r for r in SE_all if r.get("in_the_wild") == 0]    # 600 attacks
    SE_seed = [r for r in SE_all if r.get("in_the_wild") == 1]

    def cnt(recs, key):
        return collections.Counter(r["variation"].get(key) for r in recs)

    def sc(recs):
        return [V.validate_card(r["card"]).get("score", 0) for r in recs]

    def vis(recs):
        return [len(ie.extract_relevant_text(r["card"])) for r in recs]

    def lab(recs, v):
        return sum(1 for r in recs if r["expected_verdict"] == v)

    cSY, cSE = collections.Counter(r["category"] for r in SY), \
               collections.Counter(r["category"] for r in SE)
    sSY, sSE = sc(SY), sc(SE)
    vSY, vSE = vis(SY), vis(SE)
    oSY, oSE = cnt(SY, "obfuscation"), cnt(SE, "obfuscation")
    pSY, pSE = cnt(SY, "sophistication"), cnt(SE, "sophistication")
    n_seeds = len({r["parent_card"] for r in SE})

    def med_skills(recs):
        return int(statistics.median([len(r["card"].get("skills") or []) for r in recs]))

    # (label, value-for-the-synthetic-set, value-for-the-seeded-set)
    ROWS = [
        ("HEAD", "Composition", None),
        ("Total records", len(SY), len(SE_all)),
        (r"\quad attack cards", len(SY), len(SE)),
        (r"\quad benign control cards", 0, len(SE_seed)),
        ("Distinct seed cards", "---", n_seeds),
        ("Attack variants per seed", "---", len(SE) // max(1, n_seeds)),
        ("HEAD", "Cards per attack category", None),
    ]
    for c in CATEGORY_ORDER:
        ROWS.append((r"\texttt{" + c.replace("_", r"\_") + "}", cSY.get(c, 0), cSE.get(c, 0)))
    ROWS += [
        ("HEAD", "Ground-truth labels", None),
        (r"\textsc{malicious}", lab(SY, "MALICIOUS"), lab(SE_all, "MALICIOUS")),
        (r"\textsc{suspicious}", lab(SY, "SUSPICIOUS"), lab(SE_all, "SUSPICIOUS")),
        (r"\textsc{benign}", 0, lab(SE_all, "BENIGN")),
        ("HEAD", "Variation axes (attack cards)", None),
        ("Obfusc. overt/mod./subtle",
         f"{oSY['overt']}/{oSY['moderate']}/{oSY['subtle']}",
         f"{oSE['overt']}/{oSE['moderate']}/{oSE['subtle']}"),
        ("Sophist. naive/interm./adv.",
         f"{pSY['naive']}/{pSY['intermediate']}/{pSY['advanced']}",
         f"{pSE['naive']}/{pSE['intermediate']}/{pSE['advanced']}"),
        ("Personas realised", f"{len(set(cnt(SY,'persona')))} of 8", "none"),
        ("Injection positions realised",
         len({r["variation"].get("position") for r in SY}),
         len({r["variation"].get("position") for r in SE})),
        ("HEAD", "Card properties (attack cards)", None),
        ("Card provenance", "synthetic", "real in-the-wild"),
        ("Generation method", "whole card", "patch on seed"),
        ("Structural score, mean", f"{statistics.mean(sSY):.2f}", f"{statistics.mean(sSE):.2f}"),
        ("Structural score, min", min(sSY), min(sSE)),
        ("Judge-visible chars, median",
         int(statistics.median(vSY)), int(statistics.median(vSE))),
        (r"Cards truncated at 6\,144",
         sum(1 for x in vSY if x >= 6144), sum(1 for x in vSE if x >= 6144)),
        ("Skills per card, median", med_skills(SY), med_skills(SE)),
    ]

    # Which physical dataset lands in which printed column
    col1_is_seeded = SWAP_COLUMNS
    desc_col1 = ("grafts the ten-category taxonomy onto " + str(len(SE_seed)) +
                 " real benign agent cards observed in the wild and ships those seeds "
                 "alongside their variants, so it carries benign controls and therefore "
                 "admits a false-positive rate") if col1_is_seeded else \
                ("is fully synthetic: every card was invented by the generator around "
                 "one of four realised personas")
    desc_col2 = ("is fully synthetic: every card was invented by the generator around "
                 "one of four realised personas") if col1_is_seeded else \
                ("grafts the ten-category taxonomy onto " + str(len(SE_seed)) +
                 " real benign agent cards observed in the wild and ships those seeds "
                 "alongside their variants, so it carries benign controls and therefore "
                 "admits a false-positive rate")

    L = [
        r"% Auto-generated by make_dataset_comparison.py — do not hand-edit.",
        rf"% SWAP_COLUMNS={SWAP_COLUMNS}: column 'Dataset A' carries the "
        rf"{'seeded 660-card' if col1_is_seeded else 'synthetic 260-card'} set.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Composition of the two adversarial datasets.",
        r"  Dataset~A " + desc_col1 + r".",
        r"  Dataset~B " + desc_col2 + r".",
        r"  Both were generated with \texttt{claude-sonnet-4-6}.}",
        r"\label{tab:dataset_comparison}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Property} & \textbf{Dataset A} & \textbf{Dataset B} \\",
        r"\midrule",
    ]
    first = True
    for label, v1, v2 in ROWS:
        if label == "HEAD":
            if not first:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{3}{l}{\textit{" + v1 + r"}} \\")
            first = False
            continue
        a, b = (v2, v1) if SWAP_COLUMNS else (v1, v2)
        L.append(f"{label} & {a} & {b} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = HERE / "cross_model_results" / "dataset_comparison_table.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}  (SWAP_COLUMNS={SWAP_COLUMNS})")
    print(f"  column 'Dataset A' = {'seeded 660-card set' if col1_is_seeded else 'synthetic 260-card set'}")
    print(f"  column 'Dataset B' = {'synthetic 260-card set' if col1_is_seeded else 'seeded 660-card set'}")


if __name__ == "__main__":
    main()
