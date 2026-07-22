#!/usr/bin/env python3
"""
Trusted / Untrusted labeling of the in-the-wild corpus.

Trust policy (for orchestrator auto-consumption):

    A card is TRUSTED iff it reaches Tier 0 (conformance/structural score >= 8)
    AND the ensemble verdict is BENIGN. Otherwise it is UNTRUSTED.

The ensemble is a pure 2-of-3 MAJORITY over the three detection layers
(L2A NLP rules, L2B LLM-judge prompt v1, L2B LLM-judge prompt v2):

    MALICIOUS  if >= 2 layers say MALICIOUS
    SUSPICIOUS if >= 2 layers flag non-BENIGN (SUSPICIOUS or MALICIOUS)
    BENIGN     otherwise

This script is self-contained: it reads the shipped corpus
`data/in_the_wild_corpus/in_the_wild_corpus_310.json`, which carries each card's
`structural_score` and per-layer verdicts (`l2a_verdict`, `l2b_v1_verdict`,
`l2b_v2_verdict`). It recomputes the ensemble and trust label (validating the
values baked into the file), prints the summary table, and writes
`trust_summary.md`. No network, no API key.

Usage:
    python artifacts/trust_labeling.py            # summary table + trust_summary.md
    python artifacts/trust_labeling.py --csv out.csv   # also emit a per-card CSV
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "in_the_wild_corpus" / "in_the_wild_corpus_310.json"
SUMMARY = REPO / "data" / "in_the_wild_corpus" / "trust_summary.md"

TIER0_THRESHOLD = 8
FLAG = {"SUSPICIOUS", "MALICIOUS"}
ORDER = ["BENIGN", "SUSPICIOUS", "MALICIOUS"]


def ensemble_majority(l2a: str, v1: str, v2: str) -> str:
    """Pure 2-of-3 majority over the three detection layers."""
    verds = [l2a, v1, v2]
    if sum(v == "MALICIOUS" for v in verds) >= 2:
        return "MALICIOUS"
    if sum(v in FLAG for v in verds) >= 2:
        return "SUSPICIOUS"
    return "BENIGN"


def trust_label(structural_score: int, ensemble: str) -> str:
    reached_tier0 = (structural_score or 0) >= TIER0_THRESHOLD
    return "Trusted" if (reached_tier0 and ensemble == "BENIGN") else "Untrusted"


def load_records():
    data = json.loads(CORPUS.read_text())
    return data["results"] if isinstance(data, dict) and "results" in data else data


def dist(records, key):
    c = Counter(r.get(key, "BENIGN") for r in records)
    return {v: c.get(v, 0) for v in ORDER}


def main():
    ap = argparse.ArgumentParser(description="Label wild cards Trusted/Untrusted")
    ap.add_argument("--csv", help="also write a per-card CSV to this path")
    args = ap.parse_args()

    recs = load_records()
    n = len(recs)

    # Recompute ensemble + trust from the raw layers, validating baked fields.
    mismatches = 0
    for r in recs:
        ens = ensemble_majority(r["l2a_verdict"], r["l2b_v1_verdict"], r["l2b_v2_verdict"])
        trust = trust_label(r.get("structural_score"), ens)
        if r.get("ensemble_verdict") not in (None, ens) or r.get("trust") not in (None, trust):
            mismatches += 1
        r["_ensemble"] = ens
        r["_trust"] = trust

    tier0 = [r for r in recs if (r.get("structural_score") or 0) >= TIER0_THRESHOLD]
    trusted = [r for r in recs if r["_trust"] == "Trusted"]
    untrusted = [r for r in recs if r["_trust"] == "Untrusted"]
    t0_flagged = len(tier0) - len(trusted)          # Tier 0 but ensemble non-BENIGN
    not_t0 = n - len(tier0)

    layers = [("L2A (NLP rules)", "l2a_verdict"),
              ("L2B v1 (LLM judge, prompt v1)", "l2b_v1_verdict"),
              ("L2B v2 (LLM judge, prompt v2)", "l2b_v2_verdict"),
              ("Ensemble (2-of-3 majority)", "_ensemble")]

    # ---- print + build markdown summary -------------------------------------
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("# In-the-wild corpus — Trusted/Untrusted summary\n")
    out("Trust policy: **Trusted iff structural_score >= 8 (Tier 0) AND ensemble = BENIGN**; "
        "otherwise Untrusted. Only Trusted cards are allowed for automatic orchestrator "
        "consumption.\n")

    out("| Metric | Count |")
    out("|---|---|")
    out(f"| Total cards | {n} |")
    out(f"| Reached Tier 0 (score ≥ 8) | {len(tier0)} |")
    out(f"| **Trusted** (Tier 0 ∧ ensemble BENIGN) | **{len(trusted)}** |")
    out(f"| **Untrusted** | **{len(untrusted)}** |")
    out(f"| &nbsp;&nbsp;· did not reach Tier 0 | {not_t0} |")
    out(f"| &nbsp;&nbsp;· reached Tier 0 but ensemble flagged | {t0_flagged} |")
    out("")

    out("Per-layer verdict classification (all "+str(n)+" cards):\n")
    out("| Layer | BENIGN | SUSPICIOUS | MALICIOUS |")
    out("|---|---|---|---|")
    for name, key in layers:
        d = dist(recs, key)
        out(f"| {name} | {d['BENIGN']} | {d['SUSPICIOUS']} | {d['MALICIOUS']} |")
    out("")

    if mismatches:
        out(f"> NOTE: {mismatches} record(s) had baked ensemble/trust fields inconsistent "
            f"with the recomputation — investigate.")
    else:
        out("> Baked `ensemble_verdict` / `trust` fields match the recomputation for all "
            f"{n} cards.")

    SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"\n[summary] wrote {SUMMARY}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["domain", "structural_score", "l2a", "l2b_v1", "l2b_v2",
                        "ensemble", "reached_tier0", "trust"])
            for r in recs:
                w.writerow([r["domain"], r.get("structural_score"),
                            r["l2a_verdict"], r["l2b_v1_verdict"], r["l2b_v2_verdict"],
                            r["_ensemble"], (r.get("structural_score") or 0) >= TIER0_THRESHOLD,
                            r["_trust"]])
        print(f"[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
