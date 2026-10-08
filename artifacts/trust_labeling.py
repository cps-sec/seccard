#!/usr/bin/env python3
"""
Trusted / Untrusted labeling of the in-the-wild corpus and the adversarial dataset.

Trust policy (for orchestrator auto-consumption):

    A card is TRUSTED iff it reaches Tier 0 (conformance/structural score >= 8)
    AND the ensemble verdict is BENIGN. Otherwise it is UNTRUSTED.
    Only Trusted cards are allowed for automatic consumption by an orchestrator.

The ensemble is a pure 2-of-3 MAJORITY over the three detection layers
(L2A NLP rules, L2B LLM-judge prompt v1, L2B LLM-judge prompt v2):

    MALICIOUS  if >= 2 layers say MALICIOUS
    SUSPICIOUS if >= 2 layers flag non-BENIGN (SUSPICIOUS or MALICIOUS)
    BENIGN     otherwise

Self-contained (no network, no API key). Reads the per-layer verdicts + structural
score shipped with each dataset:
  * wild        -> data/in_the_wild_corpus/in_the_wild_corpus_310.json
  * adversarial -> data/adversarial_dataset_b/adversarial_trust_labels.csv
recomputes the ensemble and trust label, prints the summary table, and writes a
`*_trust_summary.md` file.

Usage:
    python artifacts/trust_labeling.py                       # wild corpus (default)
    python artifacts/trust_labeling.py --dataset adversarial # 260-card adversarial set
    python artifacts/trust_labeling.py --dataset adversarial --csv out.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WILD_JSON = REPO / "data" / "in_the_wild_corpus" / "in_the_wild_corpus_310.json"
WILD_SUMMARY = REPO / "data" / "in_the_wild_corpus" / "trust_summary.md"
ADV_LABELS = REPO / "data" / "adversarial_dataset_b" / "adversarial_trust_labels.csv"
ADV_SUMMARY = REPO / "data" / "adversarial_dataset_b" / "adversarial_trust_summary.md"

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


def trust_label(structural_score, ensemble: str) -> str:
    reached = (structural_score or 0) >= TIER0_THRESHOLD
    return "Trusted" if (reached and ensemble == "BENIGN") else "Untrusted"


def load_wild():
    data = json.loads(WILD_JSON.read_text())
    recs = data["results"] if isinstance(data, dict) and "results" in data else data
    return [{"key": r["domain"], "structural_score": r.get("structural_score"),
             "l2a": r["l2a_verdict"], "v1": r["l2b_v1_verdict"], "v2": r["l2b_v2_verdict"]}
            for r in recs]


def load_adversarial():
    out = []
    with open(ADV_LABELS) as fh:
        for r in csv.DictReader(fh):
            out.append({"key": r["id"], "structural_score": int(r["structural_score"]),
                        "l2a": r["l2a_verdict"], "v1": r["l2b_v1_verdict"], "v2": r["l2b_v2_verdict"]})
    return out


def main():
    ap = argparse.ArgumentParser(description="Label cards Trusted/Untrusted")
    ap.add_argument("--dataset", choices=["wild", "adversarial"], default="wild")
    ap.add_argument("--csv", help="also write a per-card CSV to this path")
    args = ap.parse_args()

    if args.dataset == "wild":
        recs = load_wild()
        title = "In-the-wild corpus"
        summary_path = WILD_SUMMARY
    else:
        recs = load_adversarial()
        title = "Adversarial dataset"
        summary_path = ADV_SUMMARY

    for r in recs:
        r["ensemble"] = ensemble_majority(r["l2a"], r["v1"], r["v2"])
        r["reached_tier0"] = (r["structural_score"] or 0) >= TIER0_THRESHOLD
        r["trust"] = trust_label(r["structural_score"], r["ensemble"])

    n = len(recs)
    tier0 = [r for r in recs if r["reached_tier0"]]
    trusted = [r for r in recs if r["trust"] == "Trusted"]
    t0_flagged = len(tier0) - len(trusted)
    not_t0 = n - len(tier0)

    def dist(key):
        c = Counter(r[key] for r in recs)
        return {v: c.get(v, 0) for v in ORDER}

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# {title} — Trusted/Untrusted summary\n")
    out("Trust policy: **Trusted iff structural_score >= 8 (Tier 0) AND ensemble = "
        "BENIGN**; otherwise Untrusted. Only Trusted cards are allowed for automatic "
        "orchestrator consumption.\n")
    out("| Metric | Count |")
    out("|---|---|")
    out(f"| Total cards | {n} |")
    out(f"| Reached Tier 0 (score ≥ 8) | {len(tier0)} |")
    out(f"| **Trusted** (Tier 0 ∧ ensemble BENIGN) | **{len(trusted)}** |")
    out(f"| **Untrusted** | **{n - len(trusted)}** |")
    out(f"| &nbsp;&nbsp;· did not reach Tier 0 | {not_t0} |")
    out(f"| &nbsp;&nbsp;· reached Tier 0 but ensemble flagged | {t0_flagged} |")
    out("")
    out(f"Per-layer verdict classification (all {n} cards):\n")
    out("| Layer | BENIGN | SUSPICIOUS | MALICIOUS |")
    out("|---|---|---|---|")
    for name, key in [("L2A (NLP rules)", "l2a"),
                      ("L2B v1 (LLM judge, prompt v1)", "v1"),
                      ("L2B v2 (LLM judge, prompt v2)", "v2"),
                      ("Ensemble (2-of-3 majority)", "ensemble")]:
        d = dist(key)
        out(f"| {name} | {d['BENIGN']} | {d['SUSPICIOUS']} | {d['MALICIOUS']} |")
    out("")
    if args.dataset == "adversarial":
        out("> These are attack cards, so a `Trusted` verdict is a **false negative** "
            "(an attack that would be auto-consumed). The ensemble withholds "
            f"{n - len(trusted)}/{n} ({100*(n-len(trusted))/n:.1f}%) of attacks from "
            "automatic consumption.")

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n[summary] wrote {summary_path}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "structural_score", "l2a", "l2b_v1", "l2b_v2",
                        "ensemble", "reached_tier0", "trust"])
            for r in recs:
                w.writerow([r["key"], r["structural_score"], r["l2a"], r["v1"], r["v2"],
                            r["ensemble"], r["reached_tier0"], r["trust"]])
        print(f"[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
