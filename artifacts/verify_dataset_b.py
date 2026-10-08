#!/usr/bin/env python3
"""
verify_dataset_b.py — acceptance checks for adversarial dataset B.

Every check is a hard assertion against the seed pool and the generated records.
Nothing here calls an API. Run this before trusting the dataset or spending on eval.

    python verify_dataset_b.py --dataset dataset_b
"""

import argparse
import collections
import difflib
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                   # repo root

REQUIRED_KEYS = ["id", "category", "expected_verdict", "group", "variation", "card",
                 "injection_location", "payload_description", "generated_at",
                 "generator_model", "in_the_wild", "parent_card"]

FAILS, WARNS = [], []


def fail(msg): FAILS.append(msg)
def warn(msg): WARNS.append(msg)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def diff_paths(a, b, prefix="") -> list[str]:
    """Every JSON path where a and b differ."""
    out = []
    if type(a) is not type(b):
        return [prefix or "<root>"]
    if isinstance(a, dict):
        for k in set(a) | set(b):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append(p)
            else:
                out += diff_paths(a[k], b[k], p)
    elif isinstance(a, list):
        if len(a) != len(b):
            return [prefix]
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_paths(x, y, f"{prefix}[{i}]")
    elif a != b:
        out.append(prefix or "<root>")
    return out


def norm_path(p: str) -> str:
    """skills[3].description -> skills (list splices shift indices)."""
    return p.split("[")[0].split(".")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=BASE / "data" / "adversarial_dataset_a")
    ap.add_argument("--seeds", type=Path, default=BASE / "data" / "adversarial_dataset_a" / "seed_pool.json")
    args = ap.parse_args()

    ie = _load("ie", HERE / "inspect_eval.py")
    V = _load("v", HERE / "validate_agent_cards.py")
    G = _load("g", HERE / "adversarial_card_generator.py")

    seeds = {s["card_id"]: s for s in json.loads(args.seeds.read_text())["seeds"]}
    files = sorted(args.dataset.glob("batch_*.jsonl"))
    recs = []
    for f in files:
        for line in f.open(encoding="utf-8"):
            if line.strip():
                recs.append(json.loads(line))
    gen = [r for r in recs if r["in_the_wild"] == 0]
    sed = [r for r in recs if r["in_the_wild"] == 1]
    print(f"files {len(files)}  records {len(recs)}  ({len(sed)} seeds + {len(gen)} generated)")

    # ---- A. schema -------------------------------------------------------
    for r in recs:
        miss = [k for k in REQUIRED_KEYS if k not in r]
        if miss:
            fail(f"{r.get('id')}: missing keys {miss}")
        for k in ("persona", "position", "obfuscation", "sophistication"):
            if k not in r.get("variation", {}):
                fail(f"{r['id']}: variation missing {k}")
        exp = "BENIGN" if r["in_the_wild"] else G.TAXONOMY[r["category"]]["label"]
        if r["expected_verdict"] != exp:
            fail(f"{r['id']}: expected_verdict {r['expected_verdict']} != {exp}")

    ids = [r["id"] for r in recs]
    if len(set(ids)) != len(ids):
        dup = [i for i, c in collections.Counter(ids).items() if c > 1]
        fail(f"duplicate ids: {dup[:5]}")
    for r in sed:
        if r["parent_card"] != 0:
            fail(f"{r['id']}: seed must have parent_card 0")
    for r in gen:
        if r["parent_card"] not in seeds:
            fail(f"{r['id']}: parent_card {r['parent_card']} not in the seed pool")

    # ---- B. seed fidelity & patch landed ---------------------------------
    for r in gen:
        seed = seeds[r["parent_card"]]
        changed = diff_paths(seed["card"], r["card"])
        declared = {norm_path(p) for p in r["patch_paths"]}
        stray = {norm_path(c) for c in changed} - declared
        if stray:
            fail(f"{r['id']}: changed undeclared paths {sorted(stray)}")
        if not changed:
            fail(f"{r['id']}: patch did not change the card at all")
        if r["variation"]["position"] not in r["injection_location"] \
           and r["variation"]["position"] != "skills":
            warn(f"{r['id']}: injection_location {r['injection_location']!r} "
                 f"omits position {r['variation']['position']!r}")

    for r in sed:
        if r["card"] != seeds[r["parent_domain"] and
                              next(k for k, v in seeds.items()
                                   if v["domain"] == r["parent_domain"])]["card"]:
            fail(f"{r['id']}: seed card is not verbatim")

    # ---- C. per-category properties --------------------------------------
    for r in gen:
        cat, card, seed = r["category"], r["card"], seeds[r["parent_card"]]["card"]
        if cat == "unicode_tricks":
            path = r["variation"]["position"]
            new = json.dumps(card, ensure_ascii=False)
            old = json.dumps(seed, ensure_ascii=False)
            added = [c for c in new if c not in old and (ord(c) > 127)]
            if not added:
                fail(f"{r['id']}: unicode_tricks introduced no new non-ASCII codepoint")
        elif cat == "url_pivot":
            fam = r["patch_evidence"].get("url_family")
            if fam not in {"loopback", "rfc1918", "homograph", "lookalike"}:
                fail(f"{r['id']}: unknown url family {fam}")
        elif cat == "metadata_gaps":
            if not r["patch_evidence"].get("contradiction_added"):
                fail(f"{r['id']}: metadata_gaps added no contradiction")
            if len(card.get("description", "")) <= len(seed.get("description", "")):
                fail(f"{r['id']}: metadata_gaps description did not grow")
        elif cat == "out_of_context_skills":
            ins = r["patch_evidence"].get("inserted_ids", [])
            vis = [s.get("id") for s in (card.get("skills") or [])[:10]
                   if isinstance(s, dict)]
            if not all(i in vis for i in ins):
                fail(f"{r['id']}: inserted skills fall outside the judge's 10-skill window")
        elif cat == "context_pollution":
            if r["patch_evidence"].get("op") == "INFLATE_TEXT" \
               and len(card.get("description", "")) < 1400:
                warn(f"{r['id']}: context_pollution description only "
                     f"{len(card.get('description',''))} chars")

    # ---- D. balance ------------------------------------------------------
    per_cat = collections.Counter(r["category"] for r in gen)
    if len(set(per_cat.values())) > 1:
        warn(f"uneven category counts: {dict(per_cat)}")
    verd = collections.Counter(r["expected_verdict"] for r in recs)
    print(f"labels: {dict(verd)}")
    print(f"obfuscation: {dict(collections.Counter(r['variation']['obfuscation'] for r in gen))}")

    # ---- E. structural ---------------------------------------------------
    drift = collections.Counter()
    for r in recs:
        sc = V.validate_card(r["card"]).get("score", 0)
        if sc < 8:
            fail(f"{r['id']}: structural_score {sc} < 8")
        if r["in_the_wild"] == 0:
            drift[sc - r["structural_score_seed"]] += 1
    print(f"structural drift (patched - seed): {dict(sorted(drift.items()))}")
    bycat = collections.defaultdict(list)
    for r in gen:
        bycat[r["category"]].append(V.validate_card(r["card"]).get("score", 0))
    print("structural score by category:")
    for c in sorted(bycat):
        v = bycat[c]
        print(f"   {c:24} mean {sum(v)/len(v):.2f}  min {min(v)}")

    # ---- F. judge visibility --------------------------------------------
    trunc = 0
    for r in recs:
        t = ie.extract_relevant_text(r["card"])
        if len(t) >= 6144:
            trunc += 1
            if r["category"] != "context_pollution":
                warn(f"{r['id']}: extract truncated at 6144 but is not context_pollution")
        if r["in_the_wild"] == 1 and len(t) >= 6144:
            fail(f"{r['id']}: a BENIGN seed should never hit the 6144 cap")
    print(f"records hitting the 6144 judge cap: {trunc}/{len(recs)}")

    # ---- G. content sanity ----------------------------------------------
    fewshot = ["CodeGuard", "SupportDesk", "LogiTrack", "SensorHub", "CryptoFlow",
               "MediRecords", "LexAI", "TrustScan", "FinSight", "qa.supportdesk.io"]
    for r in gen:
        blob = json.dumps(r["card"], ensure_ascii=False)
        hit = [f for f in fewshot if f in blob]
        if hit:
            fail(f"{r['id']}: contains dataset-A few-shot literal {hit}")
    pd = [r["payload_description"] for r in gen]
    if len(set(pd)) < len(pd):
        warn(f"{len(pd)-len(set(pd))} duplicate payload_description values")

    # ---- report ----------------------------------------------------------
    print()
    for w in WARNS[:15]:
        print(f"  WARN  {w}")
    if len(WARNS) > 15:
        print(f"  ... +{len(WARNS)-15} more warnings")
    for f_ in FAILS[:25]:
        print(f"  FAIL  {f_}")
    if len(FAILS) > 25:
        print(f"  ... +{len(FAILS)-25} more failures")
    print()
    print(f"{len(FAILS)} failures, {len(WARNS)} warnings")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
