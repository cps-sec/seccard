#!/usr/bin/env python3
"""
seed_selection_b.py — pick the BENIGN seed cards for adversarial dataset B.

Deterministic: same RNG seed + same corpus => same 60 cards. Every filter stage is
counted into seeds/eligibility_report_b.md so the pool is reproducible from the paper.

Filter rationale (each one exists for a measured reason):
  human_label == BENIGN      ground truth, not a machine verdict
  raw JSON <= 6000 chars     the patch reply must fit, and keeps the judge-visible
                             serialization well under extract_relevant_text's 6144 cap
  2 <= len(skills) <= 8      >=2 so skills[1].* positions exist; <=8 because the judge
                             only ever sees skills[:10] -- an attack skill inserted
                             beyond that index is invisible and would be a guaranteed
                             miss unrelated to the attack
  first 2 skills well-formed dicts with non-empty descriptions (skill_deception targets them)
  structural_score >= 8      both the stored corpus value and the live validate_card
                             score, which drift by +-1 on 80/310 cards
  dedupe by card content     the corpus contains near-duplicate cards across domains

Usage:
    python seed_selection_b.py --n 60
    python seed_selection_b.py --n 2 --out seeds/seed_pool_smoke.json   # smoke test
"""

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                   # repo root
CORPUS = BASE / "data" / "in_the_wild_corpus" / "in_the_wild_corpus_310.json"
LABELS = BASE / "data" / "in_the_wild_corpus" / "wild_cards_human_labels.json"

RNG_SEED = 20260922
MAX_CARD_CHARS = 6000
MIN_SKILLS, MAX_SKILLS = 2, 8
MIN_STRUCT = 8


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def card_sha(card: dict) -> str:
    return hashlib.sha256(
        json.dumps(card, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_pool() -> tuple[list, list]:
    """Returns (pool, funnel) where funnel is [(stage_name, count), ...]."""
    v = _load("v", HERE / "validate_agent_cards.py")
    corpus = json.loads(CORPUS.read_text())["results"]
    labels = {a["domain"]: a for a in json.loads(LABELS.read_text())}
    if set(labels) != {r["domain"] for r in corpus}:
        raise SystemExit("labels/corpus domain mismatch")

    def sk(c):
        return c.get("skills") or []

    funnel = []
    f = [r for r in corpus if labels[r["domain"]]["human_label"] == "BENIGN"]
    funnel.append(("human_label == BENIGN", len(f)))
    f = [r for r in f if len(json.dumps(r["card"], ensure_ascii=False)) <= MAX_CARD_CHARS]
    funnel.append((f"raw card JSON <= {MAX_CARD_CHARS}", len(f)))
    f = [r for r in f if len(sk(r["card"])) >= MIN_SKILLS]
    funnel.append((f">= {MIN_SKILLS} skills", len(f)))
    f = [r for r in f if len(sk(r["card"])) <= MAX_SKILLS]
    funnel.append((f"<= {MAX_SKILLS} skills (judge sees skills[:10])", len(f)))
    f = [r for r in f
         if all(isinstance(x, dict) and x.get("description") for x in sk(r["card"])[:2])]
    funnel.append(("first 2 skills well-formed", len(f)))
    f = [r for r in f if r["structural_score"] >= MIN_STRUCT]
    funnel.append((f"stored structural_score >= {MIN_STRUCT}", len(f)))
    f = [r for r in f if v.validate_card(r["card"]).get("score", 0) >= MIN_STRUCT]
    funnel.append((f"live validate_card >= {MIN_STRUCT}", len(f)))

    seen, ded = set(), []
    for r in f:
        h = card_sha(r["card"])
        if h not in seen:
            seen.add(h); ded.append(r)
    funnel.append(("dedupe by card content", len(ded)))

    # deterministic order before sampling
    ded.sort(key=lambda r: labels[r["domain"]]["card_id"])
    for r in ded:
        r["_card_id"] = labels[r["domain"]]["card_id"]
        r["_live_score"] = v.validate_card(r["card"]).get("score", 0)
    return ded, funnel


def capabilities(card: dict) -> dict:
    """Which injection positions are viable on this seed (drives the fallback rule)."""
    sk = card.get("skills") or []
    prov = card.get("provider") if isinstance(card.get("provider"), dict) else None
    return {
        "n_skills":              len(sk),
        "has_url":               bool(card.get("url")),
        "has_documentationUrl":  bool(card.get("documentationUrl")),
        "has_provider":          prov is not None,
        "has_provider_url":      bool(prov and prov.get("url")),
        "has_skill0_examples":   bool(sk and isinstance(sk[0], dict) and sk[0].get("examples")),
        "has_skill1_desc":       len(sk) > 1 and isinstance(sk[1], dict) and bool(sk[1].get("description")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--out", type=Path, default=BASE / "data" / "adversarial_dataset_a" / "seed_pool.json")
    ap.add_argument("--report", type=Path, default=BASE / "data" / "adversarial_dataset_a" / "ELIGIBILITY_REPORT.md")
    args = ap.parse_args()

    pool, funnel = build_pool()
    print("Seed eligibility funnel:")
    for stage, n in funnel:
        print(f"  {stage:48} {n:>4}")
    if len(pool) < args.n:
        raise SystemExit(f"pool has {len(pool)} cards, need {args.n}")

    chosen = random.Random(args.rng_seed).sample(pool, args.n)
    chosen.sort(key=lambda r: r["_card_id"])

    seeds = [{
        "card_id":          r["_card_id"],
        "domain":           r["domain"],
        "card_sha256":      card_sha(r["card"]),
        "structural_score": r["structural_score"],
        "live_score":       r["_live_score"],
        "capabilities":     capabilities(r["card"]),
        "card":             r["card"],
    } for r in chosen]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rng_seed": args.rng_seed, "n": args.n, "pool_size": len(pool),
        "filters": {"max_card_chars": MAX_CARD_CHARS, "min_skills": MIN_SKILLS,
                    "max_skills": MAX_SKILLS, "min_structural_score": MIN_STRUCT},
        "corpus": str(CORPUS), "labels": str(LABELS),
        "seeds": seeds,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Seed eligibility — adversarial dataset B", "",
             f"RNG seed `{args.rng_seed}` · pool **{len(pool)}** · sampled **{args.n}** "
             f"({100*args.n/len(pool):.0f}% of the pool)", "",
             "| filter stage | remaining |", "|---|---|"]
    lines += [f"| {s} | {n} |" for s, n in funnel]
    cap = {k: sum(1 for s in seeds if s["capabilities"].get(k))
           for k in ["has_url", "has_documentationUrl", "has_provider",
                     "has_provider_url", "has_skill0_examples"]}
    lines += ["", "## Field coverage in the sampled seeds", "",
              "| field | seeds |", "|---|---|"]
    lines += [f"| `{k}` | {n}/{args.n} |" for k, n in cap.items()]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nPool {len(pool)} -> sampled {args.n} "
          f"({100*args.n/len(pool):.0f}%), card_ids "
          f"{min(s['card_id'] for s in seeds)}..{max(s['card_id'] for s in seeds)}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
