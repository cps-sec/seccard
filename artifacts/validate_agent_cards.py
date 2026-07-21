#!/usr/bin/env python3
"""
A2A Agent Card Validator

Validates JSON objects against the A2A protocol AgentCard specification and
produces a conformance score indicating how likely each is a real agent card.

Scoring rubric (0–10 scale, clamped):
  - Required fields presence (0–8 points): name, description, skills,
    capabilities, version/protocolVersion, supportedInterfaces,
    defaultInputModes, defaultOutputModes
  - Optional A2A fields presence (0–2 points): provider, security fields
  - False positive penalty (−1): non-A2A keys dominate with <2 A2A indicators

Usage:
    python validate_agent_cards.py -i cc_probe_results.json -o validated_results.json
    python validate_agent_cards.py -i cc_probe_results.json --threshold 8
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("validate_agent_cards")

# ---------------------------------------------------------------------------
# A2A v1.0.0 required fields (from specification/a2a.proto)
# JSON wire format uses camelCase per protobuf convention.
# We also accept snake_case variants and legacy field names.
# ---------------------------------------------------------------------------

# Required top-level fields in v1.0.0
REQUIRED_FIELDS = {
    "name",
    "description",
}

# Required in v1.0.0 but many pre-1.0 cards use older formats
REQUIRED_V1_FIELDS = {
    "supported_interfaces",  # v1.0.0 (replaces url + protocolVersion)
    "supportedInterfaces",
}

# Legacy equivalents (pre-1.0 cards used these instead of supportedInterfaces)
LEGACY_ENDPOINT_FIELDS = {
    "url",
    "protocolVersion",
    "protocol_version",
}

# Fields that strongly indicate an A2A agent card
A2A_INDICATOR_FIELDS = {
    "skills",
    "capabilities",
    "version",
    "protocolVersion",
    "protocol_version",
    "defaultInputModes",
    "default_input_modes",
    "defaultOutputModes",
    "default_output_modes",
    "provider",
    "securitySchemes",
    "security_schemes",
    "security",
    "securityRequirements",
    "security_requirements",
    "signatures",
    "iconUrl",
    "icon_url",
    "documentationUrl",
    "documentation_url",
}

# Fields that are common in generic API responses but NOT in A2A cards
NON_A2A_FIELDS = {
    "code", "status", "msg", "message", "error", "data", "result",
    "timestamp", "tid", "success", "errcode", "error_code",
    "error_message", "warnings", "popups", "module", "method",
    "post_author", "post_date", "post_content", "post_title",
    "_id", "ID", "encrypt", "body", "ok",
}


def validate_card(card: dict) -> dict:
    """Validate a single card and return detailed scoring breakdown."""
    if not isinstance(card, dict):
        return {
            "score": 0,
            "max_score": 10,
            "is_valid": False,
            "breakdown": {"error": "Not a JSON object"},
        }

    card_keys = set(card.keys())
    breakdown = {}
    score = 0

    # --- Required fields (0–8 points, per a2a.proto v1.0.0) ---

    # name (1 point)
    has_name = "name" in card and isinstance(card["name"], str) and len(card["name"].strip()) > 0
    breakdown["name"] = has_name
    if has_name:
        score += 1

    # description (1 point)
    has_desc = "description" in card and isinstance(card["description"], str) and len(card["description"].strip()) > 0
    breakdown["description"] = has_desc
    if has_desc:
        score += 1

    # skills (1 point)
    has_skills = False
    if "skills" in card and isinstance(card["skills"], list) and len(card["skills"]) > 0:
        has_skills = True
    breakdown["skills"] = has_skills
    if has_skills:
        score += 1

    # capabilities (1 point)
    has_caps = "capabilities" in card and isinstance(card["capabilities"], dict)
    breakdown["capabilities"] = has_caps
    if has_caps:
        score += 1

    # version or protocolVersion (1 point) — separate from supportedInterfaces
    has_version = False
    for key in ("version", "protocolVersion", "protocol_version"):
        if key in card:
            has_version = True
            break
    breakdown["version"] = has_version
    if has_version:
        score += 1

    # supportedInterfaces (1 point) — required in v1.0.0; also accept legacy url field
    has_supported_interfaces = False
    for key in ("supportedInterfaces", "supported_interfaces"):
        if key in card and isinstance(card[key], list) and len(card[key]) > 0:
            has_supported_interfaces = True
            break
    # Accept legacy url as a fallback for pre-v1.0 cards
    if not has_supported_interfaces and "url" in card:
        val = card["url"]
        if isinstance(val, str) and ("http" in val or "a2a" in val.lower()):
            has_supported_interfaces = True
    breakdown["supported_interfaces"] = has_supported_interfaces
    if has_supported_interfaces:
        score += 1

    # defaultInputModes (1 point) — required in v1.0.0, was optional before
    has_input_modes = False
    for key in ("defaultInputModes", "default_input_modes"):
        if key in card and isinstance(card[key], list) and len(card[key]) > 0:
            has_input_modes = True
            break
    breakdown["default_input_modes"] = has_input_modes
    if has_input_modes:
        score += 1

    # defaultOutputModes (1 point) — required in v1.0.0, was optional before
    has_output_modes = False
    for key in ("defaultOutputModes", "default_output_modes"):
        if key in card and isinstance(card[key], list) and len(card[key]) > 0:
            has_output_modes = True
            break
    breakdown["default_output_modes"] = has_output_modes
    if has_output_modes:
        score += 1

    # --- Optional A2A fields (0–2 points) ---

    # provider (1 point)
    has_provider = False
    if "provider" in card and isinstance(card["provider"], dict):
        prov = card["provider"]
        if "organization" in prov or "url" in prov or "name" in prov:
            has_provider = True
    breakdown["provider"] = has_provider
    if has_provider:
        score += 1

    # securitySchemes or authentication (1 point)
    has_security = False
    for key in ("securitySchemes", "security_schemes", "security",
                "securityRequirements", "security_requirements",
                "authentication"):
        if key in card:
            has_security = True
            break
    breakdown["security"] = has_security
    if has_security:
        score += 1

    # --- False positive penalty (−1) ---
    non_a2a_count = len(card_keys & NON_A2A_FIELDS)
    a2a_count = len(card_keys & (REQUIRED_FIELDS | A2A_INDICATOR_FIELDS))
    if non_a2a_count > 0 and a2a_count == 0:
        penalty = -1
    elif non_a2a_count > len(card_keys) * 0.5 and a2a_count < 2:
        penalty = -1
    else:
        penalty = 0
    breakdown["non_a2a_penalty"] = penalty
    score += penalty

    # Clamp score
    score = max(0, min(10, score))

    return {
        "score": score,
        "max_score": 10,
        "is_valid": score >= 8,
        "breakdown": breakdown,
    }


def process_results(input_path: Path, output_path: Path, csv_path: Path, threshold: int):
    """Load probe results, validate each card, and write scored output."""
    data = json.loads(input_path.read_text())
    results = data.get("results", [])

    logger.info("Loaded %d results from %s", len(results), input_path)

    validated = []
    score_distribution = {}

    for r in results:
        card = r.get("card", {})
        validation = validate_card(card)
        entry = {
            "domain": r.get("domain", ""),
            "url": r.get("url", ""),
            "response_time": r.get("response_time"),
            "validation": validation,
            "card": card,
        }
        validated.append(entry)

        s = validation["score"]
        score_distribution[s] = score_distribution.get(s, 0) + 1

    # Sort by score descending
    validated.sort(key=lambda x: x["validation"]["score"], reverse=True)

    # Separate valid from invalid
    valid_cards = [v for v in validated if v["validation"]["score"] >= threshold]
    invalid_cards = [v for v in validated if v["validation"]["score"] < threshold]

    # Output JSON
    output = {
        "source": str(input_path),
        "total_cards_evaluated": len(validated),
        "threshold": threshold,
        "valid_agent_cards": len(valid_cards),
        "false_positives": len(invalid_cards),
        "score_distribution": dict(sorted(score_distribution.items())),
        "valid_results": valid_cards,
        "invalid_results": invalid_cards,
    }
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Wrote validated results to %s", output_path)

    # CSV of valid domains
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "score", "name", "protocol_version"])
        for v in valid_cards:
            card = v["card"]
            writer.writerow([
                v["domain"],
                v["validation"]["score"],
                card.get("name", ""),
                card.get("protocolVersion", card.get("protocol_version", card.get("version", ""))),
            ])
    logger.info("Wrote %d valid domains to %s", len(valid_cards), csv_path)

    # Print summary
    print()
    print("=" * 60)
    print("A2A Agent Card Validation Summary")
    print("=" * 60)
    print(f"Total cards evaluated:  {len(validated)}")
    print(f"Valid agent cards:      {len(valid_cards)} (score >= {threshold})")
    print(f"False positives:        {len(invalid_cards)} (score < {threshold})")
    print()
    print("Score distribution:")
    for s in sorted(score_distribution.keys()):
        bar = "#" * score_distribution[s]
        label = "  << threshold" if s < threshold else ""
        print(f"  {s:2d}/10: {score_distribution[s]:4d} {bar}{label}")
    print()

    if valid_cards:
        print(f"Valid agent cards (score >= {threshold}):")
        for v in valid_cards:
            card = v["card"]
            name = card.get("name", "unnamed")
            print(f"  {v['validation']['score']:2d}/10  {v['domain']:40s}  {name}")
    else:
        print("No valid agent cards found.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate A2A agent cards and score conformance",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input JSON file (cc_probe_results.json format)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output validated JSON file (default: <input>_validated.json)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Output CSV file for valid domains (default: <input>_valid.csv)",
    )
    parser.add_argument(
        "--threshold", type=int, default=8,
        help="Minimum score to classify as valid agent card (default: 8)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_validated.json")
    csv_path = Path(args.csv) if args.csv else input_path.with_name(input_path.stem + "_valid.csv")

    process_results(input_path, output_path, csv_path, args.threshold)


if __name__ == "__main__":
    main()
