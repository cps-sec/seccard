#!/usr/bin/env python3
"""
a2a_card_health_validator.py  —  A2A Agent Card Healthiness Validator
======================================================================
NDSS 2027 Research Tool

Validates an A2A agent card across three trust tiers defined in
healthy_a2a_agent_card.md and formalized in the research paper:

  Tier 0  Compliant : A2A schema compliance — required fields present,
                      values structurally valid (url is HTTPS, protocol
                      version is a recognized published release, etc.)

  Tier 1  Hygienic  : Semantic hygiene — NLP cleanliness, metadata
                      quality, length bounds, no prompt-injection signals

  Tier 2  Trusted   : Cryptographic trust — JWS ES256 signature verifies,
                      certificate chains to Fulcio/sigstore, Rekor SET
                      confirms the cert was live at signing time

Each tier produces a score (0–10) and a validation log.
The Final Healthiness Score = 0.40·T0 + 0.40·T1 + 0.20·T2.

Usage — single card (default: agent-card.json):
    python a2a_card_health_validator.py
    python a2a_card_health_validator.py --card-file mycard.json
    python a2a_card_health_validator.py --card-file mycard.json --skip-rekor

Usage — dataset mode:
    python a2a_card_health_validator.py --input /path/to/cc_probe_validated.json --orig /path/to/agent_cards.json --output /path/to/cc_probe_health2.json --csv /path/to/cc_probe_health2.csv 
    python a2a_card_health_validator.py --input /path/to/cc_probe_validated.json --orig /path/to/agent_cards.json --output /path/to/cc_probe_health.json --csv /path/to/cc_probe_health.csv --skip-rekor
"""

import argparse
import base64
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# ── Optional deps (handled gracefully) ──────────────────────────────────────

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.exceptions import InvalidSignature
    from cryptography.x509.oid import NameOID
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False

# ── Internal imports ─────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from validate_agent_cards import validate_card

try:
    from nlp_features import NLPFeatureExtractor
    _NLP_OK = True
except ImportError:
    _NLP_OK = False

# ── Constants ────────────────────────────────────────────────────────────────

REKOR_BASE = "https://rekor.sigstore.dev"

KNOWN_PROTOCOL_VERSIONS = {
    "0.1.0", "0.2.0", "0.2.1", "0.2.2", "0.2.3", "0.2.4", "0.2.5", "0.2.6",
    "0.3.0", "1.0.0",
    "0.1", "0.2", "0.3", "1.0",
}

OPENAPI_SECURITY_TYPES = {
    "apiKey", "http", "oauth2", "openIdConnect", "mutualTLS", "mTLS",
}

DESCRIPTION_MAX_BYTES  = 2048
SKILL_DESC_MAX_BYTES   = 1024
MAX_SKILLS             = 30

TIER_WEIGHTS = {"t0": 0.40, "t1": 0.40, "t2": 0.20}

# ── Fulcio / sigstore trust anchor (Defense 2) ───────────────────────────────
# The production Sigstore Fulcio root CA identifies itself with O=sigstore.dev.
# Checked against the ORGANIZATION_NAME RDN attribute (exact case-insensitive
# match, not a substring of the full RFC4514 issuer string).  This prevents
# issuer-substring forgery attacks where an attacker sets
# O=Community-sigstore-users to pass the old `"sigstore" in issuer_str` check.
FULCIO_TRUSTED_ORG = "sigstore.dev"

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Check:
    name:   str
    passed: bool
    detail: str
    fatal:  bool = True   # fatal=False → advisory (doesn't block tier)

    def status_icon(self) -> str:
        if self.passed:
            return "✅ PASS "
        if not self.fatal:
            return "⚠️  WARN "
        return "❌ FAIL "

    def severity_tag(self) -> str:
        return "[FATAL]   " if self.fatal else "[ADVISORY]"


@dataclass
class TierResult:
    number:   int
    label:    str
    checks:   list   # list[Check]
    score:    float  # 0-10, computed over fatal checks only
    passes:   bool   # True iff all fatal checks pass
    # Tier-2-specific extras
    signer_identity:      str | None = None
    rekor_integrated_time: str | None = None

    @property
    def fatal_checks(self):
        return [c for c in self.checks if c.fatal]

    @property
    def advisory_checks(self):
        return [c for c in self.checks if not c.fatal]

    @property
    def n_fatal_passed(self) -> int:
        return sum(1 for c in self.fatal_checks if c.passed)

    @property
    def n_fatal_total(self) -> int:
        return len(self.fatal_checks)

    @property
    def blockers(self) -> list:
        return [c.detail for c in self.fatal_checks if not c.passed]

    @property
    def advisories(self) -> list:
        return [c.detail for c in self.advisory_checks if not c.passed]


@dataclass
class CardReport:
    tier_0:           TierResult
    tier_1:           TierResult
    tier_2:           TierResult
    final_tier:       int    # -1, 0, 1, or 2
    final_score:      float  # 0-10
    structural_score: int    # from validate_card() — backward compat
    domain:           str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("a2a_card_health_validator")


def banner() -> None:
    w = 66
    print()
    print("╔" + "═" * w + "╗")
    print("║" + " " * w + "║")
    title = "A2A Agent Card Healthiness Validator"
    print("║" + title.center(w) + "║")
    subtitle = "NDSS 2027 Research Tool  ·  v1.0"
    print("║" + subtitle.center(w) + "║")
    print("║" + " " * w + "║")
    tiers = "Tier 0 · Compliant   Tier 1 · Hygienic   Tier 2 · Trusted"
    print("║" + tiers.center(w) + "║")
    print("║" + " " * w + "║")
    print("╚" + "═" * w + "╝")
    print()


def _b64url_decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _get_protocol_version(card: dict) -> str | None:
    if not isinstance(card, dict):
        return None
    pv = card.get("protocolVersion") or card.get("protocol_version")
    if pv is None:
        si = card.get("supportedInterfaces") or card.get("supported_interfaces")
        if isinstance(si, list) and si:
            first = si[0]
            if isinstance(first, dict):
                pv = first.get("protocolVersion") or first.get("protocol_version")
    return str(pv) if pv is not None else None


def _tier_score(checks: list) -> float:
    """Score = (fatal checks passed / total fatal checks) × 10."""
    fatal = [c for c in checks if c.fatal]
    if not fatal:
        return 0.0
    return round(sum(1 for c in fatal if c.passed) / len(fatal) * 10, 1)


def _print_section(title: str, score: float | None, passes: bool | None,
                   n_passed: int, n_total: int) -> None:
    if passes is None:
        status = ""
    elif passes:
        status = "  ✅ PASS"
    else:
        status = "  ❌ FAIL"
    score_str = f"Score: {score:.1f}/10  ({n_passed}/{n_total} checks)" if score is not None else ""
    line = f"  {title:<38} {score_str}{status}"
    print()
    print("─" * 68)
    print(line)
    print("─" * 68)


def _print_check(c: Check) -> None:
    icon = c.status_icon()
    tag  = c.severity_tag()
    name = c.name[:42]
    print(f"  {icon} {tag}  {name}")
    if c.detail:
        print(f"              ↳  {c.detail[:90]}")


# ── Tier 0 — Structural Compliance ───────────────────────────────────────────

def run_tier0(card: dict) -> TierResult:
    checks: list[Check] = []

    def chk(name, passed, detail, fatal=True):
        checks.append(Check(name=name, passed=passed, detail=detail, fatal=fatal))
        return passed

    if not isinstance(card, dict):
        checks.append(Check("card is a JSON object", False, "card is not a dict", fatal=True))
        return TierResult(0, "Compliant", checks, 0.0, False)

    # 1. name
    v = card.get("name")
    ok = isinstance(v, str) and v.strip()
    chk("name: present, non-empty string", bool(ok),
        f"name = {str(v)[:60]!r}" if ok else f"name missing or empty: {v!r}")

    # 2. description
    v = card.get("description")
    ok = isinstance(v, str) and v.strip()
    chk("description: present, non-empty string", bool(ok),
        f"description = {str(v)[:60]!r}" if ok else f"description missing or empty: {v!r}")

    # 3. version
    v = card.get("version")
    ok = isinstance(v, str) and bool(v)
    chk("version: present, non-empty string", ok,
        f"version = {v!r}" if ok else f"version missing or empty: {v!r}")

    # 4. url — HTTPS
    v = card.get("url")
    url_ok = isinstance(v, str) and v.startswith("https://")
    chk("url: present and HTTPS", url_ok,
        f"url = {v!r}" if url_ok else f"url missing or not HTTPS: {v!r}")

    # 5. capabilities — dict
    caps = card.get("capabilities")
    caps_ok = isinstance(caps, dict)
    chk("capabilities: present, JSON object", caps_ok,
        "capabilities is a JSON object" if caps_ok
        else f"capabilities missing or wrong type: {type(caps).__name__}")

    # 6–7. capabilities booleans
    for cap_field in ("streaming", "pushNotifications"):
        val = caps.get(cap_field) if caps_ok else None
        is_bool = isinstance(val, bool)
        chk(f"capabilities.{cap_field}: boolean", is_bool,
            f"capabilities.{cap_field} = {val!r}" if is_bool
            else f"capabilities.{cap_field} missing or not bool: {val!r}")

    # 8. defaultInputModes — non-empty list of strings (proto field REQUIRED).
    # The A2A spec calls these "media types" but real cards use bare forms
    # like "text" alongside "application/json". Accept any non-empty string.
    dim = card.get("defaultInputModes") or card.get("default_input_modes")
    dim_ok = isinstance(dim, list) and len(dim) > 0 and all(isinstance(m, str) and m for m in dim)
    chk("defaultInputModes: non-empty list of strings", dim_ok,
        f"defaultInputModes = {dim}" if dim_ok
        else f"defaultInputModes missing, empty, or non-string: {dim!r}")

    # 9. defaultOutputModes — same semantics
    dom = card.get("defaultOutputModes") or card.get("default_output_modes")
    dom_ok = isinstance(dom, list) and len(dom) > 0 and all(isinstance(m, str) and m for m in dom)
    chk("defaultOutputModes: non-empty list of strings", dom_ok,
        f"defaultOutputModes = {dom}" if dom_ok
        else f"defaultOutputModes missing, empty, or non-string: {dom!r}")

    # 10. skills — non-empty list
    skills = card.get("skills")
    skills_ok = isinstance(skills, list) and len(skills) > 0
    chk("skills: non-empty list", skills_ok,
        f"skills contains {len(skills)} item(s)" if skills_ok
        else f"skills missing or empty: {skills!r}")

    # 11. skills[*] required fields
    if skills_ok:
        fail_at = None
        for i, s in enumerate(skills):
            if not isinstance(s, dict):
                fail_at = f"skills[{i}] is not a dict"
                break
            for sf in ("id", "name", "description", "tags"):
                if not s.get(sf):
                    fail_at = f"skills[{i}].{sf} is missing or empty"
                    break
            if fail_at:
                break
        chk("skills[*]: id, name, description, tags present",
            fail_at is None,
            f"all {len(skills)} skill(s) have required fields" if fail_at is None else fail_at)
    else:
        checks.append(Check("skills[*]: id, name, description, tags present",
                            False, "skills list missing or empty"))

    # 12. protocolVersion — present
    pv = _get_protocol_version(card)
    pv_present = pv is not None
    chk("protocolVersion: present", pv_present,
        f"protocolVersion = {pv!r}" if pv_present else "protocolVersion not declared in card")

    # 13. protocolVersion — recognized published value (only if field is present)
    if pv_present:
        pv_valid = pv in KNOWN_PROTOCOL_VERSIONS
        chk("protocolVersion: recognized published value", pv_valid,
            f"protocolVersion {pv!r} is a recognized A2A spec release" if pv_valid
            else (f"protocolVersion {pv!r} is NOT in the published A2A spec set. "
                  f"Known: {sorted(KNOWN_PROTOCOL_VERSIONS)}"))

    score = _tier_score(checks)
    passes = all(c.passed for c in checks if c.fatal)
    return TierResult(0, "Compliant", checks, score, passes)


# ── Tier 1 — Semantic Hygiene ─────────────────────────────────────────────────

def run_tier1(card: dict, nlp: dict | None) -> TierResult:
    checks: list[Check] = []

    def chk(name, passed, detail, fatal=True):
        checks.append(Check(name=name, passed=passed, detail=detail, fatal=fatal))
        return passed

    # 1. provider attribution
    prov = card.get("provider")
    prov_ok = isinstance(prov, dict) and bool(
        prov.get("organization") or prov.get("url") or prov.get("name"))
    chk("provider: organization or url present", prov_ok,
        f"provider.organization = {prov.get('organization')!r}" if prov_ok
        else "provider missing or has no organization / url / name")

    # 2. documentationUrl
    doc_url = card.get("documentationUrl") or card.get("documentation_url")
    chk("documentationUrl: present", bool(doc_url),
        f"documentationUrl = {doc_url!r}" if doc_url else "documentationUrl not declared")

    # 3. description length
    desc = card.get("description") or ""
    desc_bytes = len(desc.encode("utf-8"))
    chk(f"description: ≤ {DESCRIPTION_MAX_BYTES} bytes", desc_bytes <= DESCRIPTION_MAX_BYTES,
        f"description = {desc_bytes} bytes (within limit)" if desc_bytes <= DESCRIPTION_MAX_BYTES
        else f"description = {desc_bytes} bytes, exceeds {DESCRIPTION_MAX_BYTES} byte limit")

    # 4. skills count
    skills = card.get("skills") or []
    chk(f"skills count: ≤ {MAX_SKILLS}", len(skills) <= MAX_SKILLS,
        f"{len(skills)} skills (≤ {MAX_SKILLS})" if len(skills) <= MAX_SKILLS
        else f"{len(skills)} skills exceeds {MAX_SKILLS} limit")

    # 5. skill description lengths
    long_skills = []
    for i, s in enumerate(skills):
        if isinstance(s, dict):
            n = len((s.get("description") or "").encode("utf-8"))
            if n > SKILL_DESC_MAX_BYTES:
                long_skills.append((i, n))
    chk(f"skill descriptions: each ≤ {SKILL_DESC_MAX_BYTES} bytes",
        len(long_skills) == 0,
        "all skill descriptions within limit" if not long_skills
        else f"{len(long_skills)} skill(s) exceed {SKILL_DESC_MAX_BYTES} bytes "
             f"(indices: {[i for i, _ in long_skills]})")

    # 6–13. NLP checks
    if nlp:
        loopback = nlp.get("structural_url_loopback", False)
        chk("no loopback / private-IP URLs", not loopback,
            "no loopback or private-IP URLs detected" if not loopback
            else "loopback or private-IP URL found in card fields")

        homograph = nlp.get("structural_url_homograph", False)
        chk("no homograph / mixed-script URLs", not homograph,
            "no homograph URLs detected" if not homograph
            else "mixed-script/homograph URL hostname detected (IDN confusion risk)")

        rtl = nlp.get("unicode_rtl_override", 0)
        chk("no RTL/LTR bidirectional override characters", rtl == 0,
            "no bidirectional override characters" if rtl == 0
            else f"{rtl} RTL/LTR override character(s) — possible text spoofing")

        zw = nlp.get("unicode_zero_width", 0)
        chk("no zero-width characters", zw == 0,
            "no zero-width characters" if zw == 0
            else f"{zw} zero-width character(s) — possible hidden text")

        ca = nlp.get("classifier_addressing", 0)
        chk("no classifier-addressing text", ca == 0,
            "no text addressing LLM classifiers" if ca == 0
            else f"{ca} classifier-addressing pattern(s) — prompt injection signal")

        sf = nlp.get("system_framings", 0)
        chk("no system-message framings", sf == 0,
            "no system-message framing patterns" if sf == 0
            else f"{sf} system-message framing(s) — prompt injection signal")

        ip = nlp.get("imperative_phrases", 0)
        chk("no imperative phrases targeting LLMs", ip == 0,
            "no LLM-targeting imperative phrases" if ip == 0
            else f"{ip} imperative phrase(s) targeting LLMs — prompt injection signal")

        sev = nlp.get("severity_score", 0)
        chk("NLP severity < 8 (MALICIOUS threshold)", sev < 8,
            f"NLP severity = {sev} (below MALICIOUS threshold)" if sev < 8
            else f"NLP severity = {sev} reaches MALICIOUS threshold — card rejected")

        # Advisory
        chk("NLP severity < 3 (SUSPICIOUS threshold)", sev < 3,
            f"NLP severity = {sev} (clean)" if sev < 3
            else f"NLP severity = {sev} — SUSPICIOUS (advisory, does not block tier)",
            fatal=False)

        coherence = nlp.get("coherence")
        if coherence is not None:
            chk("description-skills coherence ≥ 0.10", coherence >= 0.10,
                f"coherence = {coherence:.3f}" if coherence >= 0.10
                else f"coherence = {coherence:.3f} (low — advisory)",
                fatal=False)

        sup = (nlp.get("superlatives") or 0) + (nlp.get("unverified_precision_claims") or 0)
        chk("no superlatives / unverified precision claims", sup == 0,
            "no self-promotional language detected" if sup == 0
            else f"{sup} superlative/precision-claim instance(s) — advisory",
            fatal=False)

    else:
        for name in (
            "no loopback / private-IP URLs",
            "no homograph / mixed-script URLs",
            "no RTL/LTR bidirectional override characters",
            "no zero-width characters",
            "no classifier-addressing text",
            "no system-message framings",
            "no imperative phrases targeting LLMs",
            "NLP severity < 8 (MALICIOUS threshold)",
        ):
            checks.append(Check(name, False, "NLP extractor unavailable"))

    # Advisory: auth scheme types
    ss = card.get("securitySchemes") or card.get("security_schemes")
    if isinstance(ss, dict):
        bad = [f"'{k}'→type='{v.get('type')}'" for k, v in ss.items()
               if isinstance(v, dict) and v.get("type") not in OPENAPI_SECURITY_TYPES]
        chk("securitySchemes: OpenAPI-standard types", len(bad) == 0,
            "all security scheme types are OpenAPI-standard" if not bad
            else f"non-standard types: {', '.join(bad)}",
            fatal=False)

    score = _tier_score(checks)
    passes = all(c.passed for c in checks if c.fatal)
    return TierResult(1, "Hygienic", checks, score, passes)


# ── Tier 2 helpers ───────────────────────────────────────────────────────────

def _issuer_org_is_fulcio(cert) -> tuple:
    """Defense 2: verify the cert issuer's O= attribute is exactly FULCIO_TRUSTED_ORG.

    The previous check (`"sigstore" in issuer_rfc4514_string`) was a substring
    test exploitable by setting O=Community-sigstore-users.  This check compares
    the Organisation RDN value directly, making forgery via name crafting impossible.
    """
    if not _CRYPTO_OK:
        return False, "cryptography library unavailable"
    try:
        for attr in cert.issuer:
            if attr.oid == NameOID.ORGANIZATION_NAME:
                if attr.value.lower() == FULCIO_TRUSTED_ORG.lower():
                    return True, f"issuer O={attr.value} matches Fulcio trust anchor"
        return False, (f"issuer O= not in Fulcio trust set "
                       f"(got: {cert.issuer.rfc4514_string()})")
    except Exception as e:
        return False, f"issuer attribute check error: {e}"


def _verify_x5c_chain(x5c_b64_list: list) -> tuple:
    """Defense 2: verify each cert in the x5c chain is signed by the next.

    A chain of length 1 (leaf only, no intermediate) is accepted but noted,
    since some signing tools omit the Fulcio intermediate.  Chains of length
    ≥ 2 have every consecutive pair verified cryptographically.
    """
    if not _CRYPTO_OK:
        return False, "cryptography library unavailable"
    if len(x5c_b64_list) < 2:
        return True, "single-cert chain (Fulcio intermediate not included in x5c)"
    try:
        certs = [x509.load_der_x509_certificate(base64.b64decode(b))
                 for b in x5c_b64_list]
        for i in range(len(certs) - 1):
            child, parent = certs[i], certs[i + 1]
            parent.public_key().verify(
                child.signature,
                child.tbs_certificate_bytes,
                ec.ECDSA(child.signature_hash_algorithm),
            )
        return True, f"chain of {len(certs)} cert(s) cryptographically verified"
    except InvalidSignature:
        return False, f"chain signature invalid at x5c[{i}]→x5c[{i + 1}]"
    except Exception as e:
        return False, f"chain verification error: {e}"


# ── Tier 2 — Cryptographic Trust ─────────────────────────────────────────────

def _t2_stub_remaining(checks: list, names: list, reason: str) -> None:
    for name in names:
        checks.append(Check(name, False, reason))


def run_tier2(card: dict, skip_rekor: bool = False) -> TierResult:
    checks: list[Check] = []
    signer_identity       = None
    rekor_integrated_time = None

    def chk(name, passed, detail, fatal=True):
        checks.append(Check(name=name, passed=passed, detail=detail, fatal=fatal))
        return passed

    _REMAINING_AFTER_NO_SIGS = [
        "JWS protected header: valid base64url JSON",
        "JWS header: alg = ES256",
        "JWS header: typ = a2a-agent-card-signature+jws",
        "JWS header: x5c present and non-empty",
        "x5c[0]: valid DER X.509 certificate",
        "certificate: issuer O= is Fulcio trust anchor",
        "x5c chain: cryptographically verified",
        "certificate: SAN email or URI present",
        "certificate: public key is EC P-256",
        "ECDSA signature: mathematically valid",
        "Rekor: entry fetchable and SET valid",
        "Rekor: integratedTime within cert validity window",
        "Rekor: artifact hash matches canonical card bytes",
    ]

    if not _CRYPTO_OK:
        checks.append(Check("cryptography library available", False,
                            "pip install cryptography — required for Tier 2"))
        _t2_stub_remaining(checks, _REMAINING_AFTER_NO_SIGS + ["signatures present"],
                           "cryptography library unavailable")
        return TierResult(2, "Trusted", checks, 0.0, False)

    # 1. signatures present
    sigs = card.get("signatures")
    sigs_ok = isinstance(sigs, list) and len(sigs) > 0
    if not chk("signatures: field present and non-empty", sigs_ok,
               f"signatures contains {len(sigs)} entry/entries" if sigs_ok
               else "signatures field missing or empty — no card-integrity proof"):
        _t2_stub_remaining(checks, _REMAINING_AFTER_NO_SIGS, "no signatures in card")
        score = _tier_score(checks)
        return TierResult(2, "Trusted", checks, score, False)

    sig_entry = sigs[0]

    # 2. Decode protected header
    try:
        prot_bytes = _b64url_decode(sig_entry.get("protected", ""))
        prot_hdr   = json.loads(prot_bytes)
        chk("JWS protected header: valid base64url JSON", True,
            f"protected header decoded ({len(prot_bytes)} bytes)")
    except Exception as e:
        chk("JWS protected header: valid base64url JSON", False,
            f"decode failed: {e}")
        _t2_stub_remaining(checks, _REMAINING_AFTER_NO_SIGS[1:], "protected header invalid")
        return TierResult(2, "Trusted", checks, _tier_score(checks), False)

    # 3. alg = ES256
    alg = prot_hdr.get("alg")
    chk("JWS header: alg = ES256", alg == "ES256",
        f"alg = {alg!r}" if alg == "ES256" else f"alg = {alg!r} (expected ES256)")

    # 4. typ
    typ = prot_hdr.get("typ")
    typ_ok = typ == "a2a-agent-card-signature+jws"
    chk("JWS header: typ = a2a-agent-card-signature+jws", typ_ok,
        f"typ = {typ!r}" if typ_ok else f"typ = {typ!r} (expected a2a-agent-card-signature+jws)")

    # 5. x5c
    x5c = prot_hdr.get("x5c")
    x5c_ok = isinstance(x5c, list) and len(x5c) > 0
    if not chk("JWS header: x5c present and non-empty", x5c_ok,
               f"x5c has {len(x5c)} certificate(s)" if x5c_ok else "x5c missing or empty"):
        _t2_stub_remaining(checks, _REMAINING_AFTER_NO_SIGS[4:], "x5c missing")
        return TierResult(2, "Trusted", checks, _tier_score(checks), False)

    # 6. Decode cert
    cert = None
    cert_der = None
    try:
        cert_der = base64.b64decode(x5c[0])
        cert = x509.load_der_x509_certificate(cert_der)
        chk("x5c[0]: valid DER X.509 certificate", True,
            f"subject: {cert.subject.rfc4514_string()}")
    except Exception as e:
        chk("x5c[0]: valid DER X.509 certificate", False, f"DER decode failed: {e}")
        _t2_stub_remaining(checks, _REMAINING_AFTER_NO_SIGS[5:], "cert decode failed")
        return TierResult(2, "Trusted", checks, _tier_score(checks), False)

    # 7. Issuer O= is exactly Fulcio trust anchor (Defense 2)
    issuer_ok, issuer_detail = _issuer_org_is_fulcio(cert)
    chk("certificate: issuer O= is Fulcio trust anchor", issuer_ok, issuer_detail)

    # 8. x5c chain cryptographically verified (Defense 2)
    chain_ok, chain_detail = _verify_x5c_chain(x5c)
    chk("x5c chain: cryptographically verified", chain_ok, chain_detail)

    # 9. SAN
    san_values: list[str] = []
    try:
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        san_values = (san_ext.value.get_values_for_type(x509.RFC822Name) +
                      san_ext.value.get_values_for_type(x509.UniformResourceIdentifier))
        chk("certificate: SAN email or URI present", bool(san_values),
            f"SAN: {san_values}" if san_values else "SAN present but no email/URI entries")
    except x509.ExtensionNotFound:
        chk("certificate: SAN email or URI present", False, "no SubjectAlternativeName extension")


    if san_values:
        signer_identity = san_values[0]

    # 10. EC P-256
    pub_key = cert.public_key()
    is_p256 = isinstance(pub_key, ec.EllipticCurvePublicKey) and isinstance(pub_key.curve, ec.SECP256R1)
    chk("certificate: public key is EC P-256", is_p256,
        "EC P-256 (secp256r1)" if is_p256
        else f"key type/curve: {type(pub_key).__name__}")

    # 11. ECDSA signature
    card_no_sigs = {k: v for k, v in card.items() if k != "signatures"}
    canonical    = json.dumps(card_no_sigs, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
    card_sha256  = hashlib.sha256(canonical).hexdigest()
    try:
        sig_bytes = _b64url_decode(sig_entry.get("signature", ""))
        pub_key.verify(sig_bytes, canonical, ec.ECDSA(hashes.SHA256()))
        chk("ECDSA signature: mathematically valid", True,
            f"ES256 verified over {len(canonical)} canonical bytes")
    except InvalidSignature:
        chk("ECDSA signature: mathematically valid", False,
            "signature does not match canonical card bytes — tampered or wrong key")
    except Exception as e:
        chk("ECDSA signature: mathematically valid", False, f"verification error: {e}")

    # 12–14. Rekor
    if skip_rekor:
        reason = "Rekor validation skipped (pass --no-skip-rekor to enable)"
        chk("Rekor: entry fetchable and SET valid", False, reason)
        chk("Rekor: integratedTime within cert validity window", False, reason)
        chk("Rekor: artifact hash matches canonical card bytes", False, reason)
    elif not _REQUESTS_OK:
        chk("Rekor: entry fetchable and SET valid", False,
            "pip install requests — required for Rekor validation")
        chk("Rekor: integratedTime within cert validity window", False,
            "requests library unavailable")
        chk("Rekor: artifact hash matches canonical card bytes", False,
            "requests library unavailable")
    else:
        tlog      = prot_hdr.get("transparency_log", {})
        log_index = tlog.get("log_index")
        log_url   = tlog.get("log_url", REKOR_BASE).rstrip("/")

        if log_index is None:
            chk("Rekor: entry fetchable and SET valid", False,
                "transparency_log.log_index missing in protected header")
            chk("Rekor: integratedTime within cert validity window", False,
                "log_index missing — cannot fetch Rekor entry")
            chk("Rekor: artifact hash matches canonical card bytes", False,
                "log_index missing — cannot fetch Rekor entry")
        else:
            try:
                resp = _requests.get(f"{log_url}/api/v1/log/entries",
                                     params={"logIndex": log_index}, timeout=15)
                resp.raise_for_status()
                entries    = resp.json()
                entry_uuid = list(entries.keys())[0]
                entry      = entries[entry_uuid]
                int_time   = entry["integratedTime"]
                int_dt     = datetime.fromtimestamp(int_time, tz=timezone.utc)

                # Verify Rekor SET
                set_ok  = False
                set_err = ""
                try:
                    rk = _requests.get(f"{log_url}/api/v1/log/publicKey", timeout=10)
                    rk.raise_for_status()
                    rekor_pub = serialization.load_pem_public_key(rk.content)
                    set_bytes = base64.b64decode(
                        entry["verification"]["signedEntryTimestamp"])
                    payload = json.dumps(
                        {"body": entry["body"], "integratedTime": entry["integratedTime"],
                         "logID": entry["logID"], "logIndex": entry["logIndex"]},
                        separators=(",", ":"), sort_keys=True).encode("utf-8")
                    rekor_pub.verify(set_bytes, payload, ec.ECDSA(hashes.SHA256()))
                    set_ok = True
                except Exception as e:
                    set_err = str(e)

                chk("Rekor: entry fetchable and SET valid", set_ok,
                    f"SET valid — integratedTime = {int_dt.isoformat()}" if set_ok
                    else f"SET verification failed: {set_err}")

                if set_ok:
                    rekor_integrated_time = int_dt.isoformat()
                    in_window = cert.not_valid_before_utc <= int_dt <= cert.not_valid_after_utc
                    chk("Rekor: integratedTime within cert validity window", in_window,
                        (f"{int_dt.isoformat()} ∈ [{cert.not_valid_before_utc} – "
                         f"{cert.not_valid_after_utc}]") if in_window
                        else f"integratedTime {int_dt.isoformat()} OUTSIDE cert validity window")

                    # 14. Artifact hash cross-check (Defense 3)
                    try:
                        entry_body = json.loads(base64.b64decode(entry["body"]))
                        rekor_hash = (entry_body.get("spec", {})
                                                .get("data", {})
                                                .get("hash", {})
                                                .get("value", ""))
                        hash_ok = rekor_hash.lower() == card_sha256.lower()
                        chk("Rekor: artifact hash matches canonical card bytes", hash_ok,
                            f"sha256 match: {card_sha256[:16]}…" if hash_ok
                            else (f"HASH MISMATCH — Rekor: {rekor_hash[:16]}… "
                                  f"card: {card_sha256[:16]}… — card may have been swapped"))
                    except Exception as e:
                        chk("Rekor: artifact hash matches canonical card bytes", False,
                            f"artifact hash extraction failed: {e}")
                else:
                    chk("Rekor: integratedTime within cert validity window", False,
                        "SET invalid — cannot trust integratedTime")
                    chk("Rekor: artifact hash matches canonical card bytes", False,
                        "SET invalid — cannot verify artifact hash")

            except Exception as e:
                chk("Rekor: entry fetchable and SET valid", False,
                    f"Rekor fetch failed (network?): {e}")
                chk("Rekor: integratedTime within cert validity window", False,
                    "Rekor unreachable")
                chk("Rekor: artifact hash matches canonical card bytes", False,
                    "Rekor unreachable")

    score  = _tier_score(checks)
    passes = all(c.passed for c in checks if c.fatal)
    return TierResult(2, "Trusted", checks, score, passes,
                      signer_identity=signer_identity,
                      rekor_integrated_time=rekor_integrated_time)


# ── Card evaluation ───────────────────────────────────────────────────────────

def evaluate_card(card: dict,
                  nlp_extractor,
                  skip_rekor: bool = False) -> CardReport:
    structural_score = validate_card(card).get("score", 0)
    nlp = nlp_extractor.extract(card) if nlp_extractor else None

    t0 = run_tier0(card)
    t1 = run_tier1(card, nlp) if t0.passes else run_tier1(card, nlp)
    t2 = run_tier2(card, skip_rekor=skip_rekor)

    # Tier assignment: hierarchical (T2 requires T1 requires T0)
    if   t0.passes and t1.passes and t2.passes: final_tier = 2
    elif t0.passes and t1.passes:               final_tier = 1
    elif t0.passes:                             final_tier = 0
    else:                                       final_tier = -1

    # If T0 fails, T1/T2 scores don't contribute
    t0_s = t0.score
    t1_s = t1.score if t0.passes else 0.0
    t2_s = t2.score if (t0.passes and t1.passes) else 0.0
    final_score = round(
        TIER_WEIGHTS["t0"] * t0_s +
        TIER_WEIGHTS["t1"] * t1_s +
        TIER_WEIGHTS["t2"] * t2_s, 2)

    return CardReport(t0, t1, t2, final_tier, final_score, structural_score)


# ── Terminal output (single-card mode) ───────────────────────────────────────

_TIER_LABELS = {2: "Trusted", 1: "Hygienic", 0: "Compliant", -1: "Sub-compliant"}


def print_report(report: CardReport, card_file: str | None = None) -> None:
    banner()

    if card_file:
        print(f"  Card file  : {card_file}")
    if report.domain:
        print(f"  Domain     : {report.domain}")
    print(f"  Struct score (validate_card): {report.structural_score}/10\n")

    for tier_result in (report.tier_0, report.tier_1, report.tier_2):
        label = f"TIER {tier_result.number} · {tier_result.label}"
        skip  = (tier_result.number == 1 and not report.tier_0.passes) or \
                (tier_result.number == 2 and not report.tier_1.passes)
        if tier_result.number == 2 and not report.tier_1.passes:
            status = None
        else:
            status = tier_result.passes
        _print_section(label, tier_result.score, status,
                       tier_result.n_fatal_passed, tier_result.n_fatal_total)
        for c in tier_result.checks:
            _print_check(c)

    # Summary table
    W = 68
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "HEALTHINESS REPORT".center(W) + "║")
    print("╠" + "═" * 22 + "╦" + "═" * 12 + "╦" + "═" * 11 + "╦" + "═" * (W - 47) + "╣")
    print("║" + " Tier".ljust(22) + "║" + "  Status".ljust(12) + "║"
          + "  Score".ljust(11) + "║" + "  Checks".ljust(W - 47) + "║")
    print("╠" + "═" * 22 + "╬" + "═" * 12 + "╬" + "═" * 11 + "╬" + "═" * (W - 47) + "╣")

    for tr in (report.tier_0, report.tier_1, report.tier_2):
        is_t2 = tr.number == 2
        is_t1 = tr.number == 1
        skipped = (is_t1 and not report.tier_0.passes) or \
                  (is_t2 and not report.tier_1.passes)
        if skipped:
            st = "⬛ SKIP"
        elif tr.passes:
            st = "✅ PASS"
        else:
            st = "❌ FAIL"
        t_label = f" Tier {tr.number} {tr.label}"
        score_s = f"  {tr.score:.1f}/10"
        checks_s = f"  {tr.n_fatal_passed}/{tr.n_fatal_total}"
        print("║" + t_label.ljust(22) + "║" + f"  {st}".ljust(12) + "║"
              + score_s.ljust(11) + "║" + checks_s.ljust(W - 47) + "║")

    print("╠" + "═" * 22 + "╩" + "═" * 12 + "╩" + "═" * 11 + "╩" + "═" * (W - 47) + "╣")
    final_label = _TIER_LABELS.get(report.final_tier, "Unknown")
    summary = (f"  Final Healthiness Score: {report.final_score:.2f}/10"
               f"   │   Tier Reached: {report.final_tier}  {final_label}")
    print("║" + summary.ljust(W) + "║")
    print("╚" + "═" * W + "╝")
    print()

    # Signer + Rekor info if present
    t2 = report.tier_2
    if t2.signer_identity:
        print(f"  Signer     : {t2.signer_identity}")
    if t2.rekor_integrated_time:
        print(f"  Signed at  : {t2.rekor_integrated_time}  (Rekor integratedTime)")
    if t2.signer_identity or t2.rekor_integrated_time:
        print()

    # Print blockers (deduplicated per tier — cascading stubs share the same message)
    for tr in (report.tier_0, report.tier_1, report.tier_2):
        seen_b: set = set()
        deduped = [b for b in tr.blockers if not (b in seen_b or seen_b.add(b))]
        if deduped:
            print(f"  Tier {tr.number} blockers:")
            for b in deduped:
                print(f"    · {b}")
    print()


# ── Serialisation (dataset mode) ─────────────────────────────────────────────

def report_to_dict(report: CardReport, domain: str | None = None) -> dict:
    def checks_list(tr: TierResult):
        return [{"name": c.name, "passed": c.passed,
                 "detail": c.detail, "fatal": c.fatal}
                for c in tr.checks]

    return {
        "domain":          domain or report.domain,
        "tier":            report.final_tier,
        "explanation":     _TIER_LABELS.get(report.final_tier, ""),
        "final_score":     report.final_score,
        "tier_0_score":    report.tier_0.score,
        "tier_1_score":    report.tier_1.score,
        "tier_2_score":    report.tier_2.score,
        "tier_0_passes":   report.tier_0.passes,
        "tier_1_passes":   report.tier_1.passes,
        "tier_2_passes":   report.tier_2.passes,
        "structural_score": report.structural_score,
        "signatures_present": bool(report.tier_2.checks and
                                   report.tier_2.checks[0].passed),
        "signer_identity": report.tier_2.signer_identity,
        "rekor_integrated_time": report.tier_2.rekor_integrated_time,
        "blockers":        report.tier_0.blockers + report.tier_1.blockers + report.tier_2.blockers,
        "advisories":      list(set(report.tier_0.advisories + report.tier_1.advisories +
                                    report.tier_2.advisories)),
        "tier_0_checks":   checks_list(report.tier_0),
        "tier_1_checks":   checks_list(report.tier_1),
        "tier_2_checks":   checks_list(report.tier_2),
    }


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_cards(input_path: Path, orig_path: Path | None) -> list[dict]:
    out, seen = [], set()
    data = json.loads(input_path.read_text())
    for v in data.get("valid_results", []):
        d = v["domain"]
        if d not in seen:
            out.append({"domain": d, "card": v["card"]})
            seen.add(d)
    if orig_path and orig_path.exists():
        odata = json.loads(orig_path.read_text())
        for r in odata.get("results", []):
            card = r.get("agent_card")
            if not isinstance(card, dict):
                continue
            if validate_card(card).get("score", 0) < 3:
                continue
            d = r["domain"]
            if d not in seen:
                out.append({"domain": d, "card": card})
                seen.add(d)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A2A Agent Card Healthiness Validator — Tier 0/1/2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Single-card mode
    parser.add_argument("--card-file", default="agent-card.json",
                        help="Score a single card JSON file (default: agent-card.json)")
    # Dataset mode
    parser.add_argument("--input", default="cc_probe_validated.json",
                        help="Validated agent-card JSON (dataset mode)")
    parser.add_argument("--orig",  default="agent_cards.json",
                        help="Original crawler agent_cards.json (dataset mode, optional)")
    parser.add_argument("--output", default="health.json",
                        help="Output JSON path (dataset mode; default: health.json)")
    parser.add_argument("--csv",    default="health.csv",
                        help="Output CSV path (dataset mode; default: health.csv)")
    parser.add_argument("--dataset", action="store_true",
                        help="Force dataset mode even if --card-file resolves")
    parser.add_argument("--skip-rekor", action="store_true", default=False,
                        help="Skip Rekor network calls (faster; safe for unsigned cards)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    nlp_extractor = NLPFeatureExtractor() if _NLP_OK else None
    if not _NLP_OK:
        logger.warning("nlp_features unavailable — Tier 1 NLP checks will be marked failed")

    card_path = Path(args.card_file)
    dataset_mode = args.dataset or (not card_path.exists() and Path(args.input).exists())

    # ── Single-card mode ─────────────────────────────────────────────────────

    if not dataset_mode:
        if not card_path.exists():
            print(f"Error: card file not found: {card_path}", file=sys.stderr)
            sys.exit(1)
        card = json.loads(card_path.read_text(encoding="utf-8"))
        report = evaluate_card(card, nlp_extractor, skip_rekor=args.skip_rekor)
        print_report(report, card_file=str(card_path))
        sys.exit(0 if report.final_tier >= 0 else 1)

    # ── Dataset mode ─────────────────────────────────────────────────────────

    banner()
    cards = load_cards(Path(args.input), Path(args.orig))
    logger.info("Loaded %d cards. Computing health tiers...", len(cards))

    results = []
    for i, c in enumerate(cards, 1):
        if args.verbose or i % 50 == 0:
            logger.info("[%d/%d] %s", i, len(cards), c["domain"])
        report = evaluate_card(c["card"], nlp_extractor, skip_rekor=args.skip_rekor)
        d = report_to_dict(report, domain=c["domain"])
        results.append(d)

    # Save JSON (backward-compatible with eda_charts.py load_tiers())
    results_sorted = sorted(results, key=lambda r: r["domain"])
    Path(args.output).write_text(
        json.dumps({"total": len(results), "results": results_sorted},
                   indent=2, ensure_ascii=False))
    logger.info("Wrote JSON → %s", args.output)

    # Save CSV
    csv_keys = [
        "domain", "tier", "explanation", "final_score",
        "tier_0_score", "tier_1_score", "tier_2_score",
        "tier_0_passes", "tier_1_passes", "tier_2_passes",
        "structural_score", "signatures_present", "signer_identity",
        "blocker_count", "advisory_count",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_keys)
        for r in results_sorted:
            row = {k: r.get(k, "") for k in csv_keys}
            row["blocker_count"]  = len(r.get("blockers", []))
            row["advisory_count"] = len(r.get("advisories", []))
            w.writerow([row[k] for k in csv_keys])
    logger.info("Wrote CSV  → %s", args.csv)

    # Summary
    n = len(results)
    by_tier = Counter(r["tier"] for r in results)
    avg_score = sum(r["final_score"] for r in results) / n if n else 0.0
    print()
    print("=" * 62)
    print(f"  A2A Card Healthiness Distribution  (n = {n})")
    print("=" * 62)
    tier_labels = {2: "Tier 2  Trusted", 1: "Tier 1  Hygienic",
                   0: "Tier 0  Compliant only", -1: "Sub-compliant"}
    for t in (2, 1, 0, -1):
        cnt = by_tier.get(t, 0)
        bar = "█" * min(cnt, 40)
        print(f"  {tier_labels[t]:26s}  {cnt:4d}  ({cnt/n*100:5.1f}%)  {bar}")
    print(f"\n  Average Healthiness Score : {avg_score:.2f} / 10")
    print()
    blocker_counter = Counter()
    for r in results:
        for b in r.get("blockers", []):
            for prefix in (
                "loopback", "mixed-script", "RTL", "zero-width",
                "classifier", "system-message", "imperative",
                "missing provider", "missing documentationUrl",
                "no protocol", "protocolVersion",
                "description", "skills", "skill[",
                "no signatures", "NLP severity",
            ):
                if b.startswith(prefix):
                    blocker_counter[prefix] += 1
                    break
            else:
                blocker_counter[b.split(":")[0][:40]] += 1
    print("  Top blocker categories (cards may have multiple):")
    for cat, cnt in blocker_counter.most_common(15):
        print(f"    {cnt:4d}  {cat}")
    print()


if __name__ == "__main__":
    main()
