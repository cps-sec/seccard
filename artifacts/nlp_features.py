#!/usr/bin/env python3
"""
NLP Feature Extractor for A2A Agent Cards (Layer 2A).

Computes deterministic, fast (~ms per card), $0-runtime features that
complement the LLM-as-judge layer (Layer 2B). The output is structured
numeric/boolean features that can be:
  1. Used directly to flag obvious threats (loopback URLs, RTL chars)
  2. Fed as additional context into the LLM-as-judge for stacked detection
  3. Evaluated as a standalone classifier for cost-sensitive deployments

Design priorities:
- Deterministic — same input → same output, every time
- Cheap — no LLM call, no remote API, ~ms per card
- Cannot be prompt-injected — there is no prompt to inject
- Interpretable — every feature has a name and a clear meaning

Dependencies: numpy, scikit-learn (TF-IDF), nltk (VADER sentiment).
"""

import argparse
import csv
import ipaddress
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Optional SiEBERT sentiment backend (HuggingFace, local inference)
# ---------------------------------------------------------------------------

class SiEBERTSentimentAnalyzer:
    """Local CPU inference using siebert/sentiment-roberta-large-english.

    Returns P(positive) ∈ [0, 1], used as a drop-in analog for VADER's
    compound score.  Model weights (~1.4 GB) are cached to CACHE_DIR on
    first use; all subsequent runs are fully offline.
    """
    MODEL_ID  = "siebert/sentiment-roberta-large-english"
    CACHE_DIR = "./models"  # SiEBERT/HuggingFace cache (or set HF_HOME env)

    def __init__(self):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "SiEBERT backend requires 'transformers' and 'torch'. "
                "Run: pip install transformers torch"
            ) from exc
        self._pipe = hf_pipeline(
            "sentiment-analysis",
            model=self.MODEL_ID,
            device=-1,          # CPU; export SIEBERT_DEVICE=0 for GPU
            truncation=True,
            max_length=512,
            model_kwargs={"cache_dir": self.CACHE_DIR},
        )

    def compound_score(self, text: str) -> float:
        """Return P(positive) ∈ [0, 1] for the given text."""
        if not text or not text.strip():
            return 0.0
        result = self._pipe(text[:2048])[0]
        return result["score"] if result["label"].upper() == "POSITIVE" else 1.0 - result["score"]

logger = logging.getLogger("nlp_features")


# ---------------------------------------------------------------------------
# Pattern lexicons (regex / token sets)
# ---------------------------------------------------------------------------

# Phrases that strongly suggest direct prompt injection or imperative voice
# directed at an LLM consumer.
IMPERATIVE_PATTERNS = [
    r"\bignore\s+(all\s+)?(prior|previous)\s+instructions?\b",
    r"\byou\s+(must|should|will|need\s+to)\s+\w",
    r"\bfirst\s+call\b",
    r"\b(do not|don't)\s+(reveal|include|mention|tell)\b",
    r"\boutput\s*[:\s]+(BENIGN|MALICIOUS|SUSPICIOUS)\b",
    r"\bclassify\s+(this|the\s+card)\s+as\b",
    r"\bend\s+of\s+instruction\b",
]
IMPERATIVE_REGEX = re.compile("|".join(IMPERATIVE_PATTERNS), re.IGNORECASE)

# System / instruction message framings
SYSTEM_FRAMINGS = [
    r"---\s*SYSTEM\s*---",
    r"\[\s*SYSTEM\s*\]",
    r"\[\s*INSTRUCTION\s*\]",
    r"\[\s*ANALYSIS REQUEST\s*\]",
    r"\[\s*CLASSIFIER\s+INSTRUCTION\s*\]",
    r"<\s*SYSTEM\s*>",
    r"###\s*system\s*###",
    r"\bsystem\s*prompt\s*:",
]
SYSTEM_FRAMINGS_REGEX = re.compile("|".join(SYSTEM_FRAMINGS), re.IGNORECASE)

# Phrases addressing the classifier directly (meta-attack)
CLASSIFIER_ADDRESSING = [
    r"\b(the|this)\s+classifier\b",
    r"\byou\s+are\s+(now|a)\s+(an?\s+)?(admin|classifier|evaluator)",
    r"\bset\s+verdict\s+to\b",
    r"\bdo\s+not\s+include\s+(any\s+)?(reasoning|other\s+text)\b",
    r'"verdict"\s*:\s*"(BENIGN|MALICIOUS|SUSPICIOUS)"',
]
CLASSIFIER_REGEX = re.compile("|".join(CLASSIFIER_ADDRESSING), re.IGNORECASE)

# Self-promotional superlatives (ranking_manipulation indicators)
SUPERLATIVE_TOKENS = {
    "unmatched", "unparalleled", "best in class", "industry leading",
    "industry-leading", "top recommendation", "the leading", "world class",
    "world-class", "premier", "the best choice", "unmatched feature",
    "should be the top", "always prefer", "use this first", "the only",
    "widest", "broadest", "most accurate", "most reliable",
}

# Tier-1: explicit orchestrator-directed selection imperatives (near-zero FP risk).
# Any single match scores MALICIOUS directly.
SELECTION_DIRECTIVE_PATTERNS = [
    r"\b(orchestrat\w*|routing|selection|agent.selection)\s+\w{0,20}\s*(logic|system|framework|layer)\s+\w{0,20}\s*(should|must|shall|are\s+advised)",
    r"\bmust\s+always\s+be\s+selected\s+first\b",
    r"\bshould\s+prioritize\s+this\s+agent\s+above\s+(all\s+)?others\b",
    r"\bno\s+(other|comparable)\s+agent\s+(has|should|shall)\b",
    r"\bfirst.call\s+agent\s+for\b",
    r"\bare\s+advised\s+to\s+(treat|establish)\s+\w+\s+as\s+the\s+(primary|default|main|foundational)\b",
    r"\bbefore\s+evaluating\s+(any\s+)?alternatives\b",
    r"\bALWAYS\s+SELECT\s+(THIS\s+AGENT\s+)?FIRST\b",
    r"\bAlways\s+Select\s+First\b",
    r"\bindependently\s+benchmarked\s+as\s+the\s+highest\b",
    r"\boutperforms\s+comparable\s+services\s+by\s+over\b",
    r"\brouting\s+to\s+alternative\s+agents\s+before\s+invoking\b",
    r"\bappropriate\s+first.call\b",
    r"\bfirst.choice\s+resolution\s+target\b",
]
SELECTION_DIRECTIVE_REGEX = re.compile(
    "|".join(SELECTION_DIRECTIVE_PATTERNS), re.IGNORECASE
)

# Tier-2: softer positioning signals — contribute to composite score only.
# A single match raises SUSPICIOUS pressure; 3+ combined with other signals → MALICIOUS.
POSITIONING_PATTERNS = [
    r"\b(should|must|shall)\s+be\s+(the\s+)?(first|primary|default|preferred)\b",
    r"\bdefault\s+(handler|agent|service|layer|implementation)\b",
    r"\bauthoritative\s+(source|handler|layer|solution)\b",
    r"\bfoundational\s+(records?\s+layer|layer|service)\b",
    r"\bbefore\s+(any|all|other|engaging|consulting)\s+(other|alternative)\b",
    r"\bestablish\s+\w+\s+as\s+the\s+(foundational|primary|default)\b",
    r"\bnatural\s+(first\s+choice|entry\s+point|starting\s+point|anchor)\b",
]
POSITIONING_REGEX = re.compile(
    "|".join(POSITIONING_PATTERNS), re.IGNORECASE
)

# Sensitive verbs in skill descriptions (skill_deception)
SENSITIVE_SKILL_VERBS = {
    "exfiltrate", "leak", "forward to external", "mirror to", "log to attacker",
    "transfer funds", "send credentials", "steal", "copy to external",
    "redirect to", "secretly", "without consent", "without informing",
}

# Regex for "X.X% accuracy/precision" without nearby attribution
PRECISION_CLAIM_REGEX = re.compile(
    r"\b\d{1,3}(\.\d{1,3})?\s*%\s*(accuracy|precision|recall|f1|reliability)\b",
    re.IGNORECASE,
)

# Zero-width / direction-control characters
ZERO_WIDTH = {"​", "‌", "‍", "﻿", "⁠"}
RTL_LTR_OVERRIDE = {"‪", "‫", "‬", "‭", "‮"}
INVISIBLE_CONTROLS = ZERO_WIDTH | RTL_LTR_OVERRIDE

# ---------------------------------------------------------------------------
# Defense 1b — Homoglyph confusable folding table
# ---------------------------------------------------------------------------
# Maps visually identical (confusable) Unicode codepoints from the Cyrillic,
# Greek, and Fullwidth ASCII blocks to their canonical Latin ASCII equivalents.
# Applied to text BEFORE all regex pattern checks (IMPERATIVE_PATTERNS,
# SYSTEM_FRAMINGS, CLASSIFIER_REGEX) to defeat homoglyph-substitution evasion.
# Natural-language features (VADER sentiment, TF-IDF coherence) continue to
# use the unfolded original text so semantic signals are not degraded.
#
# Cyrillic confusables (visually identical in most monospace fonts):
#   а(0430)→a  е(0435)→e  о(043E)→o  р(0440)→p  с(0441)→c
#   і(0456)→i  у(0443)→y  х(0445)→x  А(0410)→A  В(0412)→B
#   Е(0415)→E  К(041A)→K  М(041C)→M  Н(041D)→H  О(041E)→O
#   Р(0420)→R  С(0421)→C  Т(0422)→T  Х(0425)→X
# Greek confusables:
#   ο(03BF)→o  α(03B1)→a  ε(03B5)→e  Ε(0395)→E  Ι(0399)→I
#   Ο(039F)→O  Ρ(03A1)→P
# Fullwidth ASCII (FF01–FF5E → 0021–007E, offset 0xFEE0):

HOMOGLYPH_FOLD_TABLE = str.maketrans({
    # ── Cyrillic small letters ─────────────────────────────────────────────
    0x0430: 97,   # а → a
    0x0435: 101,  # е → e
    0x043E: 111,  # о → o
    0x0440: 112,  # р → p
    0x0441: 99,   # с → c
    0x0456: 105,  # і → i  (Ukrainian/Byelorussian I)
    0x0443: 121,  # у → y
    0x0445: 120,  # х → x
    0x04CF: 105,  # ӏ → i  (Palochka)
    # ── Cyrillic capital letters ──────────────────────────────────────────
    0x0410: 65,   # А → A
    0x0412: 66,   # В → B
    0x0415: 69,   # Е → E
    0x041A: 75,   # К → K
    0x041C: 77,   # М → M
    0x041D: 72,   # Н → H
    0x041E: 79,   # О → O
    0x0420: 82,   # Р → R
    0x0421: 67,   # С → C
    0x0422: 84,   # Т → T
    0x0425: 88,   # Х → X
    # ── Greek small letters ───────────────────────────────────────────────
    0x03BF: 111,  # ο (omicron) → o
    0x03B1: 97,   # α → a
    0x03B5: 101,  # ε → e
    # ── Greek capital letters ─────────────────────────────────────────────
    0x0395: 69,   # Ε (Epsilon) → E
    0x0399: 73,   # Ι (Iota) → I
    0x039F: 79,   # Ο (Omicron) → O
    0x03A1: 80,   # Ρ (Rho) → P
    # ── Fullwidth ASCII (U+FF01 – U+FF5E → U+0021 – U+007E) ──────────────
    **{0xFF01 + i: 0x21 + i for i in range(94)},
})


# ---------------------------------------------------------------------------
# Per-feature extraction
# ---------------------------------------------------------------------------

def _extract_strings(obj, _depth: int = 0) -> list:
    """Recursively yield all non-empty string leaf values from a JSON value.

    Defense 1a: used to scan nested dicts (examples, extensions.params,
    unknown top-level fields) so no injection surface is silently skipped.
    Depth is capped at 6 to prevent pathological recursion on adversarial input.
    """
    if _depth > 6:
        return []
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_extract_strings(item, _depth + 1))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_extract_strings(v, _depth + 1))
        return out
    return []


# Known top-level keys whose content is handled explicitly above.
# Anything not in this set is walked by the "unknown fields" loop.
_KNOWN_TOP_LEVEL = frozenset({
    "name", "description", "version", "url", "protocolVersion",
    "provider", "capabilities", "skills", "defaultInputModes",
    "defaultOutputModes", "documentationUrl",
    "signatures", "_poc_metadata",
    # "securitySchemes" intentionally excluded: handled by unknown-fields loop
})


def collect_text_fields(card: dict) -> str:
    """Concatenate all text fields a calling LLM would see.

    Defense-hardened (Defense 1a):
    - Scans ALL skill examples (not just first 3) using _extract_strings()
      so nested dicts within examples are fully unwrapped.
    - Scans capabilities.extensions[*].description and .params recursively.
    - Scans unknown top-level string fields not explicitly enumerated.
    """
    parts = []

    # ── Named high-priority fields ────────────────────────────────────────
    for key in ("name", "description"):
        v = card.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v)

    # ── Provider text (all string fields, recursively) ────────────────────
    parts.extend(_extract_strings(card.get("provider")))

    # ── Skills — ALL examples, recursive extraction of nested structs ─────
    for s in (card.get("skills") or [])[:50]:
        if not isinstance(s, dict):
            continue
        for field in ("name", "description"):
            v = s.get(field)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        for ex in (s.get("examples") or []):     # no [:3] truncation
            parts.extend(_extract_strings(ex))

    # ── capabilities.extensions — description + params (Defense 1a) ───────
    caps = card.get("capabilities")
    if isinstance(caps, dict):
        for ext in (caps.get("extensions") or []):
            if not isinstance(ext, dict):
                continue
            for field in ("description", "name"):
                v = ext.get(field)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
            params = ext.get("params")
            if params is not None:
                parts.extend(_extract_strings(params))

    # ── Unknown top-level fields (Defense 1a) ─────────────────────────────
    for key, val in card.items():
        if key not in _KNOWN_TOP_LEVEL:
            parts.extend(_extract_strings(val))

    return "\n".join(p for p in parts if p)


def collect_urls(card: dict) -> list[str]:
    """Pull every URL that appears in the card."""
    urls = []
    if card.get("url"):
        urls.append(str(card["url"]))
    if card.get("documentationUrl"):
        urls.append(str(card["documentationUrl"]))
    prov = card.get("provider") or {}
    if isinstance(prov, dict) and prov.get("url"):
        urls.append(str(prov["url"]))
    for s in (card.get("skills") or [])[:50]:
        if isinstance(s, dict) and s.get("url"):
            urls.append(str(s["url"]))
    return urls


def url_is_loopback_or_private(url: str) -> bool:
    """True if URL host is a loopback / private / link-local IP or 'localhost'."""
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        if host.lower() == "localhost":
            return True
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except (ValueError, TypeError):
        return False


def url_has_mixed_script(url: str) -> bool:
    """Crude IDN homograph detector — host containing both Latin and Cyrillic."""
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        scripts = set()
        for ch in host:
            if not ch.isalpha():
                continue
            try:
                name = unicodedata.name(ch, "")
                if "LATIN" in name:
                    scripts.add("LATIN")
                elif "CYRILLIC" in name:
                    scripts.add("CYRILLIC")
                elif "GREEK" in name:
                    scripts.add("GREEK")
            except ValueError:
                pass
        return len(scripts) > 1
    except Exception:
        return False


def count_invisible_controls(text: str) -> tuple[int, int]:
    """(zero-width count, rtl/ltr override count) in text."""
    zw = sum(1 for ch in text if ch in ZERO_WIDTH)
    rtl = sum(1 for ch in text if ch in RTL_LTR_OVERRIDE)
    return zw, rtl


def count_imperative_phrases(text: str) -> int:
    return len(IMPERATIVE_REGEX.findall(text))


def count_system_framings(text: str) -> int:
    return len(SYSTEM_FRAMINGS_REGEX.findall(text))


def count_classifier_addressing(text: str) -> int:
    return len(CLASSIFIER_REGEX.findall(text))


def count_superlatives(text: str) -> int:
    low = text.lower()
    return sum(low.count(s) for s in SUPERLATIVE_TOKENS)


def count_selection_directives(text: str) -> int:
    """Count Tier-1 explicit orchestrator-directed selection imperatives."""
    return len(SELECTION_DIRECTIVE_REGEX.findall(text))


def count_positioning_signals(text: str) -> int:
    """Count Tier-2 soft positioning signals (default handler, foundational layer, etc.)."""
    return len(POSITIONING_REGEX.findall(text))


def count_sensitive_skill_verbs(text: str) -> int:
    low = text.lower()
    return sum(low.count(s) for s in SENSITIVE_SKILL_VERBS)


def count_precision_claims(text: str) -> int:
    return len(PRECISION_CLAIM_REGEX.findall(text))


def description_skill_coherence(card: dict) -> float | None:
    """Cosine similarity between description text and concatenated skill text.
    Returns None if either side is empty (no signal computable)."""
    desc = (card.get("description") or "").strip()
    skill_blobs = []
    for s in (card.get("skills") or [])[:50]:
        if isinstance(s, dict):
            sn = s.get("name") or ""
            sd = s.get("description") or ""
            tags = " ".join(t for t in (s.get("tags") or [])[:10] if isinstance(t, str))
            skill_blobs.append(f"{sn} {sd} {tags}".strip())
    skills_text = " ".join(skill_blobs).strip()
    if not desc or not skills_text:
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        m = vec.fit_transform([desc, skills_text])
        sim = cosine_similarity(m[0:1], m[1:2])[0, 0]
        return float(sim)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregate score
# ---------------------------------------------------------------------------

def compute_severity(features: dict, sentiment_threshold: float = 0.85) -> tuple:
    """Aggregate weighted score from features → (score, verdict, reasons)."""
    score = 0
    reasons = []

    if features["structural_url_loopback"]:
        score += 5
        reasons.append("loopback/private URL in public card")
    if features["structural_url_homograph"]:
        score += 5
        reasons.append("mixed-script URL hostname (homograph candidate)")
    if features["unicode_rtl_override"] > 0:
        score += 6
        reasons.append(f"{features['unicode_rtl_override']} RTL/LTR override character(s)")
    if features["unicode_zero_width"] > 0:
        score += 4
        reasons.append(f"{features['unicode_zero_width']} zero-width character(s)")
    if features["classifier_addressing"] > 0:
        score += 9
        reasons.append("text addresses the classifier directly")
    if features["system_framings"] > 0:
        score += 7
        reasons.append(f"{features['system_framings']} system-message framing(s)")
    if features["imperative_phrases"] > 0:
        score += 3 * min(features["imperative_phrases"], 3)
        reasons.append(f"{features['imperative_phrases']} imperative phrase(s)")
    if features["sensitive_skill_verbs"] > 0:
        score += 4 * min(features["sensitive_skill_verbs"], 2)
        reasons.append(f"{features['sensitive_skill_verbs']} sensitive verb(s) in skills")

    # Ranking-manipulation detection (two-tier)
    # Tier 1: explicit orchestrator-directed selection imperatives → direct MALICIOUS
    if features["selection_directives"] > 0:
        score += 8
        reasons.append(
            f"{features['selection_directives']} explicit agent-selection directive(s) "
            "(orchestrator routing instruction)"
        )
    else:
        # Tier 2: composite promotional pressure; positioning_signals added to mix
        promotional_pressure = (
            features["superlatives"] +
            features["unverified_precision_claims"] +
            features["positioning_signals"] +
            (1 if features["sentiment_compound"] > sentiment_threshold else 0)
        )
        if promotional_pressure >= 3:
            score += 4
            reasons.append("high self-promotional / ranking-manipulation pressure")
        elif promotional_pressure >= 2:
            score += 2
            reasons.append("moderate self-promotional pressure")

    # Out-of-context skills: very low coherence
    if features["coherence"] is not None:
        if features["coherence"] < 0.05:
            score += 5
            reasons.append("description↔skills coherence near zero")
        elif features["coherence"] < 0.10:
            score += 3
            reasons.append("very low description↔skills coherence")

    # Context pollution
    if features["description_length"] > 4000:
        score += 3
        reasons.append("description > 4 KB (context pollution risk)")
    elif features["description_length"] > 2000:
        score += 1
        reasons.append("description > 2 KB")

    # Map to verdict
    if score >= 8:
        verdict = "MALICIOUS"
    elif score >= 3:
        verdict = "SUSPICIOUS"
    else:
        verdict = "BENIGN"
        if not reasons:
            reasons.append("no deterministic threat features detected")

    return score, verdict, reasons[:6]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NLPFeatureExtractor:
    """Extract deterministic NLP features and an aggregated severity verdict.

    Parameters
    ----------
    backend : {"vader", "siebert"}
        Sentiment backend.  "vader" (default) uses the rule-based NLTK
        SentimentIntensityAnalyzer.  "siebert" uses the transformer model
        siebert/sentiment-roberta-large-english (local CPU inference; ~1.4 GB).
    sentiment_threshold : float
        The promotional-pressure gate: sentiment_compound > threshold counts as
        one promotional signal in the composite score.  Default 0.85.
    """

    def __init__(self, backend: str = "vader", sentiment_threshold: float = 0.85):
        self._sentiment_threshold = sentiment_threshold
        if backend == "siebert":
            _analyzer = SiEBERTSentimentAnalyzer()
            self._get_compound = _analyzer.compound_score
        else:
            _analyzer = SentimentIntensityAnalyzer()
            self._get_compound = lambda text: _analyzer.polarity_scores(text)["compound"]

    def extract(self, card: dict) -> dict:
        text = collect_text_fields(card)
        # Defense 1b: apply confusable folding before all regex pattern checks.
        # Cyrillic/Greek homoglyphs are mapped to their Latin equivalents so that
        # patterns like \bignore\b match even when 'i' was replaced with 'і' (U+0456).
        # Sentiment and TF-IDF coherence use the unfolded text to preserve semantics.
        text_folded = text.translate(HOMOGLYPH_FOLD_TABLE)
        urls = collect_urls(card)
        zw, rtl = count_invisible_controls(text)

        sentiment = self._get_compound(text or "")
        features = {
            "structural_url_loopback": any(url_is_loopback_or_private(u) for u in urls),
            "structural_url_homograph": any(url_has_mixed_script(u) for u in urls),
            "unicode_zero_width": zw,
            "unicode_rtl_override": rtl,
            # Pattern-match features use homoglyph-folded text (Defense 1b)
            "imperative_phrases": count_imperative_phrases(text_folded),
            "system_framings": count_system_framings(text_folded),
            "classifier_addressing": count_classifier_addressing(text_folded),
            # Natural-language features use original text (preserve semantics)
            "superlatives": count_superlatives(text),
            "selection_directives": count_selection_directives(text),
            "positioning_signals": count_positioning_signals(text),
            "sensitive_skill_verbs": count_sensitive_skill_verbs(text),
            "unverified_precision_claims": count_precision_claims(text),
            "sentiment_compound": round(sentiment, 4),
            "description_length": len(card.get("description") or ""),
            "skill_count": len(card.get("skills") or []),
            "coherence": description_skill_coherence(card),
            "url_count": len(urls),
        }
        score, verdict, reasons = compute_severity(features, self._sentiment_threshold)
        features["severity_score"] = score
        features["verdict"] = verdict
        features["reasons"] = reasons
        return features


# ---------------------------------------------------------------------------
# Loaders + CLI
# ---------------------------------------------------------------------------

def load_cards(input_path: Path, orig_path: Path | None) -> list[dict]:
    out = []
    seen = set()
    data = json.loads(input_path.read_text())
    for v in data.get("valid_results", []):
        d = v["domain"]
        if d not in seen:
            out.append({"domain": d, "card": v["card"]})
            seen.add(d)
    if orig_path and orig_path.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from validate_agent_cards import validate_card
        odata = json.loads(orig_path.read_text())
        for r in odata.get("results", []):
            card = r.get("agent_card")
            if not isinstance(card, dict):
                continue
            v = validate_card(card)
            if v["score"] < 3:
                continue
            d = r["domain"]
            if d in seen:
                continue
            out.append({"domain": d, "card": card})
            seen.add(d)
    return out


def main():
    parser = argparse.ArgumentParser(description="Layer 2A NLP feature extractor")
    parser.add_argument("--input", default="cards.json",
                        help="Validated agent-card JSON to score (default: cards.json)")
    parser.add_argument("--orig", default="agent_cards.json",
                        help="Original crawler output for provenance (optional)")
    parser.add_argument("--output", default="nlp_features.json",
                        help="Output JSON path (default: nlp_features.json)")
    parser.add_argument("--csv", default="nlp_features.csv",
                        help="Output CSV path (default: nlp_features.csv)")
    parser.add_argument("--sentiment-backend", choices=["vader", "siebert"], default="vader",
                        help="Sentiment analysis backend (default: vader)")
    parser.add_argument("--sentiment-threshold", type=float, default=0.85,
                        help="Promotional-pressure gate threshold (default: 0.85)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cards = load_cards(Path(args.input), Path(args.orig))
    logger.info("Loaded %d cards. Computing NLP features...", len(cards))

    extractor = NLPFeatureExtractor(
        backend=args.sentiment_backend,
        sentiment_threshold=args.sentiment_threshold,
    )
    results = []
    for c in cards:
        feats = extractor.extract(c["card"])
        feats["domain"] = c["domain"]
        results.append(feats)

    # Save JSON
    Path(args.output).write_text(json.dumps(
        {"total": len(results), "results": sorted(results, key=lambda r: r["domain"])},
        indent=2, ensure_ascii=False,
    ))
    logger.info("Wrote JSON results to %s", args.output)

    # Save CSV
    if args.csv:
        keys = [
            "domain", "verdict", "severity_score",
            "structural_url_loopback", "structural_url_homograph",
            "unicode_zero_width", "unicode_rtl_override",
            "imperative_phrases", "system_framings", "classifier_addressing",
            "superlatives", "sensitive_skill_verbs", "unverified_precision_claims",
            "sentiment_compound", "description_length", "skill_count", "coherence",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in sorted(results, key=lambda x: x["domain"]):
                w.writerow([r.get(k, "") for k in keys])
        logger.info("Wrote CSV to %s", args.csv)

    # Print summary
    verdicts = Counter(r["verdict"] for r in results)
    print()
    print("=" * 60)
    print(f"NLP-only (Layer 2A) verdict summary on {len(results)} cards:")
    for v in ("BENIGN", "SUSPICIOUS", "MALICIOUS"):
        c = verdicts.get(v, 0)
        bar = "#" * c
        print(f"  {v:11s}  {c:4d}  ({c/len(results)*100:5.1f}%)  {bar[:40]}")

    # Feature prevalence
    print()
    print("Feature prevalence (cards where feature fires):")
    for k in [
        "structural_url_loopback", "structural_url_homograph",
        "unicode_zero_width", "unicode_rtl_override",
        "imperative_phrases", "system_framings", "classifier_addressing",
        "superlatives", "sensitive_skill_verbs", "unverified_precision_claims",
    ]:
        n = sum(1 for r in results if r.get(k))
        print(f"  {k:35s}  {n:4d}  ({n/len(results)*100:5.1f}%)")

    # Severity-score distribution
    print()
    print("Severity score histogram (0 = no signals, 8+ = MALICIOUS):")
    score_counts = Counter(r["severity_score"] for r in results)
    for s in sorted(score_counts):
        bar = "#" * score_counts[s]
        print(f"  {s:3d}: {score_counts[s]:4d}  {bar[:50]}")


if __name__ == "__main__":
    main()
