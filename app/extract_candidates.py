from collections import Counter
from collections.abc import Iterable
import re

from spacy.tokens import Doc, Span

from app.heuristics import COMMON_NON_BRANDS_LOWER, filter_brand_candidates, is_valid_brand_candidate
from app.taxonomy import SCOPE_TAXONOMY
from app.extract_shared import (
    BRACKET_DOMAIN_PATTERN,
    CANDIDATE_WORD_PATTERN,
    EXPLICIT_URL_PATTERN,
    PLAN_QUALIFIER_PATTERN,
    SPACY_BRAND_ENTITY_LABELS,
    STRUCTURAL_LEAD_WORD_PATTERN,
    BrandResult,
    _get_brand_matcher,
)

SCOPE_KEYWORDS_LOWER = {keyword.lower() for keywords in SCOPE_TAXONOMY.values() for keyword in keywords}


def _is_brandish_word(word: str) -> bool:
    if not word or not CANDIDATE_WORD_PATTERN.fullmatch(word):
        return False
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return False
    # Accept conventional title-case and camel-case tokens (e.g., Spotify, iPhone/iPad).
    return word[0].isupper() or any(ch.isupper() for ch in word[1:])


def _normalize_candidate_from_span(span: Span) -> str:
    cleaned_tokens = [tok.text.strip("`*_:-,.;()[]{}") for tok in span]
    start_index = 0
    while start_index < len(cleaned_tokens) and cleaned_tokens[start_index].lower() in COMMON_NON_BRANDS_LOWER:
        start_index += 1

    words: list[str] = []
    for cleaned in cleaned_tokens[start_index:]:
        if not cleaned:
            continue
        if not _is_brandish_word(cleaned):
            break
        words.append(cleaned)

    if start_index > 0:
        next_token_index = span.end
        while len(words) < 3 and next_token_index < len(span.doc):
            cleaned = span.doc[next_token_index].text.strip("`*_:-,.;()[]{}")
            if not cleaned or cleaned.lower() in COMMON_NON_BRANDS_LOWER:
                break
            if not _is_brandish_word(cleaned):
                break
            words.append(cleaned)
            next_token_index += 1

    if words:
        return " ".join(words)
    return span.text.strip().strip("`*_:-")


def _get_structural_lead_words(raw_text: str) -> frozenset[str]:
    """Return words that appear only at structural markup positions in raw text."""
    structural_words = STRUCTURAL_LEAD_WORD_PATTERN.findall(raw_text)
    if not structural_words:
        return frozenset()

    # Blank out structural-prefix words and look for capitalized occurrences elsewhere.
    body_text = STRUCTURAL_LEAD_WORD_PATTERN.sub(lambda m: " " * len(m.group(0)), raw_text)
    excluded = set()
    for word in structural_words:
        if not re.search(rf"\b{re.escape(word)}\b", body_text):
            excluded.add(word)
    return frozenset(excluded)


def extract_candidates_in_order(text: str, raw_text: str | None = None, doc: Doc | None = None) -> list[str]:
    source_text = raw_text or text
    if doc is None:
        raise ValueError("Doc is required for candidate extraction.")
    matcher = _get_brand_matcher()
    structural_words = _get_structural_lead_words(source_text) if source_text else frozenset()

    spans: list[Span] = [ent for ent in doc.ents if ent.label_ in SPACY_BRAND_ENTITY_LABELS]
    for _, start, end in matcher(doc):
        spans.append(doc[start:end])

    ordered_spans = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
    seen: set[str] = set()
    ordered: list[str] = []

    for span in ordered_spans:
        token = _normalize_candidate_from_span(span)
        if not is_valid_brand_candidate(token, SCOPE_KEYWORDS_LOWER):
            continue
        if token in structural_words:
            continue
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def count_exact_mentions(text: str, brands: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    count_text = EXPLICIT_URL_PATTERN.sub(" ", text)
    count_text = BRACKET_DOMAIN_PATTERN.sub(" ", count_text)
    for brand in brands:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", flags=re.IGNORECASE)
        counts[brand] = len(pattern.findall(count_text))
    return counts


def collapse_component_brands(brands: list[str]) -> list[str]:
    """Remove single-token brands that are components of accepted multi-word brands."""
    multiword_brands = [brand for brand in brands if " " in brand]
    if not multiword_brands:
        return brands

    drop_candidates: set[str] = set()
    for multiword in multiword_brands:
        multiword_tokens = multiword.split()
        for token in multiword.split():
            if token in brands and " " not in token:
                drop_candidates.add(token)

        # Drop shorter contiguous multiword fragments of longer accepted brands.
        for other in multiword_brands:
            if other == multiword:
                continue
            other_tokens = other.split()
            if len(other_tokens) >= len(multiword_tokens):
                continue
            for idx in range(0, len(multiword_tokens) - len(other_tokens) + 1):
                if multiword_tokens[idx: idx + len(other_tokens)] == other_tokens:
                    drop_candidates.add(other)
                    break

    return [brand for brand in brands if brand not in drop_candidates]


def _canonicalize_brand_name(name: str) -> str:
    """Normalize common plan/tier variants to the base service brand name."""
    trimmed = " ".join(name.split())
    lower = trimmed.lower()

    if lower.startswith("spotify ") and PLAN_QUALIFIER_PATTERN.search(trimmed):
        return "Spotify"
    if lower.startswith("apple music ") and PLAN_QUALIFIER_PATTERN.search(trimmed):
        return "Apple Music"
    if lower.startswith("youtube music ") and PLAN_QUALIFIER_PATTERN.search(trimmed):
        return "YouTube Music"
    if lower.startswith("amazon music ") and lower.endswith(" unlimited"):
        return "Amazon Music"
    if lower in {"tidal hifi", "tidal hifi plus", "tidal pro"}:
        return "Tidal"
    return trimmed


def merge_canonical_brand_results(results: list[BrandResult]) -> list[BrandResult]:
    """Merge variant brand entries into canonical brands while preserving appearance order."""
    merged: dict[str, BrandResult] = {}

    for result in results:
        canonical = _canonicalize_brand_name(result.name)
        existing = merged.get(canonical)
        if existing is None:
            merged[canonical] = BrandResult(
                name=canonical,
                mentions_count=result.mentions_count,
                scopes=list(result.scopes),
                domain=result.domain,
                price_tiers=list(result.price_tiers),
            )
            continue

        existing.mentions_count += result.mentions_count
        for scope in result.scopes:
            if scope not in existing.scopes:
                existing.scopes.append(scope)
        if existing.domain is None and result.domain is not None:
            existing.domain = result.domain
        for tier in result.price_tiers:
            if tier not in existing.price_tiers:
                existing.price_tiers.append(tier)

    return list(merged.values())


def filter_brands(
    ordered_brands: list[str],
    *,
    source_text: str,
    mention_counts: dict[str, int],
    doc: Doc,
    brand_entity_labels: set[str],
) -> list[str]:
    return filter_brand_candidates(
        ordered_brands,
        source_text=source_text,
        mention_counts=mention_counts,
        doc=doc,
        brand_entity_labels=brand_entity_labels,
    )
