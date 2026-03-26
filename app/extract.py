from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from pydantic.dataclasses import dataclass
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.tokens import Span

from app.heuristics import BRAND_ENTITY_LABELS, COMMON_NON_BRANDS_LOWER, filter_brand_candidates, is_valid_brand_candidate
from app.taxonomy import SCOPE_TAXONOMY, get_scopes_from_contexts

SPACY_BRAND_ENTITY_LABELS = {"BRAND", "ORG", "PRODUCT", "WORK_OF_ART"}
DEFAULT_TRAINED_MODEL_PATH = Path("models") / "brand_ner"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
EXPLICIT_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")
BRACKET_DOMAIN_PATTERN = re.compile(r"\[\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[^\]\s]*)?\s*\]")
# Captures the first word after structural markdown prefixes used for section/list/table labels.
STRUCTURAL_LEAD_WORD_PATTERN = re.compile(
    r"(?:\n{2,}[ \t]*|\n[ \t]*[-*][ \t]+|\|[ \t]+|\[artifact\][ \t]+)([A-Za-z][A-Za-z0-9]*)"
)

CANDIDATE_WORD_PATTERN = re.compile(r"^[A-Za-z0-9&/\-]+$")
SCOPE_KEYWORDS_LOWER = {keyword.lower() for keywords in SCOPE_TAXONOMY.values() for keyword in keywords}
DOMAIN_TOKEN_SPLIT_PATTERN = re.compile(r"[A-Za-z0-9]+")
GENERIC_DOMAIN_TOKENS = {"www", "m", "app", "api", "docs", "blog", "support", "help"}


@lru_cache(maxsize=1)
def _get_nlp() -> spacy.language.Language:
    model_path = Path(os.getenv("BRAND_NER_MODEL_PATH", str(DEFAULT_TRAINED_MODEL_PATH)))
    if model_path.exists():
        return spacy.load(model_path)
    try:
        return spacy.load("en_core_web_lg")
    except OSError as exc:
        raise RuntimeError(
            "SpaCy model is required. Install with: python -m spacy download en_core_web_lg "
            "or train and save a model under models/brand_ner."
        ) from exc


@lru_cache(maxsize=1)
def _get_brand_matcher() -> Matcher:
    matcher = Matcher(_get_nlp().vocab)
    # Candidate brand phrases as 1-3 proper nouns (e.g., OpenAI, Apple Music, NetEase Cloud Music).
    matcher.add(
        "BRAND_PROPN_PHRASE",
        [
            [{"POS": "PROPN"}],
            [{"POS": "PROPN"}, {"POS": "PROPN"}],
            [{"POS": "PROPN"}, {"POS": "PROPN"}, {"POS": "PROPN"}],
        ],
    )
    # Candidate brand phrases as 1-3 capitalized tokens (e.g., OpenAI, Spotify, Apple Music).
    matcher.add(
        "BRAND_CAPITALIZED_PHRASE",
        [
            [{"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}}],
            [
                {"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}},
                {"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}},
            ],
            [
                {"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}},
                {"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}},
                {"TEXT": {"REGEX": r"^[A-Z][A-Za-z0-9&\-]*$"}},
            ],
        ],
    )
    return matcher


@dataclass
class BrandResult:
    name: str
    mentions_count: int
    scopes: list[str]
    domain: str | None


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


def _extract_candidates_in_order(text: str, raw_text: str | None = None, doc: Doc | None = None) -> list[str]:
    source_text = raw_text or text
    doc = doc or _get_nlp()(source_text)
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



def _count_exact_mentions(text: str, brands: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    count_text = EXPLICIT_URL_PATTERN.sub(" ", text)
    count_text = BRACKET_DOMAIN_PATTERN.sub(" ", count_text)
    for brand in brands:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", flags=re.IGNORECASE)
        counts[brand] = len(pattern.findall(count_text))
    return counts


def _collapse_component_brands(text: str, brands: list[str]) -> list[str]:
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


def _split_sentences(text: str) -> list[str]:
    chunks = SENTENCE_SPLIT_PATTERN.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _contexts_for_brand(brand: str, text: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(brand)}\b", flags=re.IGNORECASE)
    return [sentence for sentence in _split_sentences(text) if pattern.search(sentence)]


def _get_scopes_for_brand(brand: str, text: str) -> list[str]:
    contexts = _contexts_for_brand(brand, text)
    return get_scopes_from_contexts(contexts)


def _normalized_domain_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _normalized_base_url_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    return f"{scheme}://{host}"


def _extract_domains_in_contexts(contexts: list[str]) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()

    for context in contexts:
        explicit_urls = EXPLICIT_URL_PATTERN.findall(context)
        for url in explicit_urls:
            domain = _normalized_domain_from_url(url)
            if domain and domain not in seen:
                seen.add(domain)
                domains.append(domain)

        bracket_domains = BRACKET_DOMAIN_PATTERN.findall(context)
        for domain in bracket_domains:
            normalized = domain.lower()
            if normalized.startswith("www."):
                normalized = normalized[4:]
            if normalized not in seen:
                seen.add(normalized)
                domains.append(normalized)

    return domains


def _extract_domain_mentions_with_offsets(text: str) -> list[tuple[str, str, int, int, str]]:
    mentions: list[tuple[str, str, int, int, str]] = []

    for match in EXPLICIT_URL_PATTERN.finditer(text):
        raw_url = match.group(0).rstrip(".,;:!?)\"]")
        domain = _normalized_domain_from_url(raw_url)
        base_url = _normalized_base_url_from_url(raw_url)
        if domain and base_url:
            mentions.append((domain, base_url, match.start(), match.end(), "explicit"))

    for match in BRACKET_DOMAIN_PATTERN.finditer(text):
        domain = match.group(1).lower()
        if domain.startswith("www."):
            domain = domain[4:]
        mentions.append((domain, f"https://{domain}", match.start(1), match.end(1), "bracket"))

    return mentions


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_SPLIT_PATTERN.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _is_same_sentence(pos_a: int, pos_b: int, sentence_spans: list[tuple[int, int]]) -> bool:
    for start, end in sentence_spans:
        if start <= pos_a < end and start <= pos_b < end:
            return True
    return False


def _brand_tokens(brand: str) -> set[str]:
    return {
        token.lower()
        for token in DOMAIN_TOKEN_SPLIT_PATTERN.findall(brand)
        if len(token) > 1
    }


def _domain_tokens(domain: str) -> set[str]:
    parts = domain.lower().split(".")
    if len(parts) > 1:
        parts = parts[:-1]
    if len(parts) > 1 and parts[-1] in {"co", "com", "org", "net", "gov", "edu", "ac"}:
        parts = parts[:-1]

    tokens: set[str] = set()
    for part in parts:
        for token in DOMAIN_TOKEN_SPLIT_PATTERN.findall(part):
            if len(token) <= 1 or token in GENERIC_DOMAIN_TOKENS:
                continue
            tokens.add(token)
    return tokens


def _get_domain_for_brand(brand: str, text: str) -> str | None:
    brand_pattern = re.compile(rf"\b{re.escape(brand)}\b", flags=re.IGNORECASE)
    brand_mentions = [(match.start(), match.end()) for match in brand_pattern.finditer(text)]
    if not brand_mentions:
        return None

    domain_mentions = _extract_domain_mentions_with_offsets(text)
    if not domain_mentions:
        return None

    sentence_spans = _sentence_spans(text)
    brand_token_set = _brand_tokens(brand)
    domain_scores: Counter[str] = Counter()
    domain_hits: Counter[str] = Counter()
    domain_best_url: dict[str, str] = {}
    domain_best_mention_score: dict[str, int] = {}

    for domain, base_url, dom_start, dom_end, source_kind in domain_mentions:
        overlap = len(brand_token_set & _domain_tokens(domain))
        score = 0

        for brand_start, brand_end in brand_mentions:
            brand_mid = (brand_start + brand_end) // 2
            domain_mid = (dom_start + dom_end) // 2
            distance = abs(brand_mid - domain_mid)

            if distance <= 80:
                score += 6
            elif distance <= 160:
                score += 4
            elif distance <= 300:
                score += 2
            elif distance <= 600:
                score += 1

            if _is_same_sentence(brand_mid, domain_mid, sentence_spans):
                score += 2

            # Strong positive signal when the domain is attached directly after a brand mention.
            between = text[brand_end:dom_start].strip()
            if dom_start >= brand_end and len(between) <= 2 and (between == "[" or between == "("):
                score += 6 if source_kind == "bracket" else 3

        score += overlap * 8
        if overlap == 0:
            score -= 2

        if score > 0:
            domain_scores[domain] += score
            domain_hits[domain] += 1
            if score > domain_best_mention_score.get(domain, -10**9):
                domain_best_mention_score[domain] = score
                domain_best_url[domain] = base_url

    if not domain_scores:
        contexts = _contexts_for_brand(brand, text)
        domains = _extract_domains_in_contexts(contexts)
        if len(domains) == 1:
            return f"https://{domains[0]}"
        return None

    ranked = sorted(domain_scores.items(), key=lambda item: (item[1], domain_hits[item[0]]), reverse=True)
    top_domain, top_score = ranked[0]

    if len(ranked) == 1:
        return domain_best_url.get(top_domain, f"https://{top_domain}")

    second_domain, second_score = ranked[1]
    if top_score == second_score and domain_hits[top_domain] == domain_hits[second_domain]:
        return None
    if top_score - second_score < 2 and top_score < int(second_score * 1.2) + 1:
        return None
    return domain_best_url.get(top_domain, f"https://{top_domain}")


def extract_brand_analysis(answer_text: str, raw_text: str | None = None) -> list[BrandResult]:
    source_text = raw_text or answer_text
    doc = _get_nlp()(source_text)
    ordered_brands = _extract_candidates_in_order(answer_text, raw_text=raw_text, doc=doc)
    mention_counts = _count_exact_mentions(answer_text, ordered_brands)
    ordered_brands = filter_brand_candidates(
        ordered_brands,
        source_text=source_text,
        mention_counts=mention_counts,
        doc=doc,
        brand_entity_labels=BRAND_ENTITY_LABELS,
    )
    ordered_brands = _collapse_component_brands(answer_text, ordered_brands)
    mention_counts = _count_exact_mentions(answer_text, ordered_brands)
    results: list[BrandResult] = []
    for brand in ordered_brands:
        results.append(
            BrandResult(
                name=brand,
                mentions_count=mention_counts[brand],
                scopes=_get_scopes_for_brand(brand, answer_text),
                domain=_get_domain_for_brand(brand, answer_text),
            )
        )
    return results
