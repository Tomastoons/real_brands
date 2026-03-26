from collections import Counter
import re
from typing import NamedTuple
from urllib.parse import urlparse

from app.extract_shared import (
    BRACKET_ATTACHMENT_BONUS,
    BRACKET_DOMAIN_PATTERN,
    DEFAULT_SCHEME,
    DISTANCE_SCORES,
    DISTANCE_THRESHOLDS,
    DOMAIN_TOKEN_SPLIT_PATTERN,
    EXPLICIT_URL_PATTERN,
    GENERIC_DOMAIN_TOKENS,
    OVERLAP_MULTIPLIER,
    OVERLAP_PENALTY,
    PARENTHESIS_ATTACHMENT_BONUS,
    SAME_SENTENCE_BONUS,
    SENTENCE_SPLIT_PATTERN,
    TLD_PREFIXES,
    WWW_PREFIX,
    WWW_PREFIX_LEN,
)


class DomainMention(NamedTuple):
    domain: str
    base_url: str
    start_offset: int
    end_offset: int
    source_kind: str


GENERIC_BRAND_TOKENS = {
    "app",
    "apps",
    "cloud",
    "device",
    "family",
    "find",
    "live",
    "location",
    "map",
    "maps",
    "music",
    "my",
    "share",
}


def _strip_www_prefix(domain: str) -> str:
    return domain[WWW_PREFIX_LEN:] if domain.lower().startswith(WWW_PREFIX) else domain


def _normalized_domain_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = _strip_www_prefix(parsed.netloc.lower())
    return host or None


def _normalized_base_url_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = _strip_www_prefix(parsed.netloc.lower())
    if not host:
        return None
    scheme = parsed.scheme.lower() or DEFAULT_SCHEME
    return f"{scheme}://{host}"


def _split_sentences(text: str) -> list[str]:
    chunks = SENTENCE_SPLIT_PATTERN.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def contexts_for_brand(brand: str, text: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(brand)}\b", flags=re.IGNORECASE)
    return [sentence for sentence in _split_sentences(text) if pattern.search(sentence)]


def _extract_domains_in_contexts(contexts: list[str]) -> list[str]:
    seen: set[str] = set()
    domains: list[str] = []

    for context in contexts:
        for url in EXPLICIT_URL_PATTERN.findall(context):
            if domain := _normalized_domain_from_url(url):
                if domain not in seen:
                    seen.add(domain)
                    domains.append(domain)

        for domain in BRACKET_DOMAIN_PATTERN.findall(context):
            normalized = _strip_www_prefix(domain.lower())
            if normalized not in seen:
                seen.add(normalized)
                domains.append(normalized)

    return domains


def _extract_domain_mentions_with_offsets(text: str) -> list[DomainMention]:
    mentions: list[DomainMention] = []

    for match in EXPLICIT_URL_PATTERN.finditer(text):
        raw_url = match.group(0).rstrip(".,;:!?)\"]")
        if (domain := _normalized_domain_from_url(raw_url)) and (base_url := _normalized_base_url_from_url(raw_url)):
            mentions.append(DomainMention(domain, base_url, match.start(), match.end(), "explicit"))

    for match in BRACKET_DOMAIN_PATTERN.finditer(text):
        domain = _strip_www_prefix(match.group(1).lower())
        mentions.append(DomainMention(domain, f"{DEFAULT_SCHEME}://{domain}", match.start(1), match.end(1), "bracket"))

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
    return any(start <= pos_a < end and start <= pos_b < end for start, end in sentence_spans)


def _brand_tokens(brand: str) -> set[str]:
    return {
        token.lower()
        for token in DOMAIN_TOKEN_SPLIT_PATTERN.findall(brand)
        if len(token) > 1 and token.lower() not in GENERIC_BRAND_TOKENS
    }


def _domain_tokens(domain: str) -> set[str]:
    parts = domain.lower().split(".")
    parts = parts[:-1] if len(parts) > 1 else parts
    parts = parts[:-1] if len(parts) > 1 and parts[-1] in TLD_PREFIXES else parts

    return {
        token
        for part in parts
        for token in DOMAIN_TOKEN_SPLIT_PATTERN.findall(part)
        if len(token) > 1 and token not in GENERIC_DOMAIN_TOKENS
    }


def get_domain_for_brand(brand: str, text: str) -> str | None:
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

    for mention in domain_mentions:
        overlap = len(brand_token_set & _domain_tokens(mention.domain))
        # Only keep domains that lexically match the brand (e.g., openai.com for OpenAI).
        # This prevents assigning citation/article hosts as brand domains.
        if overlap == 0:
            continue
        score = 0

        for brand_start, brand_end in brand_mentions:
            brand_mid = (brand_start + brand_end) // 2
            domain_mid = (mention.start_offset + mention.end_offset) // 2
            distance = abs(brand_mid - domain_mid)

            for threshold, dist_score in zip(DISTANCE_THRESHOLDS, DISTANCE_SCORES):
                if distance <= threshold:
                    score += dist_score
                    break

            if _is_same_sentence(brand_mid, domain_mid, sentence_spans):
                score += SAME_SENTENCE_BONUS

            between = text[brand_end:mention.start_offset].strip()
            if mention.start_offset >= brand_end and len(between) <= 2 and between in {"[", "("}:
                score += BRACKET_ATTACHMENT_BONUS if mention.source_kind == "bracket" else PARENTHESIS_ATTACHMENT_BONUS

        score += overlap * OVERLAP_MULTIPLIER

        if score > 0:
            domain_scores[mention.domain] += score
            domain_hits[mention.domain] += 1
            if score > domain_best_mention_score.get(mention.domain, -10**9):
                domain_best_mention_score[mention.domain] = score
                domain_best_url[mention.domain] = mention.base_url

    if not domain_scores:
        return None

    ranked = sorted(domain_scores.items(), key=lambda item: (item[1], domain_hits[item[0]]), reverse=True)
    top_domain, top_score = ranked[0]

    if len(ranked) == 1:
        return domain_best_url.get(top_domain, f"{DEFAULT_SCHEME}://{top_domain}")

    second_domain, second_score = ranked[1]
    if top_score == second_score and domain_hits[top_domain] == domain_hits[second_domain]:
        return None
    if top_score - second_score < 2 and top_score < int(second_score * 1.2) + 1:
        return None
    return domain_best_url.get(top_domain, f"{DEFAULT_SCHEME}://{top_domain}")
