from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from spacy.tokens import Doc

ALL_CAPS_HEADING_PATTERN = re.compile(r"^[A-Z]{2,}(?:\s+[A-Z]{2,})*$")
DOMAIN_LIKE_TOKEN_PATTERN = re.compile(r"^(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s]*)?$")
WIKIDATA_ENTITY_SNAPSHOT_PATH = Path(__file__).with_name("wikidata_entity_snapshot.json")

EXCLUDED_NON_BRAND_PHRASES_LOWER = {
    "united states answer",
    "cloud gaming",
    "student plan",
    "discover weekly",
    "songcatcher",
    "airplay",
    "homepod",
    "wi-fi",
    "wifi",
    "vpns",
}

COMMON_NON_BRAND_TERMS = {
    "I", "In", "A", "An", "The", "These", "That", "Those", "This", "And", "But", "For", "On", "At", "To", "From", "With",
    "Question", "Answer", "Markdown", "Source", "Sources",
    "Who", "Why", "When", "Where", "Which", "How", "What", "Notes", "Most", "Would",
    "Short", "If", "It",
    "Other", "Unlimited", "Prime", "Out",
    "Overview", "Details", "Summary", "Guide", "Guides", "Notes", "Note", "Key", "Comparison", "Comparisons",
    "Market", "Leaders", "Leader", "Contender", "Service", "Services", "App", "Apps", "Device", "Devices",
    "Library", "Libraries", "Users", "History", "Table", "Tips", "Feature", "Features", "Pricing", "Price", "Prices",
    "Performance", "Safety", "Reliability", "Adoption", "Availability", "Access", "Support", "Quality", "Value",
    "Parent", "Company", "Strengths",
    "Budget", "Top", "Best", "Good", "Better", "Strong", "Focus", "Focused", "Focuses", "Start", "End",
    "Review", "Reviews", "Consider", "Considerations", "Compare", "Check", "Verify", "Confirm", "Use", "Uses",
    "Usage", "Test", "Try", "Need", "Needs", "Choose", "Choice", "Choices", "Option", "Options", "Popularity",
    "Audience", "Primary", "Additional", "Example", "Examples", "General", "Core", "Pros", "Cons", "Real",
    "Direct", "Here", "There", "Below", "Above", "Important", "Recommended", "Practical", "Typical", "Typically",
    "Common", "Global", "Local", "International", "Regional",
    "Music", "Large", "Deep", "Catalog", "Evaluate", "Among", "Seamless", "md",
    "iPhone", "iPad", "Android", "Alexa", "macOS",
    "iOS", "TV", "TVs", "PC", "PCs", "U.S.", "Dolby", "Atmos", "Spatial", "Audio",
    "Some", "Free", "Look", "Ensure", "Data", "Citations", "Platform", "Cross", "Often", "Many",
    "Premium", "Cheap", "Useful", "Offline", "Always", "Hi", "Res", "Generally", "Your", "You",
    "Quick", "They", "High", "Built", "Individual", "Broad", "Plus", "Higher", "Widely", "States",
    "Answer", "Expect", "Massive", "Included", "Program", "People", "Live", "Casual", "Passive",
    "Discovering", "Ties", "Wide", "Visual", "Very", "Huge",
}

COMMON_NON_BRAND_COUNTRIES = {
    "US", "United States", "USA", "China", "Germany", "Japan", "France", "Italy", "Spain", "Canada", "Australia", "India", "Brazil",
    "Mexico", "Russia", "South Korea", "UK", "GB", "United Kingdom", "Argentina", "Netherlands", "Switzerland", "Sweden",
    "Norway", "Denmark", "Finland", "Poland", "Belgium", "Austria", "Portugal", "Greece", "Ireland", "New Zealand",
    "Singapore", "Hong Kong", "Thailand", "Vietnam", "Indonesia", "Philippines", "Malaysia", "Pakistan", "Bangladesh",
    "Turkey", "Israel", "Saudi Arabia", "UAE", "Egypt", "South Africa", "Nigeria", "Kenya", "Chile", "Colombia",
    "Peru", "Venezuela", "Taiwan", "Ukraine", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Croatia",
    "DE", "JP", "FR", "IT", "ES", "CA", "AU", "IN", "BR", "MX", "RU", "KR", "SK", "NL", "CH", "SE",
    "NO", "DK", "FI", "PL", "BE", "AT", "PT", "GR", "IE", "NZ", "SG", "HK", "TH", "VN", "ID", "PH",
    "MY", "PK", "BD", "TR", "IL", "SA", "AE", "EG", "ZA", "NG", "KE", "CL", "CO", "PE", "VE", "TW",
    "UA", "CZ", "HU", "RO", "BG", "HR", "GB",
}

COMMON_NON_BRAND_DEMONYMS = {
    "American", "British", "Chinese", "French", "German", "Indian", "Italian", "Japanese", "Korean",
    "Spanish", "Dutch", "Swiss", "Swedish", "Norwegian", "Danish", "Finnish", "Polish", "Belgian",
    "Austrian", "Portuguese", "Greek", "Irish", "Australian", "Canadian", "Brazilian", "Mexican",
    "Russian", "Ukrainian", "Argentinian", "Argentine", "Colombian", "Peruvian", "Chilean", "Turkish",
    "Israeli", "Saudi", "Emirati", "Egyptian", "South African", "Nigerian", "Kenyan", "Taiwanese",
    "Singaporean", "Malaysian", "Indonesian", "Filipino", "Thai", "Vietnamese", "Pakistani", "Bangladeshi",
    "Romanian", "Bulgarian", "Croatian", "Czech", "Hungarian",
}

COMMON_NON_BRAND_TERMS_LOWER = {value.lower() for value in COMMON_NON_BRAND_TERMS}
COMMON_NON_BRAND_COUNTRIES_LOWER = {value.lower() for value in COMMON_NON_BRAND_COUNTRIES}
COMMON_NON_BRAND_DEMONYMS_LOWER = {value.lower() for value in COMMON_NON_BRAND_DEMONYMS}
MULTIWORD_COUNTRY_NAMES_LOWER = {
    value.lower() for value in COMMON_NON_BRAND_COUNTRIES if " " in value
}
COMMON_NON_BRANDS_LOWER = (
    COMMON_NON_BRAND_TERMS_LOWER | COMMON_NON_BRAND_COUNTRIES_LOWER | COMMON_NON_BRAND_DEMONYMS_LOWER
)
BRAND_CONTEXT_WORDS = {
    "bought", "buy", "buying", "wear", "wearing", "wore", "use", "uses", "used", "using",
    "prefer", "preferred", "recommend", "recommended", "recommends", "choose", "chose", "picked",
    "install", "installed", "subscribe", "subscribed", "by",
}
BRAND_ENTITY_LABELS = {"BRAND", "ORG", "PRODUCT", "WORK_OF_ART"}
EXCLUDED_ENTITY_LABELS = {"PERSON", "GPE", "LOC", "NORP", "FAC", "EVENT", "LAW"}
EXCLUDED_NAME_TERMS = {
    "police", "ministry", "government", "department", "agency", "council", "authority",
    "embassy", "commission", "office", "municipality",
}


@dataclass(frozen=True)
class CandidateHeuristics:
    name: str
    mention_count: int
    country_name: bool
    named_entity_brand_label: bool
    excluded_named_entity: bool
    excluded_name_term: bool
    mid_sentence_capitalized: bool
    sentence_start_subject: bool
    has_brand_context: bool
    mixed_case_branding: bool
    wikidata_brand_or_product: bool | None
    wikidata_excluded: bool | None

    def should_keep(self) -> bool:
        if self.country_name:
            return False
        if self.excluded_named_entity:
            return False
        if self.excluded_name_term or self.wikidata_excluded is True:
            return False
        if self.wikidata_brand_or_product is True:
            return True
        # Reject plain one-word candidates before trusting model labels.
        if " " not in self.name and not self.mixed_case_branding:
            return False
        if self.named_entity_brand_label:
            return True
        if self.has_brand_context:
            return True
        if self.mixed_case_branding:
            return True
        if self.mid_sentence_capitalized:
            return True
        return False


def is_valid_brand_candidate(token: str, scope_keywords_lower: set[str]) -> bool:
    candidate = token.strip()
    if not candidate or len(candidate) <= 1:
        return False

    lower_candidate = candidate.lower()
    if lower_candidate in EXCLUDED_NON_BRAND_PHRASES_LOWER:
        return False
    if "http://" in lower_candidate or "https://" in lower_candidate or "www." in lower_candidate:
        return False
    if any(character in candidate for character in "[]"):
        return False
    if any(character in candidate for character in "@#$%^&*_=+\\|/<>~`"):
        return False
    if ".{" in candidate or "}." in candidate or ".[" in candidate or "]." in candidate:
        return False
    if " " not in candidate and DOMAIN_LIKE_TOKEN_PATTERN.fullmatch(candidate):
        return False
    if " " not in candidate and candidate.islower():
        return False
    if " " in candidate:
        letters = [character for character in candidate if character.isalpha()]
        if letters and all(character.islower() for character in letters):
            return False
    if lower_candidate in COMMON_NON_BRANDS_LOWER:
        return False
    if " " not in candidate and lower_candidate in scope_keywords_lower:
        return False
    words = [word.lower() for word in candidate.split()]
    if words and all(word in COMMON_NON_BRANDS_LOWER for word in words):
        return False
    if ALL_CAPS_HEADING_PATTERN.fullmatch(candidate):
        return False
    return True


def filter_brand_candidates(
    candidates: list[str],
    *,
    source_text: str,
    mention_counts: dict[str, int],
    doc: Doc,
    brand_entity_labels: set[str] | None = None,
) -> list[str]:
    labels = brand_entity_labels or BRAND_ENTITY_LABELS
    filtered: list[str] = []

    for candidate in candidates:
        heuristics = evaluate_candidate_heuristics(
            candidate,
            source_text=source_text,
            mention_count=mention_counts.get(candidate, 0),
            doc=doc,
            brand_entity_labels=labels,
        )
        if heuristics.should_keep():
            filtered.append(candidate)

    return filtered


def evaluate_candidate_heuristics(
    candidate: str,
    *,
    source_text: str,
    mention_count: int,
    doc: Doc,
    brand_entity_labels: set[str] | None = None,
) -> CandidateHeuristics:
    labels = brand_entity_labels or BRAND_ENTITY_LABELS
    occurrences = _find_occurrences(source_text, candidate)
    entity_labels = _entity_labels_for_candidate(doc, occurrences)
    wikidata_match = lookup_wikidata_entity_types(candidate)

    return CandidateHeuristics(
        name=candidate,
        mention_count=mention_count,
        country_name=_is_country_name(candidate)
        or _is_country_component_in_context(candidate, source_text, occurrences),
        named_entity_brand_label=any(label in labels for label in entity_labels),
        excluded_named_entity=any(label in EXCLUDED_ENTITY_LABELS for label in entity_labels),
        excluded_name_term=_has_excluded_name_term(candidate),
        mid_sentence_capitalized=any(_is_mid_sentence_capitalized(source_text, start, candidate) for start, _ in occurrences),
        sentence_start_subject=any(_is_sentence_start_subject(source_text, start, end) for start, end in occurrences),
        has_brand_context=any(_has_brand_context(source_text, start) for start, _ in occurrences),
        mixed_case_branding=_has_mixed_case_branding(candidate),
        wikidata_brand_or_product=wikidata_match[0],
        wikidata_excluded=wikidata_match[1],
    )


def _find_occurrences(text: str, candidate: str) -> list[tuple[int, int]]:
    pattern = re.compile(rf"\b{re.escape(candidate)}\b", flags=re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _entity_labels_for_candidate(doc: Doc, occurrences: list[tuple[int, int]]) -> set[str]:
    labels: set[str] = set()
    for start, end in occurrences:
        for entity in doc.ents:
            if entity.start_char < end and start < entity.end_char:
                labels.add(entity.label_)
    return labels


def _is_mid_sentence_capitalized(text: str, start: int, candidate: str) -> bool:
    if not any(character.isupper() for character in candidate):
        return False

    probe = start - 1
    while probe >= 0 and text[probe] in " \t\r\n\"'([{":
        probe -= 1
    if probe < 0:
        return False
    return text[probe] not in ".!?;:"


def _has_brand_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40):start].lower()
    words = re.findall(r"[a-z]+", prefix)
    if not words:
        return False
    return any(word in BRAND_CONTEXT_WORDS for word in words[-3:])


def _is_sentence_start_subject(text: str, start: int, end: int) -> bool:
    probe = start - 1
    while probe >= 0 and text[probe] in " \t\r\n\"'([{":
        probe -= 1
    if probe >= 0 and text[probe] not in ".!?;:":
        return False

    suffix = text[end:]
    match = re.match(r"[ \t\r\n,;:-]*([A-Za-z][A-Za-z-]*)", suffix)
    if not match:
        return False
    return match.group(1)[0].islower()


def _has_mixed_case_branding(candidate: str) -> bool:
    token = candidate.replace(" ", "")
    return any(character.isupper() for character in token[1:])


def _has_excluded_name_term(candidate: str) -> bool:
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", candidate)}
    return any(word in EXCLUDED_NAME_TERMS for word in words)


def _is_country_name(candidate: str) -> bool:
    normalized = candidate.strip().lower()
    return normalized in COMMON_NON_BRAND_COUNTRIES_LOWER or normalized in COMMON_NON_BRAND_DEMONYMS_LOWER


def _is_country_component_in_context(
    candidate: str,
    source_text: str,
    occurrences: list[tuple[int, int]],
) -> bool:
    token = candidate.strip().lower()
    if not token or " " in token:
        return False

    for start, end in occurrences:
        previous_word = _previous_word(source_text, start)
        if previous_word and f"{previous_word} {token}" in MULTIWORD_COUNTRY_NAMES_LOWER:
            return True

        next_word = _next_word(source_text, end)
        if next_word and f"{token} {next_word}" in MULTIWORD_COUNTRY_NAMES_LOWER:
            return True

    return False


def _previous_word(text: str, index: int) -> str | None:
    prefix = text[:index]
    matches = list(re.finditer(r"[A-Za-z][A-Za-z-]*", prefix))
    if not matches:
        return None
    return matches[-1].group(0).lower()


def _next_word(text: str, index: int) -> str | None:
    suffix = text[index:]
    match = re.search(r"[A-Za-z][A-Za-z-]*", suffix)
    if not match:
        return None
    return match.group(0).lower()


@lru_cache(maxsize=512)
def lookup_wikidata_entity_types(candidate: str) -> tuple[bool | None, bool | None]:
    snapshot = _load_wikidata_entity_snapshot()
    if not snapshot:
        return None, None

    entry = snapshot.get(candidate.strip().lower())
    if not entry:
        return None, None
    return entry.get("brand_or_product"), entry.get("excluded")


@lru_cache(maxsize=1)
def _load_wikidata_entity_snapshot() -> dict[str, dict[str, bool | None]]:
    try:
        with WIKIDATA_ENTITY_SNAPSHOT_PATH.open("r", encoding="utf-8") as snapshot_file:
            payload = json.load(snapshot_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    entries = payload.get("entities", {})
    return {
        key.lower(): {
            "brand_or_product": value.get("brand_or_product"),
            "excluded": value.get("excluded"),
        }
        for key, value in entries.items()
        if isinstance(value, dict)
    }