from functools import lru_cache
import os
from pathlib import Path
import re

from pydantic.dataclasses import dataclass
import spacy
from spacy.matcher import Matcher

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
DOMAIN_TOKEN_SPLIT_PATTERN = re.compile(r"[A-Za-z0-9]+")
GENERIC_DOMAIN_TOKENS = {"www", "m", "app", "api", "docs", "blog", "support", "help"}

# Distance thresholds and scoring weights for domain proximity to brand mentions
DISTANCE_THRESHOLDS = [80, 160, 300, 600]
DISTANCE_SCORES = [6, 4, 2, 1]  # Corresponding scores for each threshold
SAME_SENTENCE_BONUS = 2
BRACKET_ATTACHMENT_BONUS = 6  # For bracket-style mentions like [domain]
PARENTHESIS_ATTACHMENT_BONUS = 3  # For parenthesis-style mentions like (domain)
OVERLAP_MULTIPLIER = 8  # Score per overlapping token
OVERLAP_PENALTY = -2  # Penalty if no overlapping tokens
WWW_PREFIX = "www."
WWW_PREFIX_LEN = len(WWW_PREFIX)
TLD_PREFIXES = {"co", "com", "org", "net", "gov", "edu", "ac"}
DEFAULT_SCHEME = "https"
MIN_TOKEN_LEN = 1  # Minimum character length for tokens
PLAN_QUALIFIER_PATTERN = re.compile(r"\b(premium|student|plan|hifi|pro|plus|free)\b", flags=re.IGNORECASE)


@dataclass
class BrandResult:
    name: str
    mentions_count: int
    scopes: list[str]
    domain: str | None


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
