from app.heuristics import BRAND_ENTITY_LABELS
from app.taxonomy import get_scopes_from_contexts
from app.extract_candidates import (
    collapse_component_brands,
    count_exact_mentions,
    extract_candidates_in_order,
    filter_brands,
    merge_canonical_brand_results,
)
from app.extract_domains import contexts_for_brand, get_domain_for_brand
from app.extract_shared import BrandResult, _get_nlp


def _get_scopes_for_brand(brand: str, text: str) -> list[str]:
    contexts = contexts_for_brand(brand, text)
    return get_scopes_from_contexts(contexts)


def extract_brand_analysis(answer_text: str, raw_text: str | None = None) -> list[BrandResult]:
    """Extract brand information including mentions, scopes, and domains from text."""
    source_text = raw_text or answer_text
    doc = _get_nlp()(source_text)
    ordered_brands = extract_candidates_in_order(answer_text, raw_text=raw_text, doc=doc)
    mention_counts = count_exact_mentions(answer_text, ordered_brands)

    ordered_brands = filter_brands(
        ordered_brands,
        source_text=source_text,
        mention_counts=mention_counts,
        doc=doc,
        brand_entity_labels=BRAND_ENTITY_LABELS,
    )
    ordered_brands = collapse_component_brands(ordered_brands)
    mention_counts = count_exact_mentions(answer_text, ordered_brands)

    raw_results = [
        BrandResult(
            name=brand,
            mentions_count=mention_counts[brand],
            scopes=_get_scopes_for_brand(brand, answer_text),
            domain=get_domain_for_brand(brand, answer_text),
        )
        for brand in ordered_brands
    ]
    return merge_canonical_brand_results(raw_results)
