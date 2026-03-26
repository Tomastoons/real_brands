import app.heuristics as heuristics
from app.extract import _get_nlp, extract_brand_analysis


def test_extract_brand_analysis_preserves_order_and_exact_mentions() -> None:
    answer = (
        "Spotify offers strong discovery. "
        "Apple Music is deeply integrated with iOS. "
        "Spotify pricing can be competitive."
    )

    result = extract_brand_analysis(answer)
    names = [item.name for item in result]
    assert names[:2] == ["Spotify", "Apple Music"]

    spotify = next(item for item in result if item.name == "Spotify")
    apple = next(item for item in result if item.name == "Apple Music")
    assert spotify.mentions_count == 2
    assert apple.mentions_count == 1


def test_extract_brand_analysis_infers_scopes() -> None:
    answer = (
        "OpenAI has strong API features and tooling. "
        "OpenAI pricing is flexible with multiple plans. "
        "Anthropic focuses on safety and reliable output."
    )

    result = extract_brand_analysis(answer)

    openai = next(item for item in result if item.name == "OpenAI")
    anthropic = next(item for item in result if item.name == "Anthropic")
    assert "features" in openai.scopes
    assert "pricing" in openai.scopes
    assert "safety" in anthropic.scopes
    assert "reliability" in anthropic.scopes


def test_extract_brand_analysis_domain_prefers_brand_matching_evidence() -> None:
    answer = (
        "Spotify [spotify.com] is popular globally. "
        "OpenAI appears on https://openai.com/docs for API details. "
        "OpenAI is also discussed at https://example.com/openai-overview."
    )

    result = extract_brand_analysis(answer)
    spotify = next(item for item in result if item.name == "Spotify")
    openai = next(item for item in result if item.name == "OpenAI")

    assert spotify.domain == "spotify.com"
    assert openai.domain == "openai.com"


def test_extract_brand_analysis_filters_question_words() -> None:
    answer = "What is best? Spotify and Deezer are common picks."
    names = [item.name for item in extract_brand_analysis(answer)]
    assert "What" not in names
    assert "Spotify" in names
    assert "Deezer" in names


def test_extract_brand_analysis_handles_missing_raw_text() -> None:
    answer = "OpenAI [openai.com] has strong API features."

    result = extract_brand_analysis(answer, raw_text=None)

    openai = next(item for item in result if item.name == "OpenAI")
    assert openai.mentions_count == 1
    assert openai.domain == "openai.com"


def test_extract_brand_analysis_excludes_markdown_heading_titles() -> None:
    answer = (
        "## Market Leaders\n"
        "Spotify and Apple Music are popular.\n"
        "## Key Comparison\n"
        "Spotify has strong discovery features."
    )
    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Market Leaders" not in names
    assert "Key Comparison" not in names
    assert "Spotify" in names
    assert "Apple Music" in names


def test_extract_brand_analysis_excludes_inline_markdown_heading_titles() -> None:
    answer = "## Market Leaders KuGou Music dominates. ## Strong Contender NetEase Cloud Music grows fast."
    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Market Leaders" not in names
    assert "Strong Contender" not in names
    assert "KuGou Music" in names
    assert "NetEase Cloud Music" in names


def test_extract_brand_analysis_excludes_markdown_table_header_cells() -> None:
    answer = (
        "| Service | Key Strengths | Parent Company |\n"
        "|---------|---------------|----------------|\n"
        "| KuGou Music | Social features, audio quality | Tencent |"
    )
    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Service" not in names
    assert "Key Strengths" not in names
    assert "Parent Company" not in names
    assert "Social" not in names
    assert "KuGou Music" in names


def test_extract_brand_analysis_excludes_scope_descriptor_singletons() -> None:
    answer = "Community support matters. Spotify has strong community and social features."
    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Community" not in names
    assert "Spotify" in names


def test_extract_brand_analysis_filters_government_and_location_entities() -> None:
    answer = (
        "India has many options. "
        "Spotify remains a popular service. "
        "Delhi Police operates a public safety hotline."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "India" not in names
    assert "Delhi Police" not in names
    assert "Spotify" in names


def test_extract_brand_analysis_filters_country_components() -> None:
    answer = (
        "United States has many options for users. "
        "Spotify remains popular globally."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "United" not in names
    assert "States" not in names
    assert "Spotify" in names


def test_extract_brand_analysis_filters_sentence_start_generic_terms() -> None:
    answer = "Overview: Spotify is widely used. Details: Apple Music integrates well with iOS."

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Overview" not in names
    assert "Details" not in names
    assert "Spotify" in names
    assert "Apple Music" in names


def test_extract_brand_analysis_can_keep_wikidata_brand_signal(monkeypatch) -> None:
    def fake_lookup(candidate: str) -> tuple[bool | None, bool | None]:
        if candidate == "FictionalBrand":
            return True, False
        return None, None

    monkeypatch.setattr(heuristics, "lookup_wikidata_entity_types", fake_lookup)

    answer = "FictionalBrand offers affordable plans for creators."
    names = [item.name for item in extract_brand_analysis(answer)]

    assert "FictionalBrand" in names


def test_country_name_is_rejected_even_with_positive_signals(monkeypatch) -> None:
    def fake_lookup(candidate: str) -> tuple[bool | None, bool | None]:
        if candidate == "Japan":
            return True, False
        return None, None

    monkeypatch.setattr(heuristics, "lookup_wikidata_entity_types", fake_lookup)

    doc = _get_nlp()("Japan offers many options.")
    candidate = heuristics.evaluate_candidate_heuristics(
        "Japan",
        source_text="Japan offers many options.",
        mention_count=1,
        doc=doc,
    )

    assert candidate.country_name is True
    assert candidate.should_keep() is False


def test_extract_brand_analysis_filters_demonyms_regression() -> None:
    answer = (
        "Japanese users often pick Spotify. "
        "German listeners also compare Apple Music with Spotify. "
        "Chinese audiences may still evaluate YouTube Music options."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Japanese" not in names
    assert "German" not in names
    assert "Chinese" not in names
    assert "Spotify" in names
    assert "Apple Music" in names
    assert "YouTube Music" in names


def test_extract_brand_analysis_keeps_multiword_brand_not_components() -> None:
    answer = (
        "Apple Music is a top choice. "
        "Many users compare Apple Music with Spotify."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Apple Music" in names
    assert "Apple" not in names
    assert "Music" not in names
    assert "Spotify" in names


def test_extract_brand_analysis_filters_generic_noise_words() -> None:
    answer = (
        "Overview: Large catalog and deep features are useful. "
        "Spotify is still recommended."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Large" not in names
    assert "Deep" not in names
    assert "Catalog" not in names
    assert "Spotify" in names


def test_extract_brand_analysis_filters_structural_noise_terms() -> None:
    answer = (
        "```md\n"
        "Short answer: Spotify is a strong option.\n"
        "- Seamless ecosystem and quality improvements are notable.\n"
        "Among options, Apple Music is also strong."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "md Short" not in names
    assert "Seamless" not in names
    assert "Among" not in names
    assert "Spotify" in names
    assert "Apple Music" in names


def test_extract_brand_analysis_filters_device_platform_terms() -> None:
    answer = (
        "Apple Music works on iPhone and iPad. "
        "Spotify also works on Android and via Alexa."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Apple Music" in names
    assert "Spotify" in names
    assert "iPhone" not in names
    assert "iPad" not in names
    assert "Android" not in names
    assert "Alexa" not in names


def test_extract_brand_analysis_filters_platform_feature_tokens() -> None:
    answer = (
        "Apple Music supports iOS and Dolby Atmos with Spatial Audio. "
        "Spotify is also available on TVs and PCs."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "Apple Music" in names
    assert "Spotify" in names
    assert "iOS" not in names
    assert "Dolby" not in names
    assert "Atmos" not in names
    assert "Spatial" not in names
    assert "Audio" not in names
    assert "TVs" not in names
    assert "PCs" not in names


def test_extract_brand_analysis_drops_multiword_subfragments() -> None:
    answer = (
        "NetEase Cloud Music is widely used. "
        "NetEase Cloud Music offers social features."
    )

    names = [item.name for item in extract_brand_analysis(answer)]

    assert "NetEase Cloud Music" in names
    assert "NetEase Cloud" not in names
    assert "Cloud Music" not in names


def test_lookup_wikidata_entity_types_uses_local_snapshot() -> None:
    heuristics._load_wikidata_entity_snapshot.cache_clear()
    heuristics.lookup_wikidata_entity_types.cache_clear()

    assert heuristics.lookup_wikidata_entity_types("Spotify") == (True, False)
    assert heuristics.lookup_wikidata_entity_types("UnknownCandidate") == (None, None)