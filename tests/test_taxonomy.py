from app.taxonomy import get_scopes_from_contexts


def test_taxonomy_uses_word_boundaries_for_keywords() -> None:
    contexts = ["A rapid rollout improved user onboarding."]

    scopes = get_scopes_from_contexts(contexts)

    # "api" should not match inside "rapid".
    assert "features" not in scopes


def test_taxonomy_matches_multiword_keywords() -> None:
    contexts = ["The service has strong market share and is widely used."]

    scopes = get_scopes_from_contexts(contexts)

    assert "adoption" in scopes
