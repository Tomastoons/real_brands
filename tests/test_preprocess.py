from app.preprocess import build_user_message, clean_answer_text


def test_clean_answer_text_normalizes_citations_and_spacing() -> None:
	answer = "OpenAI [1] is fast. See https://openai.com,\n\nAnthropic [2, 3] too."
	cleaned = clean_answer_text(answer)

	assert "[1]" not in cleaned
	assert "[2, 3]" not in cleaned
	assert "\n" not in cleaned
	assert "https://openai.com ," not in cleaned


def test_build_user_message_combines_fields() -> None:
	result = build_user_message("Best model?", "OpenAI is popular.")
	assert result.startswith("Question: Best model?")
	assert "Answer: OpenAI is popular." in result


def test_clean_answer_text_keeps_domain_tokens_intact() -> None:
	answer = "OpenAI [openai.com] is useful."
	cleaned = clean_answer_text(answer)
	assert "openai. com" not in cleaned
	assert "openai.com" in cleaned
