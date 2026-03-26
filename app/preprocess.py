import re

# Citation markers often look like [1] or [2, 3] and are removed for cleaner parsing.
CITATION_MARKERS_PATTERN = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
URL_WITH_TRAILING_PUNCT_PATTERN = re.compile(r"(https?://[^\s)\]>]+)([.,;:!?])")


def clean_answer_text(answer: str) -> str:
	"""Normalize artifacts in LLM answer while keeping extraction evidence."""
	text = answer.strip()

	text = CITATION_MARKERS_PATTERN.sub("", text)
	text = URL_WITH_TRAILING_PUNCT_PATTERN.sub(r"\1 \2", text)

	text = re.sub(r"[\t\r\n]+", " ", text) # Replace tabs and newlines with a single space to normalize spacing.
	text = re.sub(r"\s+", " ", text) #Replaces multiple consecutive spaces with a single space.
	text = re.sub(r"\s+([,.;:!?])", r"\1", text) # Remove space before punctuation e.g., hello , becomes hello,
	text = re.sub(r"([,;:!?])(\w)", r"\1 \2", text) # Ensure space after punctuation if followed by a word character, but avoid splitting domains like openai.com.

	return text.strip()


def build_user_message(question: str, cleaned_answer: str) -> str:
	question_clean = re.sub(r"\s+", " ", question.strip())
	return f"Question: {question_clean} Answer: {cleaned_answer}".strip()
