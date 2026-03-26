from collections.abc import Iterable
import re

SCOPE_TAXONOMY: dict[str, tuple[str, ...]] = {
	"performance": (
		"latency",
		"fast",
		"faster",
		"speed",
		"throughput",
		"benchmark",
		"quality",
		"lossless",
		"hi-res",
		"audio quality",
	),
	"pricing": (
		"price",
		"pricing",
		"cost",
		"plan",
		"plans",
		"tier",
		"tiers",
		"affordable",
		"value",
		"discount",
		"student",
		"family",
		"bundle",
	),
	"safety": (
		"safety",
		"safe",
		"compliance",
		"privacy",
		"moderation",
		"risk",
		"secure",
		"security",
		"gdpr",
	),
	"features": (
		"feature",
		"features",
		"integration",
		"integrates",
		"api",
		"ecosystem",
		"tool",
		"tools",
		"multimodal",
		"playlist",
		"discovery",
		"social",
	),
	"reliability": (
		"reliability",
		"reliable",
		"stable",
		"stability",
		"uptime",
		"consistent",
		"consistency",
		"production",
		"trust",
	),
	"adoption": (
		"popular",
		"widely used",
		"market share",
		"users",
		"mau",
		"maus",
		"adoption",
		"community",
		"ecosystem",
		"dominates",
	),
}

SCOPE_ORDER: tuple[str, ...] = tuple(SCOPE_TAXONOMY.keys())


def _compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
	escaped = re.escape(keyword)
	# Allow flexible whitespace for multi-word keywords like "market share".
	escaped = escaped.replace(r"\ ", r"\s+")
	return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


SCOPE_TAXONOMY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
	label: tuple(_compile_keyword_pattern(keyword) for keyword in keywords)
	for label, keywords in SCOPE_TAXONOMY.items()
}


def get_scopes_from_contexts(contexts: Iterable[str]) -> list[str]:
	matched: list[str] = []
	context_list = list(contexts)

	for label in SCOPE_ORDER:
		patterns = SCOPE_TAXONOMY_PATTERNS[label]
		if any(pattern.search(context) for context in context_list for pattern in patterns):
			matched.append(label)

	return matched
