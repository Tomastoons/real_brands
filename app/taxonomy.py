from collections.abc import Iterable

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


def get_scopes_from_contexts(contexts: Iterable[str]) -> list[str]:
	matched: list[str] = []
	lowered_contexts = [ctx.lower() for ctx in contexts]

	for label in SCOPE_ORDER:
		keywords = SCOPE_TAXONOMY[label]
		if any(keyword in context for context in lowered_contexts for keyword in keywords):
			matched.append(label)

	return matched
