import argparse
import json
from pathlib import Path
import re
from typing import Any

from app.extract import extract_brand_analysis
from app.preprocess import clean_answer_text


def _extract_qa(item: dict[str, Any]) -> tuple[str, str]:
    content = item.get("payload", {}).get("results", [{}])[0].get("content", {})
    question = str(content.get("prompt_query", "")).strip()
    answer = str(content.get("answer_results_md", "")).strip()
    return question, answer


def _is_boundary(text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isalnum()
    return left_ok and right_ok


def _find_brand_spans(text: str, brand_names: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    # Prefer longer names first to avoid overlaps (e.g., Apple Music before Apple).
    unique = sorted(set(brand_names), key=len, reverse=True)
    occupied: list[tuple[int, int]] = []

    for brand in unique:
        pattern = re.compile(re.escape(brand), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if not _is_boundary(text, start, end):
                continue
            if any(not (end <= os or start >= oe) for os, oe in occupied):
                continue
            occupied.append((start, end))
            spans.append((start, end, "BRAND"))

    return sorted(spans, key=lambda item: item[0])


def prepare_training_data(llm_chats_path: Path, output_path: Path, max_records: int) -> int:
    with llm_chats_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("items", [])
    total_items = len(items)
    use_records = total_items if max_records <= 0 else min(total_items, max_records)
    examples: list[dict[str, Any]] = []

    for idx in range(1, use_records + 1):
        _, answer = _extract_qa(items[idx - 1])
        if not answer:
            continue

        cleaned = clean_answer_text(answer)
        extracted = extract_brand_analysis(cleaned)
        brand_names = [item.name for item in extracted]
        spans = _find_brand_spans(cleaned, brand_names)
        if not spans:
            continue

        examples.append({"text": cleaned, "entities": spans, "source": f"llm_chat_{idx:04d}"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpaCy NER training data from llm_chats by extracting brands directly")
    parser.add_argument("--llm-chats", default="llm_chats.json", help="Path to llm_chats.json")
    parser.add_argument("--output", default="training/brand_ner.jsonl", help="Output training JSONL path")
    parser.add_argument("--max-records", type=int, default=0, help="How many records to use from llm_chats (0 = all)")
    args = parser.parse_args()

    count = prepare_training_data(
        llm_chats_path=Path(args.llm_chats),
        output_path=Path(args.output),
        max_records=args.max_records,
    )
    print(f"Prepared {count} training examples at {args.output}")


if __name__ == "__main__":
    main()
