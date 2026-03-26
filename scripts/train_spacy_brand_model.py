import argparse
import json
from pathlib import Path
import random

import spacy
from spacy.training import Example


def _load_examples(jsonl_path: Path) -> list[dict]:
    examples: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _build_spacy_examples(nlp: spacy.language.Language, records: list[dict]) -> list[Example]:
    examples: list[Example] = []
    for record in records:
        text = record["text"]
        doc = nlp.make_doc(text)
        aligned_ents: list[tuple[int, int, str]] = []
        for start, end, label in record.get("entities", []):
            s = int(start)
            e = int(end)
            span = doc.char_span(s, e, label=label, alignment_mode="contract")
            if span is None:
                continue
            text = span.text.strip()
            if not any(ch.isalnum() for ch in text):
                continue
            if label == "URL" and "." not in text:
                continue
            aligned_ents.append((span.start_char, span.end_char, label))
        examples.append(Example.from_dict(doc, {"entities": aligned_ents}))
    return examples


def train_brand_model(training_data: Path, output_dir: Path, base_model: str, iterations: int, seed: int) -> None:
    random.seed(seed)

    records = _load_examples(training_data)
    if not records:
        raise RuntimeError(f"No training records found in {training_data}")

    nlp = spacy.load(base_model)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    labels = {
        str(label)
        for record in records
        for _, _, label in record.get("entities", [])
    }
    for label in sorted(labels):
        ner.add_label(label)

    examples = _build_spacy_examples(nlp, records)
    disabled = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*disabled):
        optimizer = nlp.resume_training()
        for i in range(iterations):
            random.shuffle(examples)
            losses: dict[str, float] = {}
            for example in examples:
                nlp.update([example], drop=0.15, sgd=optimizer, losses=losses)
            print(f"iteration={i + 1}/{iterations} losses={losses}")

    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Saved trained model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SpaCy BRAND NER model")
    parser.add_argument("--train-data", default="training/brand_ner.jsonl", help="Path to training JSONL data")
    parser.add_argument("--output-dir", default="models/brand_ner", help="Directory to save the trained model")
    parser.add_argument("--base-model", default="en_core_web_lg", help="Base SpaCy model name/path")
    parser.add_argument("--iterations", type=int, default=15, help="Training iterations")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    train_brand_model(
        training_data=Path(args.train_data),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        iterations=args.iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
