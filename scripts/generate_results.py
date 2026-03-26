import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app


def _extract_payload_fields(item: dict[str, Any]) -> tuple[str, str]:
	content = item.get("payload", {}).get("results", [{}])[0].get("content", {})
	question = str(content.get("prompt_query", "")).strip()
	answer = str(content.get("answer_results_md", "")).strip()
	return question, answer


def generate_results(input_path: Path, output_dir: Path) -> None:
	with input_path.open("r", encoding="utf-8") as f:
		raw = json.load(f)

	items = raw.get("items", [])
	output_dir.mkdir(parents=True, exist_ok=True)

	client = TestClient(app)
	manifest: dict[str, Any] = {
		"input_file": str(input_path),
		"records_total": len(items),
		"generated_files": [],
	}

	for index, item in enumerate(items, start=1):
		question, answer = _extract_payload_fields(item)
		if not question or not answer:
			continue

		response = client.post("/analysis", json={"question": question, "answer": answer})
		response.raise_for_status()

		out_name = f"analysis_{index:04d}.json"
		out_path = output_dir / out_name
		with out_path.open("w", encoding="utf-8") as out_file:
			json.dump(response.json(), out_file, indent=2, ensure_ascii=False)
			out_file.write("\n")

		manifest["generated_files"].append(out_name)

	manifest_path = output_dir / "manifest.json"
	with manifest_path.open("w", encoding="utf-8") as manifest_file:
		json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
		manifest_file.write("\n")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate analysis outputs from llm_chats.json")
	parser.add_argument("--input", required=True, help="Path to llm_chats.json")
	parser.add_argument("--output-dir", required=True, help="Directory to store analysis outputs")
	args = parser.parse_args()

	generate_results(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
	main()
