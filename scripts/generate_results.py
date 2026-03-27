import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from app.service import analyze


def _extract_payload_fields(item: dict[str, Any]) -> tuple[str, str]:
	content = item.get("payload", {}).get("results", [{}])[0].get("content", {})
	question = str(content.get("prompt_query", "")).strip()
	answer = str(content.get("answer_results_md", "")).strip()
	return question, answer


def _process_records_chunk(chunk: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
	results: list[tuple[int, dict[str, Any]]] = []
	for index, item in chunk:
		question, answer = _extract_payload_fields(item)
		if not question or not answer:
			continue

		analysis = analyze(question, answer)
		results.append((index, analysis.model_dump(mode="json")))

	return results


def _iter_chunks(
	indexed_items: list[tuple[int, dict[str, Any]]], chunk_size: int
) -> list[list[tuple[int, dict[str, Any]]]]:
	return [indexed_items[i : i + chunk_size] for i in range(0, len(indexed_items), chunk_size)]


def _safe_worker_count(workers: int) -> int:
	if workers < 1:
		return 1
	return min(workers, os.cpu_count() or 1)


def _auto_tune_settings(record_count: int) -> tuple[int, int]:
	"""Choose safe defaults for CPU-bound batch analysis based on workload size."""
	if record_count <= 0:
		return 1, 32

	cpu_count = os.cpu_count() or 1
	worker_cap = max(1, min(cpu_count, 8))

	if record_count <= 64:
		return 1, max(8, min(64, record_count))

	workers = min(worker_cap, max(2, math.ceil(record_count / 200)))
	chunk_size = max(16, min(128, math.ceil(record_count / (workers * 4))))
	return workers, chunk_size


def generate_results(
	input_path: Path,
	output_dir: Path,
	*,
	workers: int = 1,
	chunk_size: int = 32,
	auto_tune: bool = False,
) -> None:
	started_at = time.perf_counter()

	with input_path.open("r", encoding="utf-8") as f:
		raw = json.load(f)

	items = raw.get("items", [])
	indexed_items = list(enumerate(items, start=1))
	output_dir.mkdir(parents=True, exist_ok=True)

	manifest: dict[str, Any] = {
		"input_file": str(input_path),
		"records_total": len(items),
		"generated_files": [],
	}

	if auto_tune:
		tuned_workers, tuned_chunk_size = _auto_tune_settings(len(items))
		worker_count = _safe_worker_count(tuned_workers)
		effective_chunk_size = tuned_chunk_size
	else:
		if chunk_size < 1:
			raise ValueError("chunk_size must be >= 1")
		worker_count = _safe_worker_count(workers)
		effective_chunk_size = chunk_size

	manifest["execution"] = {
		"auto_tuned": auto_tune,
		"workers": worker_count,
		"chunk_size": effective_chunk_size,
	}

	results: list[tuple[int, dict[str, Any]]] = []

	if worker_count == 1:
		for index, item in indexed_items:
			question, answer = _extract_payload_fields(item)
			if not question or not answer:
				continue
			analysis = analyze(question, answer)
			results.append((index, analysis.model_dump(mode="json")))
	else:
		chunks = _iter_chunks(indexed_items, effective_chunk_size)
		with ProcessPoolExecutor(max_workers=worker_count) as executor:
			for chunk_results in executor.map(_process_records_chunk, chunks):
				results.extend(chunk_results)

	results.sort(key=lambda item: item[0])

	for index, payload in results:
		out_name = f"analysis_{index:04d}.json"
		out_path = output_dir / out_name
		with out_path.open("w", encoding="utf-8") as out_file:
			json.dump(payload, out_file, indent=2, ensure_ascii=False)
			out_file.write("\n")

		manifest["generated_files"].append(out_name)

	elapsed_seconds = time.perf_counter() - started_at
	manifest["execution"]["elapsed_seconds"] = round(elapsed_seconds, 3)

	manifest_path = output_dir / "manifest.json"
	with manifest_path.open("w", encoding="utf-8") as manifest_file:
		json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
		manifest_file.write("\n")

	print(
		f"Generation finished in {elapsed_seconds:.2f}s "
		f"(records: {len(items)}, generated: {len(manifest['generated_files'])}, workers: {worker_count}, chunk_size: {effective_chunk_size})"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate analysis outputs from llm_chats.json")
	parser.add_argument("--input", required=True, help="Path to llm_chats.json")
	parser.add_argument("--output-dir", required=True, help="Directory to store analysis outputs")
	parser.add_argument(
		"--workers",
		type=int,
		default=1,
		help="Number of process workers for analysis (default: 1).",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=32,
		help="Records per process task chunk when workers > 1 (default: 32).",
	)
	parser.add_argument(
		"--auto-tune",
		action="store_true",
		help="Auto-select workers and chunk size from dataset size and CPU count.",
	)
	args = parser.parse_args()

	generate_results(
		Path(args.input),
		Path(args.output_dir),
		workers=args.workers,
		chunk_size=args.chunk_size,
		auto_tune=args.auto_tune,
	)


if __name__ == "__main__":
	main()
