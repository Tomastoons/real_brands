import json
from pathlib import Path

from scripts.generate_results import _auto_tune_settings, generate_results


def test_generate_results_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "sample_llm_chats.json"
    output_dir = tmp_path / "results"

    sample = {
        "items": [
            {
                "file_name": "001.json",
                "payload": {
                    "results": [
                        {
                            "content": {
                                "prompt_query": "Best provider?",
                                "answer_results_md": "OpenAI [openai.com] has strong API features.",
                            }
                        }
                    ]
                },
            }
        ]
    }

    input_path.write_text(json.dumps(sample), encoding="utf-8")
    generate_results(input_path=input_path, output_dir=output_dir)

    analysis_file = output_dir / "analysis_0001.json"
    manifest_file = output_dir / "manifest.json"

    assert analysis_file.exists()
    assert manifest_file.exists()

    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert "brands" in analysis
    assert manifest["records_total"] == 1
    assert "analysis_0001.json" in manifest["generated_files"]


def test_auto_tune_settings_scale_with_dataset(monkeypatch) -> None:
    monkeypatch.setattr("scripts.generate_results.os.cpu_count", lambda: 8)

    workers_small, chunk_small = _auto_tune_settings(20)
    workers_large, chunk_large = _auto_tune_settings(1000)

    assert workers_small == 1
    assert 8 <= chunk_small <= 64
    assert 2 <= workers_large <= 8
    assert 16 <= chunk_large <= 128


def test_generate_results_auto_tune_includes_execution_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "sample_llm_chats.json"
    output_dir = tmp_path / "results"

    sample = {
        "items": [
            {
                "file_name": "001.json",
                "payload": {
                    "results": [
                        {
                            "content": {
                                "prompt_query": "Best provider?",
                                "answer_results_md": "OpenAI [openai.com] has strong API features.",
                            }
                        }
                    ]
                },
            }
        ]
    }

    input_path.write_text(json.dumps(sample), encoding="utf-8")
    generate_results(input_path=input_path, output_dir=output_dir, auto_tune=True)

    manifest_file = output_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert manifest["execution"]["auto_tuned"] is True
    assert manifest["execution"]["workers"] >= 1
    assert manifest["execution"]["chunk_size"] >= 1