import json
from pathlib import Path

from scripts.generate_results import generate_results


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