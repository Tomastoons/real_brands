
Small Python backend service that transforms a user question plus a noisy LLM answer into a clean, brand-centric analysis.

## Project Structure

```text
.
|-- llm_chats.json
|-- pytest.ini
|-- README.md
|-- requirements.txt
|-- app/
|   |-- __init__.py
|   |-- main.py              # FastAPI app and HTTP routes
|   |-- extract.py           # Brand extraction + post-processing
|   |-- extract_candidates.py
|   |-- extract_domains.py
|   |-- extract_pipeline.py
|   |-- extract_shared.py
|   |-- heuristics.py
|   |-- preprocess.py        # Pre-processing pipeline
|   |-- schemas.py           # Typed Pydantic models
|   |-- service.py
|   `-- taxonomy.py          # Scope taxonomy labels and matching keywords
|-- scripts/
|   |-- __init__.py
|   |-- generate_results.py
|   |-- prepare_spacy_brand_data.py
|   `-- train_spacy_brand_model.py
|-- tests/
|   |-- test_analysis_api.py
|   |-- test_extract.py
|   |-- test_generate_results.py
|   |-- test_preprocess.py
|   `-- test_taxonomy.py
|-- training/
|   `-- brand_ner.jsonl
|-- models/
|   `-- brand_ner/
|-- results/
|   |-- analysis_0001.json
|   |-- analysis_0002.json
|   |-- ...
|   `-- analysis_00xx.json
`-- app/models/
	`-- brand_ner/
```

## Scope Taxonomy

| Label | Definition | Examples |
|---|---|---|
| performance | Speed, latency, throughput, and technical capability discussion. | "low latency", "better benchmark" |
| pricing | Cost, token pricing, plans, and value claims. | "cheaper per token", "enterprise pricing" |
| safety | Safety, compliance, privacy, moderation, or risk controls. | "GDPR", "content moderation" |
| features | Product features, integrations, APIs, and workflow fit. | "tool calling", "multimodal support" |
| reliability | Stability, consistency, uptime, and operational trust. | "stable output", "production reliability" |
| adoption | Market usage, ecosystem strength, and popularity. | "widely adopted", "strong ecosystem" |
| sentiment | Expressed opinion or evaluation of the brand. | "best option", "disappointing experience" |
| content_type | Types of content offered or emphasized by the brand. | "exclusive podcast", "hi-fi streaming", "audiobooks" |

## API

### POST /analysis

Request:

```json
{
	"question": "Which model is best for startup support?",
	"answer": "Long markdown/prose LLM output mentioning multiple brands..."
}
```

Response:

```json
{
	"brands": [
		{
			"name": "OpenAI",
			"mentions_count": 2,
			"scopes": ["features", "pricing"],
			"domain": "https://openai.com",
			"price_tiers": ["free trial", "premium"]
		}
	]
}
```

Notes:
- mentions_count is exact string-mention count for extracted brand names.
- Brand order is preserved based on first appearance in processed text.
- domain is inferred dynamically from URL/domain evidence in text using proximity and token-overlap scoring; if evidence is conflicting, domain is null.
- price_tiers is a deduplicated list of pricing tier labels detected in sentences near the brand (e.g. `"free"`, `"free trial"`, `"premium"`, `"student plan"`, `"family plan"`, `"duo plan"`, `"bundle"`); empty list if none detected.

## Pre/Post Processing

Pre-process behavior:
- remove citation markers like [1], [2,3]
- normalize whitespace and punctuation spacing
- preserve URLs/source evidence
- combine into internal message format: "Question: ... Answer: ..."

Post-process behavior:
- extract ordered brand candidates from cleaned text
- compute exact mention counts
- infer taxonomy scopes from sentence-level context keywords
- infer domain dynamically from explicit URL/domain evidence near brand mentions (no static brand-domain dictionary)

## Model Files

The extractor loads a SpaCy model from:
- `models/brand_ner` (default)
- or `BRAND_NER_MODEL_PATH` if set

Important:
- `models/brand_ner/vocab/vectors` is a large file and is tracked with Git LFS.
- If you clone this repository, install and fetch LFS objects before running the app:

```bash
git lfs install
git lfs pull
```

If the model path is missing, the app falls back to `en_core_web_lg`.

## Setup

Requires **Python 3.10+**.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> `en_core_web_lg` is the base SpaCy model used as a fallback and for training. Skip this if you are using a pre-trained model from the repository.

## Run Service

```bash
uvicorn app.main:app --reload
```

## Run Tests

```bash
pytest
```

`pytest.ini` is configured so tests resolve local imports from the repository root.

## Generate Results From Dataset

Place the provided `llm_chats.json` file (supplied with the task) in the repository root, then run:

```bash
python -m scripts.generate_results --input llm_chats.json --output-dir results
```

This command reads each dataset record from the expected payload shape,
calls POST /analysis, and writes output files:
- results/analysis_0001.json, results/analysis_0002.json, ...
- results/manifest.json

> The `/results` folder is not pre-populated in the repository. Run this command against your copy of `llm_chats.json` to reproduce the outputs. The service must be running (`uvicorn app.main:app --reload`) before invoking this script.

## Scripts Reference

### scripts/generate_results.py

Purpose:
- Batch-runs the API analysis pipeline for each record in `llm_chats.json`.
- Useful for evaluation snapshots, QA, and assignment deliverables.

Command:

```bash
python -m scripts.generate_results --input llm_chats.json --output-dir results
```

Arguments:
- `--input` (required): path to dataset JSON file.
- `--output-dir` (required): folder where `analysis_XXXX.json` and `manifest.json` are written.

### scripts/prepare_spacy_brand_data.py

Purpose:
- Creates weakly-labeled BRAND NER training data from dataset answers.
- It cleans answer text, extracts brand names with current extraction logic, then writes SpaCy JSONL records.

Command:

```bash
python -m scripts.prepare_spacy_brand_data --llm-chats llm_chats.json --output training/brand_ner.jsonl --max-records 0
```

Arguments:
- `--llm-chats` (default: `llm_chats.json`): input dataset.
- `--output` (default: `training/brand_ner.jsonl`): output JSONL file for training.
- `--max-records` (default: `0`): number of records to use (`0` means all).

### scripts/prepare_user_brand_training_data.py

Purpose:
- Writes curated/user-provided training examples to JSONL.
- Applies auto-correction for entity spans so data is cleaner for SpaCy NER training.

Command:

```bash
python -m scripts.prepare_user_brand_training_data --output training/user_brand_data.jsonl
```

Arguments:
- `--output` (default: `training/user_brand_data.jsonl`): destination JSONL path.

### scripts/train_spacy_brand_model.py

Purpose:
- Trains the SpaCy NER model used by the extractor.
- Saves a model directory that can be loaded by the app via default path or env override.

Command:

```bash
python -m scripts.train_spacy_brand_model --train-data training/brand_ner.jsonl --output-dir models/brand_ner --base-model en_core_web_lg --iterations 15 --seed 13
```

Arguments:
- `--train-data` (default: `training/brand_ner.jsonl`): JSONL training examples.
- `--output-dir` (default: `models/brand_ner`): where trained model artifacts are saved.
- `--base-model` (default: `en_core_web_lg`): initial SpaCy model to fine-tune.
- `--iterations` (default: `15`): number of training epochs.
- `--seed` (default: `13`): random seed for reproducibility.

## Recommended Training Workflow

1. Generate weak labels from your dataset:

```bash
python -m scripts.prepare_spacy_brand_data --llm-chats llm_chats.json --output training/brand_ner.jsonl
```

3. Train model from prepared training JSONL:

```bash
python -m scripts.train_spacy_brand_model --train-data training/brand_ner.jsonl --output-dir models/brand_ner
```

4. Run tests to verify behavior after retraining:

```bash
pytest
```

5. Generate dataset outputs with the newly trained model:

```bash
python -m scripts.generate_results --input llm_chats.json --output-dir results
```
