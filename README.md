
Small Python backend service that transforms a user question plus a noisy LLM answer into a clean, brand-centric analysis.

## Project Structure

```text
.
|-- app/
|   |-- __init__.py
|   |-- main.py              # FastAPI app and HTTP routes
|   |-- schemas.py           # Typed Pydantic models
|   |-- preprocess.py        # Pre-processing pipeline
|   |-- extract.py           # Brand extraction + post-processing
|   `-- taxonomy.py          # Scope taxonomy labels and matching keywords
|-- scripts/
|   |-- __init__.py
|   `-- generate_results.py  # Calls POST /analysis for llm_chats.json records
|-- tests/
|   |-- test_analysis_api.py
|   `-- test_preprocess.py
|-- results/
|   `-- .gitkeep
|-- requirements.txt
|-- pytest.ini
`-- README.md
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
			"domain": "openai.com"
		}
	]
}
```

Notes:
- mentions_count is exact string-mention count for extracted brand names.
- Brand order is preserved based on first appearance in processed text.
- domain is only populated when grounded by explicit URL evidence; otherwise null.

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
- infer domain only from explicit URL evidence in brand-local context

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Service

```bash
uvicorn app.main:app --reload
```

## Run Tests

```bash
pytest
```

## Generate Results From Dataset

Put llm_chats.json in the repository root and run:

```bash
python -m scripts.generate_results --input llm_chats.json --output-dir results
```

This command reads each dataset record from the expected payload shape,
calls POST /analysis, and writes output files:
- results/analysis_0001.json, results/analysis_0002.json, ...
- results/manifest.json
