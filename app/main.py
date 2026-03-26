from fastapi import FastAPI, HTTPException

from app.schemas import AnalysisRequest, AnalysisResponse
from app.service import analyze

app = FastAPI(title="Brand Analysis API", version="1.0.0")


@app.post("/analysis", response_model=AnalysisResponse)
def run_analysis(payload: AnalysisRequest) -> AnalysisResponse:
    if not payload.question.strip() or not payload.answer.strip():
        raise HTTPException(status_code=400, detail="question and answer must be non-empty")
    return analyze(payload.question, payload.answer)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
