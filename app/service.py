from app.extract import extract_brand_analysis
from app.preprocess import build_user_message, clean_answer_text
from app.schemas import AnalysisResponse, BrandAnalysis


def analyze(question: str, answer: str) -> AnalysisResponse:
	cleaned_answer = clean_answer_text(answer)
	user_input = build_user_message(question, cleaned_answer)

	brands = extract_brand_analysis(user_input)
	return AnalysisResponse(
		brands=[
			BrandAnalysis(
				name=item.name,
				mentions_count=item.mentions_count,
				scopes=item.scopes,
				domain=item.domain,
			)
			for item in brands
		]
	)
