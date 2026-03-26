from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_analysis_endpoint_returns_brand_structure() -> None:
	payload = {
		"question": "What providers are best for startups?",
		"answer": (
			"OpenAI [openai.com] offers strong feature coverage and reliable APIs. "
			"Anthropic emphasizes safety and long context windows. "
			"OpenAI pricing can be high for some workloads."
		),
	}

	response = client.post("/analysis", json=payload)
	assert response.status_code == 200

	body = response.json()
	assert "brands" in body
	assert len(body["brands"]) >= 2

	openai = next(item for item in body["brands"] if item["name"] == "OpenAI")
	assert openai["mentions_count"] == 2
	assert "features" in openai["scopes"] or "pricing" in openai["scopes"]
	assert openai["domain"] == "https://openai.com"


def test_analysis_endpoint_ignores_question_words_as_brands() -> None:
	payload = {
		"question": "What is the best option?",
		"answer": "Spotify and Deezer are both popular choices.",
	}

	response = client.post("/analysis", json=payload)
	assert response.status_code == 200
	brand_names = [item["name"] for item in response.json()["brands"]]

	assert "What" not in brand_names
	assert "Spotify" in brand_names
	assert "Deezer" in brand_names


def test_analysis_endpoint_validates_input() -> None:
	response = client.post("/analysis", json={"question": " ", "answer": " "})
	assert response.status_code == 400
