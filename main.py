from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

detector = pipeline(
    "text-classification",
    model="roberta-base-openai-detector",
    tokenizer="roberta-base-openai-detector"
)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(request: TextRequest):
    result = detector(request.text)[0]

    label = result["label"]
    score = round(result["score"] * 100, 2)

    return {
        "prediction": label,
        "confidence": f"{score}%",
        "note": "AI detection is probabilistic, not definitive."
    }
