from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

detector = pipeline(
    "text-classification",
    model="roberta-base-openai-detector"
)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(request: TextRequest):
    result = detector(request.text)[0]
    return {
        "prediction": result["label"],
        "confidence": f"{round(result['score'] * 100, 2)}%"
    }
