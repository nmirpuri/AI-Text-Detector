from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ ADD THIS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (OK for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    return {
        "prediction": result["label"],
        "confidence": f"{round(result['score'] * 100, 2)}%"
    }
