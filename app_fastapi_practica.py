from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Práctica MLOps y LLMOps",
    description="API con endpoints básicos y dos pipelines de Hugging Face",
    version="1.0.0",
)

# Pipelines Hugging Face: se cargan una vez al iniciar la API
sentiment_pipeline = pipeline("sentiment-analysis")
translation_pipeline = pipeline("translation_en_to_fr")

class TextRequest(BaseModel):
    text: str

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "API de práctica MLOps y LLMOps funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "project": "Clasificación con MLflow y despliegue con FastAPI",
        "author": "Diana Ospina",
        "modules": 5,
    }

@app.post("/sentiment")
def sentiment(request: TextRequest):
    result = sentiment_pipeline(request.text)
    return {"input": request.text, "result": result}

@app.post("/translate")
def translate(request: TextRequest):
    result = translation_pipeline(request.text)
    return {"input": request.text, "translation": result}

@app.post("/predict_iris_rule")
def predict_iris_rule(request: PredictRequest):
    """Endpoint sencillo de ejemplo con reglas para clasificar Iris."""
    if request.petal_width < 0.8:
        prediction = "setosa"
    elif request.petal_width < 1.8:
        prediction = "versicolor"
    else:
        prediction = "virginica"

    return {
        "features": request.model_dump(),
        "prediction": prediction,
    }
