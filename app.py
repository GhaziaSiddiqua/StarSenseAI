
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="StarSense AI")

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

model = joblib.load(
    BASE_DIR / "starsense_model.joblib"
)

STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}


class StarInput(BaseModel):
    temperature: float = Field(gt=0)
    luminosity: float = Field(gt=0)
    radius: float = Field(gt=0)
    magnitude: float
    color: str = Field(min_length=1)
    spectral_class: str = Field(min_length=1)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_star(star: StarInput):
    features = pd.DataFrame([{
        "Temperature (K)": star.temperature,
        "Luminosity(L/Lo)": star.luminosity,
        "Radius(R/Ro)": star.radius,
        "Absolute magnitude(Mv)": star.magnitude,
        "Star color": star.color,
        "Spectral Class": star.spectral_class
    }])

    try:
        prediction = int(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not process these measurements."
        ) from exc

    return {
        "star_type": STAR_TYPES[prediction],
        "class_id": prediction,
        "message": (
            "Prediction from the StarSense AI "
            "educational machine-learning model."
        )
    }
