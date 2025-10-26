import pickle
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn


# request
class Lead(BaseModel):
    lead_source: Literal[
        "organic_search", "social_media", "paid_ads", "referral", "events", "NA"
    ]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


# response
class PredictResponse(BaseModel):
    lead_probability: float
    lead: bool


app = FastAPI(title="lead-score-prediction")

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict_single(lead):
    result = pipeline.predict_proba(lead)[0, 1]
    return float(result)


@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.model_dump())

    return PredictResponse(lead_probability=prob, lead=prob >= 0.5)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
