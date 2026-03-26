import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor_service import run_prediction


def _parse_origins(raw_value):
    if not raw_value:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


class PredictionRequest(BaseModel):
    data_source: str = "sample"
    historical_matches: Optional[str] = None
    upcoming_fixtures: Optional[str] = None
    team_context: Optional[str] = None
    gameweek: Optional[int] = None
    future_gameweek_only: bool = False
    competition_code: str = "PL"
    api_token: Optional[str] = None
    historical_seasons: Optional[str] = None
    season: Optional[int] = None


app = FastAPI(title="Goborr Predictor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(os.getenv("ALLOWED_ORIGINS")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        payload, _ = run_prediction(
            data_source=request.data_source,
            historical_matches=request.historical_matches,
            upcoming_fixtures=request.upcoming_fixtures,
            team_context=request.team_context,
            gameweek=request.gameweek,
            future_gameweek_only=request.future_gameweek_only,
            competition_code=request.competition_code,
            api_token=request.api_token,
            historical_seasons=request.historical_seasons,
            season=request.season,
        )
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
