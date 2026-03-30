import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from predictor_service import run_prediction


def _parse_origins(raw_value):
    if not raw_value:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data_source: str = Field(default="sample", alias="dataSource")
    historical_matches: Optional[str] = Field(default=None, alias="historicalMatches")
    upcoming_fixtures: Optional[str] = Field(default=None, alias="upcomingFixtures")
    team_context: Optional[str] = Field(default=None, alias="teamContext")
    gameweek: Optional[int] = None
    future_gameweek_only: bool = Field(default=False, alias="futureGameweekOnly")
    competition_code: str = Field(default="PL", alias="competitionCode")
    api_token: Optional[str] = Field(default=None, alias="apiToken")
    historical_seasons: Optional[str] = Field(default=None, alias="historicalSeasons")
    season: Optional[int] = None
    use_gemini_context: bool = Field(default=True, alias="useGeminiContext")
    gemini_api_key: Optional[str] = Field(default=None, alias="geminiApiKey")
    gemini_model: Optional[str] = Field(default=None, alias="geminiModel")
    gemini_context_output_path: Optional[str] = Field(
        default=None, alias="geminiContextOutputPath"
    )


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
            use_gemini_context=request.use_gemini_context,
            gemini_api_key=request.gemini_api_key,
            gemini_model=request.gemini_model,
            gemini_context_output_path=request.gemini_context_output_path,
        )
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
