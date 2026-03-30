import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from data_processor import DataProcessor
from gemini_enrichment import (
    GeminiTeamContextEnricher,
    calibrate_team_context,
    merge_team_context_frames,
)
from prediction_model import MatchPredictor
from scraper import EPLScraper


def parse_seasons(raw_value):
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def run_prediction(
    data_source="sample",
    historical_matches=None,
    upcoming_fixtures=None,
    team_context=None,
    gameweek=None,
    future_gameweek_only=False,
    competition_code="PL",
    api_token=None,
    historical_seasons=None,
    season=None,
    use_gemini_context=True,
    gemini_api_key=None,
    gemini_model=None,
    gemini_context_output_path=None,
):
    load_dotenv()

    loader = EPLScraper(
        api_token=api_token or os.getenv("FOOTBALL_DATA_API_TOKEN"),
        competition_code=competition_code,
    )
    if data_source == "football-data-api":
        parsed_seasons = parse_seasons(historical_seasons)
        historical_matches_df = loader.load_historical_matches_from_api(
            seasons=parsed_seasons
        )
        upcoming_fixtures_df = loader.load_upcoming_fixtures_from_api(
            gameweek=gameweek,
            season=season,
            future_gameweek_only=future_gameweek_only,
        )
        team_context_df = loader.load_team_context(team_context)
        data_source_label = f"football-data.org ({competition_code})"
    else:
        historical_path = historical_matches or "sample_data/historical_matches.csv"
        fixtures_path = upcoming_fixtures or "sample_data/upcoming_fixtures.csv"
        team_context_path = team_context or "sample_data/team_context.csv"
        historical_matches_df = loader.load_historical_matches(historical_path)
        upcoming_fixtures_df = loader.load_upcoming_fixtures(
            fixtures_path, gameweek=gameweek
        )
        team_context_df = loader.load_team_context(team_context_path)
        data_source_label = "bundled sample CSVs"
    gemini_context_df = pd.DataFrame()
    gemini_model_used = None
    gemini_context_artifact_path = None

    if use_gemini_context:
        enricher = GeminiTeamContextEnricher(
            api_key=gemini_api_key,
            model=gemini_model,
        )
        gemini_model_used = enricher.model
        raw_gemini_context_df = enricher.build_team_context(
            upcoming_fixtures_df, competition_code=competition_code
        )
        gemini_context_df = calibrate_team_context(
            team_context_df,
            raw_gemini_context_df,
        )
        gemini_context_artifact_path = persist_gemini_context(
            gemini_context_df,
            output_path=gemini_context_output_path,
        )
        team_context_df = merge_team_context_frames(team_context_df, gemini_context_df)

    selected_gameweeks = (
        sorted(upcoming_fixtures_df["gameweek"].dropna().astype(int).unique().tolist())
        if "gameweek" in upcoming_fixtures_df.columns
        else []
    )

    processor = DataProcessor(historical_matches_df, team_context_df)
    training_data = processor.build_training_data()
    fixture_features = processor.build_fixture_features(upcoming_fixtures_df)

    predictor = MatchPredictor(feature_columns=processor.FEATURE_COLUMNS)
    metrics = predictor.train(training_data)
    predictions = predictor.predict_fixtures(fixture_features)

    output_payload = build_output_payload(
        request_payload={
            "data_source": data_source,
            "competition_code": competition_code,
            "gameweek": gameweek,
            "future_gameweek_only": future_gameweek_only,
            "historical_seasons": parse_seasons(historical_seasons),
            "season": season,
            "use_gemini_context": use_gemini_context,
            "gemini_model": gemini_model_used,
            "gemini_context_output_path": gemini_context_artifact_path,
        },
        data_source_label=data_source_label,
        historical_matches=historical_matches_df,
        upcoming_fixtures=upcoming_fixtures_df,
        team_context=team_context_df,
        gemini_context=gemini_context_df,
        gemini_context_artifact_path=gemini_context_artifact_path,
        selected_gameweeks=selected_gameweeks,
        metrics=metrics,
        predictions=predictions,
    )
    return output_payload, predictions


def build_output_payload(
    request_payload,
    data_source_label,
    historical_matches,
    upcoming_fixtures,
    team_context,
    gemini_context,
    gemini_context_artifact_path,
    selected_gameweeks,
    metrics,
    predictions,
):
    return {
        "request": request_payload,
        "summary": {
            "data_source_label": data_source_label,
            "historical_match_count": int(len(historical_matches)),
            "upcoming_fixture_count": int(len(upcoming_fixtures)),
            "team_context_count": int(len(team_context)),
            "gemini_context_count": int(len(gemini_context)),
            "gemini_context_artifact_path": gemini_context_artifact_path,
            "selected_gameweeks": selected_gameweeks,
        },
        "gemini_context_rows": serialise_context_rows(gemini_context),
        "validation_metrics": normalise_metrics(metrics),
        "predictions": serialise_predictions(predictions),
    }


def normalise_metrics(metrics):
    if not metrics:
        return {}
    return {
        key: float(value) if isinstance(value, float) else int(value)
        for key, value in metrics.items()
    }


def serialise_predictions(predictions):
    serialised = []
    for row in predictions.to_dict(orient="records"):
        serialised.append(
            {
                "fixture_date": row["fixture_date"].strftime("%Y-%m-%d"),
                "gameweek": int(row["gameweek"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_probability": float(row["home_win_probability"]),
                "draw_probability": float(row["draw_probability"]),
                "away_win_probability": float(row["away_win_probability"]),
                "predicted_home_goals": int(row["predicted_home_goals"]),
                "predicted_away_goals": int(row["predicted_away_goals"]),
                "predicted_outcome": row["predicted_outcome"],
                "model_confidence": float(row["model_confidence"]),
                "predicted_scoreline": row["predicted_scoreline"],
                "scoreline_probability": float(row["scoreline_probability"]),
            }
        )
    return serialised


def serialise_context_rows(context_df):
    if context_df is None or context_df.empty:
        return []

    serialised = []
    for row in context_df.to_dict(orient="records"):
        serialised.append(
            {
                "team": row.get("team"),
                "effective_date": _serialise_optional_date(row.get("effective_date")),
                "gameweek": _serialise_optional_int(row.get("gameweek")),
                "squad_strength": _serialise_optional_float(row.get("squad_strength")),
                "availability_score": _serialise_optional_float(
                    row.get("availability_score")
                ),
                "expected_lineup_strength": _serialise_optional_float(
                    row.get("expected_lineup_strength")
                ),
                "injury_count": _serialise_optional_float(row.get("injury_count")),
                "suspended_count": _serialise_optional_float(
                    row.get("suspended_count")
                ),
                "confidence": _serialise_optional_float(row.get("confidence")),
                "notes": row.get("notes"),
                "source_summary": row.get("source_summary"),
            }
        )
    return serialised


def persist_gemini_context(context_df, output_path=None):
    if context_df is None or context_df.empty:
        return None

    if output_path:
        artifact_path = Path(output_path)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        artifact_path = (
            Path("artifacts")
            / "gemini_context"
            / f"gemini_context_{timestamp}.csv"
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = context_df.copy()
    if "effective_date" in export_df.columns:
        export_df["effective_date"] = pd.to_datetime(
            export_df["effective_date"]
        ).dt.strftime("%Y-%m-%d")
    export_df.to_csv(artifact_path, index=False)
    return str(artifact_path.resolve())


def _serialise_optional_date(value):
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _serialise_optional_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _serialise_optional_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def render_text_output(output_payload, predictions):
    summary = output_payload["summary"]
    print(f"Data source: {summary['data_source_label']}")
    print(f"Loaded {summary['historical_match_count']} historical matches")
    print(f"Loaded {summary['upcoming_fixture_count']} upcoming fixtures")
    print(f"Loaded {summary['team_context_count']} team context rows")
    if summary.get("gemini_context_count"):
        print(f"Generated {summary['gemini_context_count']} Gemini context rows")
    if summary.get("gemini_context_artifact_path"):
        print(f"Saved Gemini context: {summary['gemini_context_artifact_path']}")
    if summary["selected_gameweeks"]:
        selected = ", ".join(str(value) for value in summary["selected_gameweeks"])
        print(f"Selected gameweek(s): {selected}")

    metrics = output_payload["validation_metrics"]
    if metrics:
        print(
            "\nValidation metrics "
            f"(holdout on {metrics['training_rows']} engineered rows):"
        )
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Log loss: {metrics['log_loss']:.3f}")
        print(f"Home goals MAE: {metrics['home_goals_mae']:.3f}")
        print(f"Away goals MAE: {metrics['away_goals_mae']:.3f}")

    print("\nGameweek predictions:")
    print(
        predictions.to_string(
            index=False,
            formatters={
                "fixture_date": lambda value: value.strftime("%Y-%m-%d"),
                "home_win_probability": lambda value: f"{value:.1%}",
                "draw_probability": lambda value: f"{value:.1%}",
                "away_win_probability": lambda value: f"{value:.1%}",
                "model_confidence": lambda value: f"{value:.1%}",
                "scoreline_probability": lambda value: f"{value:.1%}",
            },
        )
    )


def render_json_output(output_payload):
    print(json.dumps(output_payload, indent=2))
