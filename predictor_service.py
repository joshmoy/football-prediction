import json
import os

from dotenv import load_dotenv

from data_processor import DataProcessor
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
        },
        data_source_label=data_source_label,
        historical_matches=historical_matches_df,
        upcoming_fixtures=upcoming_fixtures_df,
        team_context=team_context_df,
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
            "selected_gameweeks": selected_gameweeks,
        },
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
                "expected_home_goals": float(row["expected_home_goals"]),
                "expected_away_goals": float(row["expected_away_goals"]),
                "predicted_outcome": row["predicted_outcome"],
                "model_confidence": float(row["model_confidence"]),
                "predicted_scoreline": row["predicted_scoreline"],
                "scoreline_probability": float(row["scoreline_probability"]),
            }
        )
    return serialised


def render_text_output(output_payload, predictions):
    summary = output_payload["summary"]
    print(f"Data source: {summary['data_source_label']}")
    print(f"Loaded {summary['historical_match_count']} historical matches")
    print(f"Loaded {summary['upcoming_fixture_count']} upcoming fixtures")
    print(f"Loaded {summary['team_context_count']} team context rows")
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
                "expected_home_goals": lambda value: f"{value:.2f}",
                "expected_away_goals": lambda value: f"{value:.2f}",
                "scoreline_probability": lambda value: f"{value:.1%}",
            },
        )
    )


def render_json_output(output_payload):
    print(json.dumps(output_payload, indent=2))
