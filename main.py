import argparse
import json
import os
import sys

from dotenv import load_dotenv

from data_processor import DataProcessor
from prediction_model import MatchPredictor
from scraper import EPLScraper


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train on EPL match history and predict every fixture in a gameweek."
    )
    parser.add_argument(
        "--data-source",
        choices=["sample", "football-data-api"],
        default="sample",
        help="Use bundled sample CSVs or fetch real Premier League data from football-data.org.",
    )
    parser.add_argument(
        "--historical-matches",
        default=None,
        help="CSV of historical match results and pre-match team signals.",
    )
    parser.add_argument(
        "--upcoming-fixtures",
        default=None,
        help="CSV of upcoming fixtures to score.",
    )
    parser.add_argument(
        "--team-context",
        default=None,
        help="Optional CSV of current injuries, availability, squad strength, and lineup ratings.",
    )
    parser.add_argument(
        "--gameweek",
        type=int,
        default=None,
        help="Only score fixtures from a single gameweek.",
    )
    parser.add_argument(
        "--future-gameweek-only",
        action="store_true",
        help="When using football-data.org without --gameweek, skip the current partially remaining matchweek and select the next future gameweek instead.",
    )
    parser.add_argument(
        "--competition-code",
        default="PL",
        help="football-data.org competition code. Defaults to Premier League (`PL`).",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="football-data.org API token. Falls back to FOOTBALL_DATA_API_TOKEN.",
    )
    parser.add_argument(
        "--historical-seasons",
        default=None,
        help="Comma-separated season start years for API training data. On the football-data.org free tier, current season only is the safest option.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season start year for upcoming API fixtures. Defaults to the current football-data.org season.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Render terminal-friendly text or structured JSON.",
    )
    return parser


def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        loader = EPLScraper(
            api_token=args.api_token or os.getenv("FOOTBALL_DATA_API_TOKEN"),
            competition_code=args.competition_code,
        )
        if args.data_source == "football-data-api":
            historical_seasons = parse_seasons(args.historical_seasons)
            historical_matches = loader.load_historical_matches_from_api(
                seasons=historical_seasons
            )
            upcoming_fixtures = loader.load_upcoming_fixtures_from_api(
                gameweek=args.gameweek,
                season=args.season,
                future_gameweek_only=args.future_gameweek_only,
            )
            team_context = loader.load_team_context(args.team_context)
            data_source_label = f"football-data.org ({args.competition_code})"
        else:
            historical_path = args.historical_matches or "sample_data/historical_matches.csv"
            fixtures_path = args.upcoming_fixtures or "sample_data/upcoming_fixtures.csv"
            team_context_path = args.team_context or "sample_data/team_context.csv"
            historical_matches = loader.load_historical_matches(historical_path)
            upcoming_fixtures = loader.load_upcoming_fixtures(
                fixtures_path, gameweek=args.gameweek
            )
            team_context = loader.load_team_context(team_context_path)
            data_source_label = "bundled sample CSVs"

        selected_gameweeks = (
            sorted(upcoming_fixtures["gameweek"].dropna().astype(int).unique().tolist())
            if "gameweek" in upcoming_fixtures.columns
            else []
        )

        processor = DataProcessor(historical_matches, team_context)
        training_data = processor.build_training_data()
        fixture_features = processor.build_fixture_features(upcoming_fixtures)

        predictor = MatchPredictor(feature_columns=processor.FEATURE_COLUMNS)
        metrics = predictor.train(training_data)
        predictions = predictor.predict_fixtures(fixture_features)

        output_payload = build_output_payload(
            args=args,
            data_source_label=data_source_label,
            historical_matches=historical_matches,
            upcoming_fixtures=upcoming_fixtures,
            team_context=team_context,
            selected_gameweeks=selected_gameweeks,
            metrics=metrics,
            predictions=predictions,
        )

        if args.output == "json":
            print(json.dumps(output_payload, indent=2))
        else:
            render_text_output(output_payload, predictions)

    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        raise


def parse_seasons(raw_value):
    if not raw_value:
        return None
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_output_payload(
    args,
    data_source_label,
    historical_matches,
    upcoming_fixtures,
    team_context,
    selected_gameweeks,
    metrics,
    predictions,
):
    return {
        "request": {
            "data_source": args.data_source,
            "competition_code": args.competition_code,
            "gameweek": args.gameweek,
            "future_gameweek_only": args.future_gameweek_only,
            "historical_seasons": parse_seasons(args.historical_seasons),
            "season": args.season,
        },
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


if __name__ == "__main__":
    main()
