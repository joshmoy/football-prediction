import argparse
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

        print(f"Data source: {data_source_label}")
        print(f"Loaded {len(historical_matches)} historical matches")
        print(f"Loaded {len(upcoming_fixtures)} upcoming fixtures")
        print(f"Loaded {len(team_context)} team context rows")
        selected_gameweeks = (
            sorted(upcoming_fixtures["gameweek"].dropna().astype(int).unique().tolist())
            if "gameweek" in upcoming_fixtures.columns
            else []
        )
        if selected_gameweeks:
            print(f"Selected gameweek(s): {', '.join(str(value) for value in selected_gameweeks)}")

        processor = DataProcessor(historical_matches, team_context)
        training_data = processor.build_training_data()
        fixture_features = processor.build_fixture_features(upcoming_fixtures)

        predictor = MatchPredictor(feature_columns=processor.FEATURE_COLUMNS)
        metrics = predictor.train(training_data)
        predictions = predictor.predict_fixtures(fixture_features)

        if metrics:
            print(
                "\nValidation metrics "
                f"(holdout on {metrics['training_rows']} engineered rows):"
            )
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Log loss: {metrics['log_loss']:.3f}")

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
                },
            )
        )

    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        raise


def parse_seasons(raw_value):
    if not raw_value:
        return None
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
