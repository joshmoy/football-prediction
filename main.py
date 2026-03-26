import argparse
import sys

from predictor_service import render_json_output, render_text_output, run_prediction


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
    parser = build_parser()
    args = parser.parse_args()

    try:
        output_payload, predictions = run_prediction(
            data_source=args.data_source,
            historical_matches=args.historical_matches,
            upcoming_fixtures=args.upcoming_fixtures,
            team_context=args.team_context,
            gameweek=args.gameweek,
            future_gameweek_only=args.future_gameweek_only,
            competition_code=args.competition_code,
            api_token=args.api_token,
            historical_seasons=args.historical_seasons,
            season=args.season,
        )

        if args.output == "json":
            render_json_output(output_payload)
        else:
            render_text_output(output_payload, predictions)

    except Exception as exc:
        print(f"An error occurred: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
