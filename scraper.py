import os
from pathlib import Path

import pandas as pd
import requests
from requests import HTTPError


class EPLScraper:
    """Load EPL training and fixture data from CSV files or football-data.org."""

    API_BASE_URL = "https://api.football-data.org/v4"
    DEFAULT_COMPETITION_CODE = "PL"
    REQUIRED_HISTORICAL_COLUMNS = {
        "season",
        "match_date",
        "gameweek",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }
    REQUIRED_FIXTURE_COLUMNS = {"fixture_date", "gameweek", "home_team", "away_team"}
    TEAM_CONTEXT_NUMERIC_COLUMNS = [
        "squad_strength",
        "availability_score",
        "expected_lineup_strength",
        "injury_count",
        "suspended_count",
    ]
    UPCOMING_STATUSES = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "POSTPONED"}

    def __init__(self, api_token=None, competition_code=DEFAULT_COMPETITION_CODE):
        self.api_token = api_token or os.getenv("FOOTBALL_DATA_API_TOKEN")
        self.competition_code = competition_code
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "goborr-ai/1.0"})
        if self.api_token:
            self.session.headers["X-Auth-Token"] = self.api_token

    def load_historical_matches(self, csv_path):
        historical_matches = self._read_csv(csv_path, parse_dates=["match_date"])
        return self._prepare_historical_matches(historical_matches)

    def load_historical_matches_from_api(self, seasons=None):
        competition = self.get_competition()
        if seasons is None:
            current_season = self._season_start_year(competition["currentSeason"])
            seasons = [current_season]

        historical_rows = []
        for season in seasons:
            try:
                payload = self._request_json(
                    f"/competitions/{self.competition_code}/matches",
                    params={"season": season, "status": "FINISHED"},
                )
            except HTTPError as exc:
                self._raise_api_access_error(exc, season=season)
            for match in payload.get("matches", []):
                row = self._normalise_api_match(match, season)
                if row is not None:
                    historical_rows.append(row)

        historical_matches = pd.DataFrame(historical_rows)
        if historical_matches.empty:
            raise ValueError(
                "No historical matches were returned by football-data.org for the requested seasons."
            )
        return self._prepare_historical_matches(historical_matches)

    def load_upcoming_fixtures(self, csv_path, gameweek=None):
        fixtures = self._read_csv(csv_path, parse_dates=["fixture_date"])
        return self._prepare_upcoming_fixtures(fixtures, gameweek=gameweek)

    def load_upcoming_fixtures_from_api(
        self, gameweek=None, season=None, future_gameweek_only=False
    ):
        competition = self.get_competition()
        if season is None:
            season = self._season_start_year(competition["currentSeason"])

        payload = self._request_json(
            f"/competitions/{self.competition_code}/matches", params={"season": season}
        )
        all_fixture_rows = [
            self._normalise_api_fixture(match) for match in payload.get("matches", [])
        ]

        if gameweek is not None:
            fixtures = pd.DataFrame(all_fixture_rows)
            if fixtures.empty:
                raise ValueError(
                    "No fixtures were returned by football-data.org for the requested season."
                )
            return self._prepare_upcoming_fixtures(fixtures, gameweek=gameweek)

        fixture_rows = [
            row
            for row in all_fixture_rows
            if row["status"] in self.UPCOMING_STATUSES
        ]
        fixtures = pd.DataFrame(fixture_rows)
        if fixtures.empty:
            raise ValueError("No upcoming fixtures were returned by football-data.org.")

        if gameweek is None:
            gameweek = self._infer_next_gameweek(
                fixtures, competition, future_gameweek_only=future_gameweek_only
            )
        return self._prepare_upcoming_fixtures(fixtures, gameweek=gameweek)

    def load_team_context(self, csv_path=None):
        if not csv_path:
            return pd.DataFrame(columns=["team", "effective_date", "gameweek"])

        team_context = self._read_csv(csv_path)
        if "team" not in team_context.columns:
            raise ValueError("Team context data must include a 'team' column.")

        if "effective_date" not in team_context.columns:
            team_context["effective_date"] = pd.NaT
        else:
            team_context["effective_date"] = pd.to_datetime(
                team_context["effective_date"]
            ).dt.tz_localize(None)

        if "gameweek" not in team_context.columns:
            team_context["gameweek"] = pd.NA

        return self._coerce_numeric_columns(
            team_context, ["gameweek", *self.TEAM_CONTEXT_NUMERIC_COLUMNS]
        )

    def get_competition(self):
        return self._request_json(f"/competitions/{self.competition_code}")

    def _prepare_historical_matches(self, historical_matches):
        self._validate_columns(
            historical_matches,
            self.REQUIRED_HISTORICAL_COLUMNS,
            dataset_name="historical matches",
        )
        historical_matches = historical_matches.sort_values(
            ["match_date", "season", "gameweek", "home_team", "away_team"]
        ).reset_index(drop=True)
        historical_matches["match_date"] = pd.to_datetime(
            historical_matches["match_date"]
        ).dt.tz_localize(None)

        numeric_columns = [
            "gameweek",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_lineup_strength",
            "away_lineup_strength",
            "home_availability_score",
            "away_availability_score",
            "home_squad_strength",
            "away_squad_strength",
            "home_injury_count",
            "away_injury_count",
            "home_suspended_count",
            "away_suspended_count",
        ]
        return self._coerce_numeric_columns(historical_matches, numeric_columns)

    def _prepare_upcoming_fixtures(self, fixtures, gameweek=None):
        self._validate_columns(
            fixtures,
            self.REQUIRED_FIXTURE_COLUMNS,
            dataset_name="upcoming fixtures",
        )
        fixtures["fixture_date"] = pd.to_datetime(fixtures["fixture_date"]).dt.tz_localize(
            None
        )
        fixtures = self._coerce_numeric_columns(fixtures, ["gameweek"])

        if gameweek is not None:
            fixtures = fixtures[fixtures["gameweek"] == int(gameweek)]

        if fixtures.empty:
            raise ValueError("No fixtures found for the requested gameweek.")

        return fixtures.sort_values(
            ["fixture_date", "gameweek", "home_team", "away_team"]
        ).reset_index(drop=True)

    def _request_json(self, path, params=None):
        if not self.api_token:
            raise ValueError(
                "football-data.org API token missing. Set FOOTBALL_DATA_API_TOKEN or pass --api-token."
            )
        response = self.session.get(f"{self.API_BASE_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _raise_api_access_error(self, exc, season=None):
        response = exc.response
        if response is not None and response.status_code == 403:
            season_text = f" for season {season}" if season is not None else ""
            raise ValueError(
                "football-data.org denied access"
                f"{season_text}. On the free tier, older historical seasons may be unavailable. "
                "Try the current season only, for example omit --historical-seasons or use "
                f"--historical-seasons {self._current_season_hint()}."
            ) from exc
        raise exc

    def _current_season_hint(self):
        try:
            competition = self.get_competition()
            return str(self._season_start_year(competition.get("currentSeason")))
        except Exception:
            return "CURRENT_SEASON_START_YEAR"

    def _normalise_api_match(self, match, requested_season):
        full_time = (match.get("score") or {}).get("fullTime") or {}
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if home_goals is None or away_goals is None:
            return None

        season = match.get("season") or {}
        season_start_year = self._season_start_year(season, fallback=requested_season)
        return {
            "season": f"{season_start_year}/{str(season_start_year + 1)[-2:]}",
            "match_date": match.get("utcDate"),
            "gameweek": match.get("matchday"),
            "home_team": self._team_name(match, "homeTeam"),
            "away_team": self._team_name(match, "awayTeam"),
            "home_goals": home_goals,
            "away_goals": away_goals,
        }

    def _normalise_api_fixture(self, match):
        return {
            "fixture_date": match.get("utcDate"),
            "gameweek": match.get("matchday"),
            "home_team": self._team_name(match, "homeTeam"),
            "away_team": self._team_name(match, "awayTeam"),
            "status": match.get("status"),
        }

    def _infer_next_gameweek(self, fixtures, competition, future_gameweek_only=False):
        remaining = fixtures.sort_values(["gameweek", "fixture_date"])
        matchday_values = remaining["gameweek"].dropna().astype(int)
        if not matchday_values.empty:
            unique_matchdays = sorted(matchday_values.unique())
            if future_gameweek_only and len(unique_matchdays) > 1:
                return int(unique_matchdays[1])
            return int(unique_matchdays[0])

        current_matchday = (
            competition.get("currentSeason", {}).get("currentMatchday")
            if competition
            else None
        )
        if current_matchday is None:
            raise ValueError("Could not infer the next gameweek from football-data.org.")
        return int(current_matchday)

    def _season_start_year(self, season, fallback=None):
        start_date = (season or {}).get("startDate")
        if start_date:
            return int(str(start_date)[:4])
        if fallback is None:
            raise ValueError("Could not infer the season start year from football-data.org.")
        return int(fallback)

    def _team_name(self, match, key):
        return ((match.get(key) or {}).get("shortName") or (match.get(key) or {}).get("name") or "").strip()

    def _read_csv(self, csv_path, parse_dates=None):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find dataset: {path}")
        return pd.read_csv(path, parse_dates=parse_dates or [])

    def _validate_columns(self, dataframe, required_columns, dataset_name):
        missing_columns = sorted(required_columns - set(dataframe.columns))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing columns in {dataset_name}: {missing}")

    def _coerce_numeric_columns(self, dataframe, columns):
        for column in columns:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return dataframe
