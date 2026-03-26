from collections import defaultdict

import pandas as pd


class DataProcessor:
    FEATURE_COLUMNS = [
        "home_matches_played",
        "away_matches_played",
        "home_points_per_match",
        "away_points_per_match",
        "home_goal_diff_per_match",
        "away_goal_diff_per_match",
        "home_form_points_last5",
        "away_form_points_last5",
        "home_goals_for_last5",
        "away_goals_for_last5",
        "home_goals_against_last5",
        "away_goals_against_last5",
        "home_xg_for_last5",
        "away_xg_for_last5",
        "home_xg_against_last5",
        "away_xg_against_last5",
        "home_shots_for_last5",
        "away_shots_for_last5",
        "home_shots_against_last5",
        "away_shots_against_last5",
        "home_venue_points_per_match",
        "away_venue_points_per_match",
        "home_days_rest",
        "away_days_rest",
        "home_squad_strength",
        "away_squad_strength",
        "home_availability_score",
        "away_availability_score",
        "home_expected_lineup_strength",
        "away_expected_lineup_strength",
        "home_injury_count",
        "away_injury_count",
        "home_suspended_count",
        "away_suspended_count",
    ]
    TEAM_CONTEXT_COLUMNS = [
        "squad_strength",
        "availability_score",
        "expected_lineup_strength",
        "injury_count",
        "suspended_count",
    ]
    OUTCOME_LABELS = {0: "AWAY_WIN", 1: "DRAW", 2: "HOME_WIN"}

    def __init__(self, historical_matches_df, team_context_df=None, form_window=5):
        self.historical_matches_df = historical_matches_df.copy().sort_values(
            ["match_date", "season", "gameweek", "home_team", "away_team"]
        ).reset_index(drop=True)
        self.team_context_df = (
            team_context_df.copy()
            if team_context_df is not None
            else pd.DataFrame(columns=["team", "effective_date", "gameweek"])
        )
        self.form_window = form_window

    def build_training_data(self):
        histories = defaultdict(list)
        training_rows = []

        for _, match in self.historical_matches_df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["match_date"]

            home_snapshot = self._build_team_snapshot(
                histories[home_team], preferred_venue="home", reference_date=match_date
            )
            away_snapshot = self._build_team_snapshot(
                histories[away_team], preferred_venue="away", reference_date=match_date
            )

            row = self._compose_feature_row(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                gameweek=match["gameweek"],
                home_snapshot=home_snapshot,
                away_snapshot=away_snapshot,
                home_context=self._extract_context_from_match(match, prefix="home"),
                away_context=self._extract_context_from_match(match, prefix="away"),
            )
            row["outcome"] = self._result_to_label(
                home_goals=match["home_goals"], away_goals=match["away_goals"]
            )
            row["home_goals"] = float(match["home_goals"])
            row["away_goals"] = float(match["away_goals"])
            training_rows.append(row)

            self._append_match_to_history(histories, match)

        training_df = pd.DataFrame(training_rows)
        if training_df.empty:
            raise ValueError("Training data is empty after feature engineering.")
        return training_df

    def build_fixture_features(self, fixtures_df):
        histories = self._seed_histories_from_training_data()
        fixture_rows = []

        for _, fixture in fixtures_df.iterrows():
            fixture_date = fixture["fixture_date"]
            home_team = fixture["home_team"]
            away_team = fixture["away_team"]

            home_snapshot = self._build_team_snapshot(
                histories[home_team], preferred_venue="home", reference_date=fixture_date
            )
            away_snapshot = self._build_team_snapshot(
                histories[away_team], preferred_venue="away", reference_date=fixture_date
            )

            home_context = self._lookup_team_context(
                team=home_team,
                reference_date=fixture_date,
                gameweek=fixture.get("gameweek"),
            )
            away_context = self._lookup_team_context(
                team=away_team,
                reference_date=fixture_date,
                gameweek=fixture.get("gameweek"),
            )

            row = self._compose_feature_row(
                home_team=home_team,
                away_team=away_team,
                match_date=fixture_date,
                gameweek=fixture["gameweek"],
                home_snapshot=home_snapshot,
                away_snapshot=away_snapshot,
                home_context=home_context,
                away_context=away_context,
            )
            fixture_rows.append(row)

        if not fixture_rows:
            raise ValueError("No fixture rows were generated for prediction.")

        engineered_fixtures = pd.DataFrame(fixture_rows)[self.FEATURE_COLUMNS]
        fixture_metadata = fixtures_df.reset_index(drop=True).copy()
        return pd.concat([fixture_metadata, engineered_fixtures], axis=1)

    def _seed_histories_from_training_data(self):
        histories = defaultdict(list)
        for _, match in self.historical_matches_df.iterrows():
            self._append_match_to_history(histories, match)
        return histories

    def _append_match_to_history(self, histories, match):
        home_goals = float(match["home_goals"])
        away_goals = float(match["away_goals"])
        match_date = match["match_date"]

        histories[match["home_team"]].append(
            {
                "match_date": match_date,
                "venue": "home",
                "goals_for": home_goals,
                "goals_against": away_goals,
                "points": self._points_for_score(home_goals, away_goals),
                "xg_for": self._value_or_default(match, "home_xg", home_goals),
                "xg_against": self._value_or_default(match, "away_xg", away_goals),
                "shots_for": self._value_or_default(match, "home_shots", 0.0),
                "shots_against": self._value_or_default(match, "away_shots", 0.0),
            }
        )
        histories[match["away_team"]].append(
            {
                "match_date": match_date,
                "venue": "away",
                "goals_for": away_goals,
                "goals_against": home_goals,
                "points": self._points_for_score(away_goals, home_goals),
                "xg_for": self._value_or_default(match, "away_xg", away_goals),
                "xg_against": self._value_or_default(match, "home_xg", home_goals),
                "shots_for": self._value_or_default(match, "away_shots", 0.0),
                "shots_against": self._value_or_default(match, "home_shots", 0.0),
            }
        )

    def _build_team_snapshot(self, team_history, preferred_venue, reference_date):
        if not team_history:
            return {
                "matches_played": 0.0,
                "points_per_match": 0.0,
                "goal_diff_per_match": 0.0,
                "form_points_last5": 0.0,
                "goals_for_last5": 0.0,
                "goals_against_last5": 0.0,
                "xg_for_last5": 0.0,
                "xg_against_last5": 0.0,
                "shots_for_last5": 0.0,
                "shots_against_last5": 0.0,
                "venue_points_per_match": 0.0,
                "days_rest": 0.0,
            }

        recent_matches = team_history[-self.form_window :]
        venue_matches = [
            row for row in team_history if row["venue"] == preferred_venue
        ][-self.form_window :]
        total_matches = len(team_history)

        last_match_date = team_history[-1]["match_date"]
        days_rest = max((reference_date - last_match_date).days, 0)

        return {
            "matches_played": float(total_matches),
            "points_per_match": self._mean([row["points"] for row in team_history]),
            "goal_diff_per_match": self._mean(
                [row["goals_for"] - row["goals_against"] for row in team_history]
            ),
            "form_points_last5": float(sum(row["points"] for row in recent_matches)),
            "goals_for_last5": self._mean([row["goals_for"] for row in recent_matches]),
            "goals_against_last5": self._mean(
                [row["goals_against"] for row in recent_matches]
            ),
            "xg_for_last5": self._mean([row["xg_for"] for row in recent_matches]),
            "xg_against_last5": self._mean(
                [row["xg_against"] for row in recent_matches]
            ),
            "shots_for_last5": self._mean([row["shots_for"] for row in recent_matches]),
            "shots_against_last5": self._mean(
                [row["shots_against"] for row in recent_matches]
            ),
            "venue_points_per_match": self._mean(
                [row["points"] for row in venue_matches]
            ),
            "days_rest": float(days_rest),
        }

    def _compose_feature_row(
        self,
        home_team,
        away_team,
        match_date,
        gameweek,
        home_snapshot,
        away_snapshot,
        home_context,
        away_context,
    ):
        return {
            "match_date": match_date,
            "gameweek": int(gameweek) if pd.notna(gameweek) else pd.NA,
            "home_team": home_team,
            "away_team": away_team,
            "home_matches_played": home_snapshot["matches_played"],
            "away_matches_played": away_snapshot["matches_played"],
            "home_points_per_match": home_snapshot["points_per_match"],
            "away_points_per_match": away_snapshot["points_per_match"],
            "home_goal_diff_per_match": home_snapshot["goal_diff_per_match"],
            "away_goal_diff_per_match": away_snapshot["goal_diff_per_match"],
            "home_form_points_last5": home_snapshot["form_points_last5"],
            "away_form_points_last5": away_snapshot["form_points_last5"],
            "home_goals_for_last5": home_snapshot["goals_for_last5"],
            "away_goals_for_last5": away_snapshot["goals_for_last5"],
            "home_goals_against_last5": home_snapshot["goals_against_last5"],
            "away_goals_against_last5": away_snapshot["goals_against_last5"],
            "home_xg_for_last5": home_snapshot["xg_for_last5"],
            "away_xg_for_last5": away_snapshot["xg_for_last5"],
            "home_xg_against_last5": home_snapshot["xg_against_last5"],
            "away_xg_against_last5": away_snapshot["xg_against_last5"],
            "home_shots_for_last5": home_snapshot["shots_for_last5"],
            "away_shots_for_last5": away_snapshot["shots_for_last5"],
            "home_shots_against_last5": home_snapshot["shots_against_last5"],
            "away_shots_against_last5": away_snapshot["shots_against_last5"],
            "home_venue_points_per_match": home_snapshot["venue_points_per_match"],
            "away_venue_points_per_match": away_snapshot["venue_points_per_match"],
            "home_days_rest": home_snapshot["days_rest"],
            "away_days_rest": away_snapshot["days_rest"],
            "home_squad_strength": home_context["squad_strength"],
            "away_squad_strength": away_context["squad_strength"],
            "home_availability_score": home_context["availability_score"],
            "away_availability_score": away_context["availability_score"],
            "home_expected_lineup_strength": home_context[
                "expected_lineup_strength"
            ],
            "away_expected_lineup_strength": away_context[
                "expected_lineup_strength"
            ],
            "home_injury_count": home_context["injury_count"],
            "away_injury_count": away_context["injury_count"],
            "home_suspended_count": home_context["suspended_count"],
            "away_suspended_count": away_context["suspended_count"],
        }

    def _extract_context_from_match(self, match, prefix):
        return {
            "squad_strength": self._value_or_default(
                match, f"{prefix}_squad_strength", 0.0
            ),
            "availability_score": self._value_or_default(
                match, f"{prefix}_availability_score", 0.0
            ),
            "expected_lineup_strength": self._value_or_default(
                match, f"{prefix}_lineup_strength", 0.0
            ),
            "injury_count": self._value_or_default(
                match, f"{prefix}_injury_count", 0.0
            ),
            "suspended_count": self._value_or_default(
                match, f"{prefix}_suspended_count", 0.0
            ),
        }

    def _lookup_team_context(self, team, reference_date, gameweek):
        if self.team_context_df.empty:
            return {column: 0.0 for column in self.TEAM_CONTEXT_COLUMNS}

        team_rows = self.team_context_df[self.team_context_df["team"] == team].copy()
        if team_rows.empty:
            return {column: 0.0 for column in self.TEAM_CONTEXT_COLUMNS}

        eligible_rows = team_rows
        if "gameweek" in team_rows.columns and pd.notna(gameweek):
            gameweek_rows = team_rows[team_rows["gameweek"] == int(gameweek)]
            if not gameweek_rows.empty:
                eligible_rows = gameweek_rows

        if (
            "effective_date" in eligible_rows.columns
            and eligible_rows["effective_date"].notna().any()
        ):
            dated_rows = eligible_rows[eligible_rows["effective_date"].notna()]
            dated_rows = dated_rows[dated_rows["effective_date"] <= reference_date]
            if not dated_rows.empty:
                eligible_rows = dated_rows

        selected_row = eligible_rows.sort_values(
            ["effective_date", "gameweek"], na_position="last"
        ).iloc[-1]
        return {
            column: self._safe_float(selected_row.get(column, 0.0))
            for column in self.TEAM_CONTEXT_COLUMNS
        }

    def _points_for_score(self, goals_for, goals_against):
        if goals_for > goals_against:
            return 3.0
        if goals_for == goals_against:
            return 1.0
        return 0.0

    def _result_to_label(self, home_goals, away_goals):
        if home_goals > away_goals:
            return 2
        if home_goals == away_goals:
            return 1
        return 0

    def _mean(self, values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _value_or_default(self, row, column, default):
        if column not in row.index:
            return float(default)
        value = row[column]
        if pd.isna(value):
            return float(default)
        return float(value)

    def _safe_float(self, value):
        if pd.isna(value):
            return 0.0
        return float(value)
