import math

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MatchPredictor:
    OUTCOME_MAP = {0: "away_win", 1: "draw", 2: "home_win"}
    MAX_GOALS = 6

    def __init__(self, feature_columns, random_state=42):
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.outcome_model = GradientBoostingClassifier(random_state=random_state)
        self.home_goals_model = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=0.2, max_iter=2000)
        )
        self.away_goals_model = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=0.2, max_iter=2000)
        )
        self.validation_metrics = {}

    def train(self, training_df):
        if training_df.empty:
            raise ValueError("Cannot train predictor with an empty training dataset.")

        X = training_df[self.feature_columns].fillna(0.0)
        y_outcome = training_df["outcome"]
        y_home_goals = training_df["home_goals"]
        y_away_goals = training_df["away_goals"]

        if y_outcome.nunique() < 2:
            raise ValueError("Training data must contain at least two outcome classes.")

        should_validate = len(training_df) >= 18 and y_outcome.value_counts().min() >= 2
        if should_validate:
            (
                X_train,
                X_valid,
                y_outcome_train,
                y_outcome_valid,
                y_home_train,
                y_home_valid,
                y_away_train,
                y_away_valid,
            ) = train_test_split(
                X,
                y_outcome,
                y_home_goals,
                y_away_goals,
                test_size=0.25,
                random_state=self.random_state,
                stratify=y_outcome,
            )
            self.outcome_model.fit(X_train, y_outcome_train)
            self.home_goals_model.fit(X_train, y_home_train)
            self.away_goals_model.fit(X_train, y_away_train)

            valid_probabilities = self.outcome_model.predict_proba(X_valid)
            valid_predictions = self.outcome_model.predict(X_valid)
            valid_home_goal_predictions = self._clip_goal_predictions(
                self.home_goals_model.predict(X_valid)
            )
            valid_away_goal_predictions = self._clip_goal_predictions(
                self.away_goals_model.predict(X_valid)
            )
            self.validation_metrics = {
                "accuracy": accuracy_score(y_outcome_valid, valid_predictions),
                "log_loss": log_loss(
                    y_outcome_valid,
                    valid_probabilities,
                    labels=self.outcome_model.classes_,
                ),
                "home_goals_mae": mean_absolute_error(
                    y_home_valid, valid_home_goal_predictions
                ),
                "away_goals_mae": mean_absolute_error(
                    y_away_valid, valid_away_goal_predictions
                ),
                "training_rows": len(training_df),
            }

        self.outcome_model.fit(X, y_outcome)
        self.home_goals_model.fit(X, y_home_goals)
        self.away_goals_model.fit(X, y_away_goals)
        return self.validation_metrics

    def predict_fixtures(self, fixtures_df):
        X = fixtures_df[self.feature_columns].fillna(0.0)
        probabilities = self.outcome_model.predict_proba(X)
        outcome_probabilities = self._normalise_probabilities(probabilities)
        expected_home_goals = self._clip_goal_predictions(self.home_goals_model.predict(X))
        expected_away_goals = self._clip_goal_predictions(self.away_goals_model.predict(X))

        results = fixtures_df[
            ["fixture_date", "gameweek", "home_team", "away_team"]
        ].copy()
        results["home_win_probability"] = outcome_probabilities["home_win"]
        results["draw_probability"] = outcome_probabilities["draw"]
        results["away_win_probability"] = outcome_probabilities["away_win"]
        results["expected_home_goals"] = expected_home_goals
        results["expected_away_goals"] = expected_away_goals
        results["predicted_outcome"] = results[
            [
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ]
        ].idxmax(axis=1)
        results["predicted_outcome"] = results["predicted_outcome"].map(
            {
                "home_win_probability": "HOME_WIN",
                "draw_probability": "DRAW",
                "away_win_probability": "AWAY_WIN",
            }
        )
        results["model_confidence"] = results[
            [
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ]
        ].max(axis=1)
        scorelines = [
            self._most_likely_scoreline(home_goals, away_goals)
            for home_goals, away_goals in zip(expected_home_goals, expected_away_goals)
        ]
        results["predicted_scoreline"] = [
            f"{home_goals}-{away_goals}" for home_goals, away_goals, _ in scorelines
        ]
        results["scoreline_probability"] = [
            probability for _, _, probability in scorelines
        ]
        return results.sort_values(["fixture_date", "home_team"]).reset_index(drop=True)

    def _normalise_probabilities(self, probabilities):
        probability_map = {name: [0.0] * len(probabilities) for name in self.OUTCOME_MAP.values()}
        for class_index, class_label in enumerate(self.outcome_model.classes_):
            outcome_name = self.OUTCOME_MAP[int(class_label)]
            probability_map[outcome_name] = probabilities[:, class_index]
        return probability_map

    def _clip_goal_predictions(self, values):
        return [max(float(value), 0.05) for value in values]

    def _most_likely_scoreline(self, expected_home_goals, expected_away_goals):
        best_home_goals = 0
        best_away_goals = 0
        best_probability = -1.0

        for home_goals in range(self.MAX_GOALS + 1):
            home_probability = self._poisson_pmf(home_goals, expected_home_goals)
            for away_goals in range(self.MAX_GOALS + 1):
                probability = home_probability * self._poisson_pmf(
                    away_goals, expected_away_goals
                )
                if probability > best_probability:
                    best_home_goals = home_goals
                    best_away_goals = away_goals
                    best_probability = probability

        return best_home_goals, best_away_goals, best_probability

    def _poisson_pmf(self, goals, expected_goals):
        return (
            math.exp(-expected_goals)
            * (expected_goals ** goals)
            / math.factorial(goals)
        )
