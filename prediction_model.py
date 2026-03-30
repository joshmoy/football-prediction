import math

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MatchPredictor:
    OUTCOME_INDEX = {"away_win": 0, "draw": 1, "home_win": 2}
    OUTCOME_LABELS = {0: "AWAY_WIN", 1: "DRAW", 2: "HOME_WIN"}
    MAX_GOALS = 10

    def __init__(self, feature_columns, random_state=42):
        self.feature_columns = feature_columns
        self.random_state = random_state
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
            self.home_goals_model.fit(X_train, y_home_train)
            self.away_goals_model.fit(X_train, y_away_train)

            valid_home_goal_predictions = self._clip_goal_predictions(
                self.home_goals_model.predict(X_valid)
            )
            valid_away_goal_predictions = self._clip_goal_predictions(
                self.away_goals_model.predict(X_valid)
            )
            valid_probability_rows = self._outcome_probability_rows(
                valid_home_goal_predictions,
                valid_away_goal_predictions,
            )
            valid_predictions = [
                self._predicted_outcome_index(probability_row)
                for probability_row in valid_probability_rows
            ]
            self.validation_metrics = {
                "accuracy": accuracy_score(y_outcome_valid, valid_predictions),
                "log_loss": log_loss(
                    y_outcome_valid,
                    valid_probability_rows,
                    labels=[0, 1, 2],
                ),
                "home_goals_mae": mean_absolute_error(
                    y_home_valid, valid_home_goal_predictions
                ),
                "away_goals_mae": mean_absolute_error(
                    y_away_valid, valid_away_goal_predictions
                ),
                "training_rows": len(training_df),
            }

        self.home_goals_model.fit(X, y_home_goals)
        self.away_goals_model.fit(X, y_away_goals)
        return self.validation_metrics

    def predict_fixtures(self, fixtures_df):
        X = fixtures_df[self.feature_columns].fillna(0.0)
        expected_home_goals = self._clip_goal_predictions(self.home_goals_model.predict(X))
        expected_away_goals = self._clip_goal_predictions(self.away_goals_model.predict(X))
        score_distributions = [
            self._joint_score_distribution(home_goals, away_goals)
            for home_goals, away_goals in zip(expected_home_goals, expected_away_goals)
        ]
        outcome_probabilities = [
            self._outcome_probabilities_from_distribution(distribution)
            for distribution in score_distributions
        ]

        results = fixtures_df[
            ["fixture_date", "gameweek", "home_team", "away_team"]
        ].copy()
        results["home_win_probability"] = [
            probabilities["home_win"] for probabilities in outcome_probabilities
        ]
        results["draw_probability"] = [
            probabilities["draw"] for probabilities in outcome_probabilities
        ]
        results["away_win_probability"] = [
            probabilities["away_win"] for probabilities in outcome_probabilities
        ]
        predicted_outcomes = [
            self._predicted_outcome_name(probabilities)
            for probabilities in outcome_probabilities
        ]
        results["predicted_outcome"] = [
            self.OUTCOME_LABELS[self.OUTCOME_INDEX[outcome_name]]
            for outcome_name in predicted_outcomes
        ]
        results["model_confidence"] = [
            max(probabilities.values()) for probabilities in outcome_probabilities
        ]
        scorelines = [
            self._most_likely_scoreline_for_outcome(distribution, outcome_name)
            for distribution, outcome_name in zip(score_distributions, predicted_outcomes)
        ]
        results["predicted_scoreline"] = [
            f"{home_goals}-{away_goals}" for home_goals, away_goals, _ in scorelines
        ]
        results["predicted_home_goals"] = [
            home_goals for home_goals, _, _ in scorelines
        ]
        results["predicted_away_goals"] = [
            away_goals for _, away_goals, _ in scorelines
        ]
        results["scoreline_probability"] = [
            probability for _, _, probability in scorelines
        ]
        return results.sort_values(["fixture_date", "home_team"]).reset_index(drop=True)

    def _clip_goal_predictions(self, values):
        return [max(float(value), 0.05) for value in values]

    def _outcome_probability_rows(self, expected_home_goals, expected_away_goals):
        probability_rows = []
        for home_goals, away_goals in zip(expected_home_goals, expected_away_goals):
            outcome_probabilities = self._outcome_probabilities_from_distribution(
                self._joint_score_distribution(home_goals, away_goals)
            )
            probability_rows.append(
                [
                    outcome_probabilities["away_win"],
                    outcome_probabilities["draw"],
                    outcome_probabilities["home_win"],
                ]
            )
        return probability_rows

    def _predicted_outcome_name(self, outcome_probabilities):
        return max(
            outcome_probabilities,
            key=lambda outcome_name: outcome_probabilities[outcome_name],
        )

    def _predicted_outcome_index(self, probability_row):
        best_index = max(range(len(probability_row)), key=lambda index: probability_row[index])
        return best_index

    def _joint_score_distribution(self, expected_home_goals, expected_away_goals):
        distribution = []
        total_probability = 0.0
        for home_goals in range(self.MAX_GOALS + 1):
            home_probability = self._poisson_pmf(home_goals, expected_home_goals)
            for away_goals in range(self.MAX_GOALS + 1):
                probability = home_probability * self._poisson_pmf(
                    away_goals, expected_away_goals
                )
                distribution.append((home_goals, away_goals, probability))
                total_probability += probability

        if total_probability <= 0.0:
            raise ValueError("Joint score distribution could not be normalised.")

        return [
            (home_goals, away_goals, probability / total_probability)
            for home_goals, away_goals, probability in distribution
        ]

    def _outcome_probabilities_from_distribution(self, distribution):
        probabilities = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for home_goals, away_goals, probability in distribution:
            if home_goals > away_goals:
                probabilities["home_win"] += probability
            elif home_goals == away_goals:
                probabilities["draw"] += probability
            else:
                probabilities["away_win"] += probability
        return probabilities

    def _most_likely_scoreline_for_outcome(self, distribution, outcome_name):
        best_home_goals = 0
        best_away_goals = 0
        best_probability = -1.0

        for home_goals, away_goals, probability in distribution:
            if outcome_name == "home_win" and home_goals <= away_goals:
                continue
            if outcome_name == "draw" and home_goals != away_goals:
                continue
            if outcome_name == "away_win" and home_goals >= away_goals:
                continue
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
