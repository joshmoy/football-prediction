from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


class MatchPredictor:
    OUTCOME_MAP = {0: "away_win", 1: "draw", 2: "home_win"}

    def __init__(self, feature_columns, random_state=42):
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.model = GradientBoostingClassifier(random_state=random_state)
        self.validation_metrics = {}

    def train(self, training_df):
        if training_df.empty:
            raise ValueError("Cannot train predictor with an empty training dataset.")

        X = training_df[self.feature_columns].fillna(0.0)
        y = training_df["outcome"]

        if y.nunique() < 2:
            raise ValueError("Training data must contain at least two outcome classes.")

        should_validate = len(training_df) >= 18 and y.value_counts().min() >= 2
        if should_validate:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=self.random_state,
                stratify=y,
            )
            self.model.fit(X_train, y_train)
            valid_probabilities = self.model.predict_proba(X_valid)
            valid_predictions = self.model.predict(X_valid)
            self.validation_metrics = {
                "accuracy": accuracy_score(y_valid, valid_predictions),
                "log_loss": log_loss(y_valid, valid_probabilities, labels=self.model.classes_),
                "training_rows": len(training_df),
            }

        self.model.fit(X, y)
        return self.validation_metrics

    def predict_fixtures(self, fixtures_df):
        X = fixtures_df[self.feature_columns].fillna(0.0)
        probabilities = self.model.predict_proba(X)
        outcome_probabilities = self._normalise_probabilities(probabilities)

        results = fixtures_df[
            ["fixture_date", "gameweek", "home_team", "away_team"]
        ].copy()
        results["home_win_probability"] = outcome_probabilities["home_win"]
        results["draw_probability"] = outcome_probabilities["draw"]
        results["away_win_probability"] = outcome_probabilities["away_win"]
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
        return results.sort_values(["fixture_date", "home_team"]).reset_index(drop=True)

    def _normalise_probabilities(self, probabilities):
        probability_map = {name: [0.0] * len(probabilities) for name in self.OUTCOME_MAP.values()}
        for class_index, class_label in enumerate(self.model.classes_):
            outcome_name = self.OUTCOME_MAP[int(class_label)]
            probability_map[outcome_name] = probabilities[:, class_index]
        return probability_map
