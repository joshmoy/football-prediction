import json
import os
import re

import pandas as pd
import requests
from requests import HTTPError


class GeminiTeamContextEnricher:
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-2.5-flash"
    NUMERIC_COLUMNS = [
        "squad_strength",
        "availability_score",
        "expected_lineup_strength",
        "injury_count",
        "suspended_count",
    ]
    ABSOLUTE_BOUNDS = {
        "squad_strength": (0.0, 100.0),
        "availability_score": (0.0, 100.0),
        "expected_lineup_strength": (0.0, 100.0),
        "injury_count": (0.0, 15.0),
        "suspended_count": (0.0, 10.0),
    }
    NEUTRAL_PRIORS = {
        "squad_strength": 85.0,
        "availability_score": 85.0,
        "expected_lineup_strength": 84.0,
        "injury_count": 2.0,
        "suspended_count": 0.5,
    }
    MAX_DELTAS = {
        "squad_strength": 8.0,
        "availability_score": 10.0,
        "expected_lineup_strength": 10.0,
        "injury_count": 3.0,
        "suspended_count": 2.0,
    }

    def __init__(self, api_key=None, model=None, timeout=90):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.timeout = timeout

    def build_team_context(self, fixtures_df, competition_code="PL"):
        if fixtures_df.empty:
            return self._empty_context_frame()

        if not self.api_key:
            raise ValueError(
                "Gemini context requested but no Gemini API key was found. Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --gemini-api-key."
            )

        team_requests = self._build_team_requests(fixtures_df)
        response_payload = self._call_gemini(team_requests, competition_code)
        response_text = self._extract_output_text(response_payload)
        parsed = self._parse_json_payload(response_text)
        context_df = self._normalise_team_context(parsed, team_requests)
        if context_df.empty:
            raise ValueError("Gemini did not return any team context rows.")
        return context_df

    def _build_team_requests(self, fixtures_df):
        fixture_rows = (
            fixtures_df[["home_team", "away_team", "fixture_date", "gameweek"]]
            .sort_values(["fixture_date", "home_team", "away_team"])
            .reset_index(drop=True)
        )
        team_requests = {}
        for _, fixture in fixture_rows.iterrows():
            for team in [fixture["home_team"], fixture["away_team"]]:
                if team in team_requests:
                    continue
                team_requests[team] = {
                    "team": team,
                    "effective_date": fixture["fixture_date"],
                    "gameweek": int(fixture["gameweek"]) if pd.notna(fixture["gameweek"]) else None,
                }
        return list(team_requests.values())

    def _call_gemini(self, team_requests, competition_code):
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": self._build_prompt(team_requests, competition_code)
                        }
                    ]
                }
            ],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.2,
            },
        }

        response = requests.post(
            f"{self.API_BASE_URL}/{self.model}:generateContent",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            error_message = response.text.strip() or str(exc)
            raise ValueError(f"Gemini request failed: {error_message}") from exc
        return response.json()

    def _build_prompt(self, team_requests, competition_code):
        requested_teams = [
            {
                "team": item["team"],
                "gameweek": item["gameweek"],
                "effective_date": item["effective_date"].strftime("%Y-%m-%d"),
            }
            for item in team_requests
        ]
        schema = {
            "teams": [
                {
                    "team": "string",
                    "squad_strength": "number 0-100",
                    "availability_score": "number 0-100",
                    "expected_lineup_strength": "number 0-100",
                    "injury_count": "integer >= 0",
                    "suspended_count": "integer >= 0",
                    "confidence": "number 0-1",
                    "notes": "short string",
                    "source_summary": ["short source descriptions"],
                }
            ]
        }
        return (
            f"You are enriching football prediction features for competition {competition_code}. "
            "Use Google Search grounding to find current injury news, suspensions, expected lineup information, "
            "and broad team-strength context. "
            "Return JSON only with no markdown fences or commentary. "
            "If some details are uncertain, estimate conservatively and reflect that in confidence. "
            "Scales: 100 means strongest/fullest availability, 0 means weakest/unavailable. "
            "Use official club/injury news and reputable sports reporting where available. "
            "Schema: "
            f"{json.dumps(schema)}. "
            f"Requested teams: {json.dumps(requested_teams)}"
        )

    def _extract_output_text(self, response_payload):
        candidates = response_payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(texts).strip()

    def _parse_json_payload(self, response_text):
        if not response_text:
            raise ValueError("Gemini returned an empty response.")

        cleaned = response_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        return json.loads(cleaned)

    def _normalise_team_context(self, parsed_payload, team_requests):
        rows = []
        team_lookup = {item["team"]: item for item in team_requests}

        for item in parsed_payload.get("teams", []):
            team_name = item.get("team")
            if team_name not in team_lookup:
                continue
            request_metadata = team_lookup[team_name]
            rows.append(
                {
                    "team": team_name,
                    "effective_date": request_metadata["effective_date"],
                    "gameweek": request_metadata["gameweek"],
                    "squad_strength": self._bounded_float(
                        item.get("squad_strength"), lower=0.0, upper=100.0
                    ),
                    "availability_score": self._bounded_float(
                        item.get("availability_score"), lower=0.0, upper=100.0
                    ),
                    "expected_lineup_strength": self._bounded_float(
                        item.get("expected_lineup_strength"), lower=0.0, upper=100.0
                    ),
                    "injury_count": self._bounded_float(
                        item.get("injury_count"), lower=0.0, upper=15.0
                    ),
                    "suspended_count": self._bounded_float(
                        item.get("suspended_count"), lower=0.0, upper=10.0
                    ),
                    "confidence": self._bounded_float(
                        item.get("confidence"), lower=0.0, upper=1.0
                    ),
                    "notes": str(item.get("notes", "")).strip(),
                    "source_summary": " | ".join(
                        str(value).strip()
                        for value in item.get("source_summary", [])
                        if str(value).strip()
                    ),
                }
            )

        if not rows:
            return self._empty_context_frame()

        context_df = pd.DataFrame(rows)
        context_df["effective_date"] = pd.to_datetime(context_df["effective_date"])
        return context_df

    def _empty_context_frame(self):
        return pd.DataFrame(
            columns=[
                "team",
                "effective_date",
                "gameweek",
                "squad_strength",
                "availability_score",
                "expected_lineup_strength",
                "injury_count",
                "suspended_count",
                "confidence",
                "notes",
                "source_summary",
            ]
        )

    def _bounded_float(self, value, lower, upper):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return lower
        return max(lower, min(upper, numeric))


def merge_team_context_frames(base_context_df, enrichment_df):
    if base_context_df is None or base_context_df.empty:
        return enrichment_df.reset_index(drop=True)
    if enrichment_df is None or enrichment_df.empty:
        return base_context_df.reset_index(drop=True)

    combined = pd.concat([base_context_df, enrichment_df], ignore_index=True, sort=False)
    subset = [
        column
        for column in ["team", "gameweek", "effective_date"]
        if column in combined.columns
    ]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    return combined.reset_index(drop=True)


def calibrate_team_context(base_context_df, enrichment_df, enrichment_weight=0.35):
    if enrichment_df is None or enrichment_df.empty:
        return enrichment_df

    calibrated_rows = []
    for _, row in enrichment_df.iterrows():
        team_name = row["team"]
        anchor_row = _find_anchor_row(base_context_df, row)
        calibrated = row.copy()

        for column in GeminiTeamContextEnricher.NUMERIC_COLUMNS:
            base_value = _anchor_value(anchor_row, column)
            raw_value = _safe_float(row.get(column), base_value)
            blended = base_value + ((raw_value - base_value) * enrichment_weight)
            max_delta = GeminiTeamContextEnricher.MAX_DELTAS[column]
            lower = max(
                GeminiTeamContextEnricher.ABSOLUTE_BOUNDS[column][0],
                base_value - max_delta,
            )
            upper = min(
                GeminiTeamContextEnricher.ABSOLUTE_BOUNDS[column][1],
                base_value + max_delta,
            )
            calibrated[column] = max(lower, min(upper, blended))

        confidence = _safe_float(row.get("confidence"), 0.5)
        calibrated["confidence"] = max(0.0, min(1.0, confidence * 0.85))
        note_prefix = f"Gemini-calibrated for {team_name}"
        raw_notes = str(row.get("notes", "")).strip()
        calibrated["notes"] = (
            f"{note_prefix}: {raw_notes}" if raw_notes else note_prefix
        )
        calibrated_rows.append(calibrated)

    calibrated_df = pd.DataFrame(calibrated_rows)
    if "effective_date" in calibrated_df.columns:
        calibrated_df["effective_date"] = pd.to_datetime(calibrated_df["effective_date"])
    return calibrated_df


def _find_anchor_row(base_context_df, enrichment_row):
    if base_context_df is None or base_context_df.empty:
        return None

    team_rows = base_context_df[base_context_df["team"] == enrichment_row["team"]].copy()
    if team_rows.empty:
        return None

    target_gameweek = enrichment_row.get("gameweek")
    if "gameweek" in team_rows.columns and pd.notna(target_gameweek):
        matching_gameweek = team_rows[team_rows["gameweek"] == target_gameweek]
        if not matching_gameweek.empty:
            team_rows = matching_gameweek

    if "effective_date" in team_rows.columns and team_rows["effective_date"].notna().any():
        target_date = enrichment_row.get("effective_date")
        dated_rows = team_rows[team_rows["effective_date"].notna()]
        if pd.notna(target_date):
            eligible_rows = dated_rows[dated_rows["effective_date"] <= target_date]
            if not eligible_rows.empty:
                team_rows = eligible_rows
            elif not dated_rows.empty:
                team_rows = dated_rows

    sort_columns = [
        column
        for column in ["effective_date", "gameweek"]
        if column in team_rows.columns
    ]
    if sort_columns:
        team_rows = team_rows.sort_values(sort_columns, na_position="last")
    return team_rows.iloc[-1]


def _anchor_value(anchor_row, column):
    if anchor_row is None:
        return GeminiTeamContextEnricher.NEUTRAL_PRIORS[column]
    return _safe_float(anchor_row.get(column), GeminiTeamContextEnricher.NEUTRAL_PRIORS[column])


def _safe_float(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(numeric):
        return float(default)
    return numeric
