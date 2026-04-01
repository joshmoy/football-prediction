import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests import HTTPError


class GoborrClient:
    DEFAULT_BASE_URL = "https://staging-api.goborr.com/api/v1"
    DEFAULT_COMPETITION_SLUG = "epl"
    TOKEN_CACHE_PATH = Path("artifacts/goborr/token_cache.json")
    TOKEN_TTL_HOURS = 24

    def __init__(
        self,
        base_url=None,
        email=None,
        password=None,
        token_cache_path=None,
        competition_slug=DEFAULT_COMPETITION_SLUG,
    ):
        load_dotenv()
        self.base_url = (base_url or os.getenv("GOBORR_API_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.email = email or os.getenv("GOBORR_EMAIL")
        self.password = password or os.getenv("GOBORR_PASSWORD")
        self.competition_slug = competition_slug or os.getenv("GOBORR_COMPETITION_SLUG") or self.DEFAULT_COMPETITION_SLUG
        self.token_cache_path = Path(token_cache_path or os.getenv("GOBORR_TOKEN_CACHE_PATH") or self.TOKEN_CACHE_PATH)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
                "User-Agent": "goborr-ai/1.0",
            }
        )

    def publish_prediction(self, round_number, home_team, away_team, home_score, away_score):
        token = self.get_token()
        fixtures = self.get_fixtures(round_number=round_number, token=token)
        fixture = self._match_fixture(fixtures, home_team=home_team, away_team=away_team)
        if fixture is None:
            raise ValueError(
                f"No Goborr fixture match found for round {round_number}: {home_team} vs {away_team}."
            )

        fixture_id = fixture["id"]
        payload = {
            "home_team_score": int(home_score),
            "away_team_score": int(away_score),
        }
        response = self.session.post(
            f"{self.base_url}/competition/{self.competition_slug}/fixture/{fixture_id}/predict",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=20,
        )
        self._raise_for_status(response, "publish prediction")
        response_payload = response.json()
        return {
            "fixture_id": fixture_id,
            "fixture": {
                "id": fixture_id,
                "round": fixture.get("round"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "kickoff_time": fixture.get("kickoff_time"),
            },
            "submitted_prediction": payload,
            "response": response_payload,
        }

    def get_fixtures(self, round_number, token=None):
        access_token = token or self.get_token()
        response = self.session.get(
            f"{self.base_url}/competition/{self.competition_slug}/fixture",
            params={"round": int(round_number)},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20,
        )
        self._raise_for_status(response, "fetch fixtures")
        body = response.json()
        return ((body.get("data") or {}).get("fixtures")) or []

    def get_token(self, force_refresh=False):
        if not force_refresh:
            cached = self._read_cached_token()
            if cached:
                return cached

        if not self.email or not self.password:
            raise ValueError(
                "Goborr credentials are missing. Set GOBORR_EMAIL and GOBORR_PASSWORD."
            )

        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"email": self.email, "password": self.password},
            timeout=20,
        )
        self._raise_for_status(response, "login")
        body = response.json()
        token = ((body.get("data") or {}).get("token")) or body.get("token")
        if not token:
            raise ValueError("Goborr login succeeded but no token was returned.")
        self._write_cached_token(token)
        return token

    def _match_fixture(self, fixtures, home_team, away_team):
        expected_home = self._normalise_name(home_team)
        expected_away = self._normalise_name(away_team)

        for fixture in fixtures:
            fixture_home = self._normalise_name(fixture.get("home_team"))
            fixture_away = self._normalise_name(fixture.get("away_team"))
            if fixture_home == expected_home and fixture_away == expected_away:
                return fixture
        return None

    def _normalise_name(self, team_name):
        aliases = {
            "manchester city": "man city",
            "manchester united": "man utd",
            "newcastle united": "newcastle",
            "tottenham": "spurs",
            "tottenham hotspur": "spurs",
            "wolverhampton": "wolves",
            "wolverhampton wanderers": "wolves",
            "nottingham forest": "nott'm forest",
            "west ham united": "west ham",
            "brighton & hove albion": "brighton",
            "ipswich town": "ipswich",
            "leicester city": "leicester",
            "leeds united": "leeds",
        }
        normalised = " ".join(str(team_name or "").lower().strip().split())
        return aliases.get(normalised, normalised)

    def _read_cached_token(self):
        if not self.token_cache_path.exists():
            return None

        try:
            payload = json.loads(self.token_cache_path.read_text())
        except Exception:
            return None

        token = payload.get("token")
        expires_at_raw = payload.get("expires_at")
        if not token or not expires_at_raw:
            return None

        try:
            expires_at = datetime.fromisoformat(expires_at_raw)
        except ValueError:
            return None

        if expires_at <= datetime.utcnow():
            return None

        return token

    def _write_cached_token(self, token):
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": token,
            "expires_at": (datetime.utcnow() + timedelta(hours=self.TOKEN_TTL_HOURS)).isoformat(),
        }
        self.token_cache_path.write_text(json.dumps(payload, indent=2))

    def _raise_for_status(self, response, action):
        try:
            response.raise_for_status()
        except HTTPError as exc:
            detail = response.text.strip() or str(exc)
            raise ValueError(f"Goborr {action} failed: {detail}") from exc
