# Sample Data Schema

The app now expects three CSV inputs:

`historical_matches.csv`
- One row per completed match.
- Required columns: `season`, `match_date`, `gameweek`, `home_team`, `away_team`, `home_goals`, `away_goals`.
- Optional pre-match signal columns the model can learn from:
  `home_xg`, `away_xg`, `home_shots`, `away_shots`,
  `home_lineup_strength`, `away_lineup_strength`,
  `home_availability_score`, `away_availability_score`,
  `home_squad_strength`, `away_squad_strength`,
  `home_injury_count`, `away_injury_count`,
  `home_suspended_count`, `away_suspended_count`.

`upcoming_fixtures.csv`
- One row per fixture to score.
- Required columns: `fixture_date`, `gameweek`, `home_team`, `away_team`.

`team_context.csv`
- Optional current team snapshot used at prediction time.
- Required columns: `team`.
- Recommended columns: `effective_date`, `gameweek`, `squad_strength`,
  `availability_score`, `expected_lineup_strength`, `injury_count`,
  `suspended_count`.

Example command:

```bash
python3 main.py --gameweek 27
```
