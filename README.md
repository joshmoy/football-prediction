# Goborr AI Fixture Predictor

This project trains a simple match outcome model and predicts every fixture in a target gameweek.

It now supports two data sources:

- Bundled sample CSVs for local smoke tests
- Real Premier League data from [football-data.org](https://www.football-data.org/) using the free API tier

## Quick start

Use the bundled sample data:

```bash
./venv/bin/python main.py --data-source sample --gameweek 27
```

Use real Premier League fixtures and match history from football-data.org:

```bash
cp .env.example .env
# then set FOOTBALL_DATA_API_TOKEN in .env

./venv/bin/python main.py \
  --data-source football-data-api \
  --competition-code PL
```

Skip a partially remaining current matchweek and score the next future one instead:

```bash
./venv/bin/python main.py \
  --data-source football-data-api \
  --competition-code PL \
  --future-gameweek-only
```

Optional team context can still be supplied as a CSV overlay:

```bash
./venv/bin/python main.py \
  --data-source football-data-api \
  --competition-code PL \
  --historical-seasons 2022,2023,2024 \
  --team-context /absolute/path/to/team_context.csv
```

## Notes

- `football-data.org` gives this project real fixtures, results, season metadata, and team names.
- On the free tier, older historical seasons can be restricted. If you hit a `403`, retry with the current season only.
- The free tier does not give this repo rich pre-match injuries and predicted lineups, so those remain an optional local CSV overlay.
- The model is still a lightweight baseline. Real-world accuracy will improve most from better historical depth and better pre-match team context.

## FastAPI

Run the backend locally:

```bash
./venv/bin/uvicorn api_server:app --reload --port 8000
```

Available endpoints:

- `GET /health`
- `POST /predict`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data_source":"sample","gameweek":27,"competition_code":"PL"}'
```

## Railway

This repo now includes [railway.toml](/Users/joshua/Desktop/goborr-ai/railway.toml) for backend deployment on Railway.

Suggested Railway setup:

1. Deploy the `goborr-ai` repo as a Railway service
2. Add the environment variable `FOOTBALL_DATA_API_TOKEN`
3. Add `ALLOWED_ORIGINS` with your deployed frontend URL
4. Railway will use:
   - start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - healthcheck: `/health`

If you deploy from a monorepo or non-root folder, Railway's docs note that the config file path must be set explicitly in the service settings.
