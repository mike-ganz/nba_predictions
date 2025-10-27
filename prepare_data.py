from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from generate_team_stats import generate_team_stats


TEAM_NAME_TO_ABBR = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS",
}


def _get_value(row: dict, *keys):
    for key in keys:
        variants = {
            key,
            key.upper(),
            key.lower(),
            key.title(),
            key.replace("_", " "),
            key.replace("_", " ").upper(),
            key.replace("_", " ").lower(),
        }
        for variant in variants:
            if variant in row:
                value = row[variant]
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                return value
    return None


def parse_market_value(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    for token in ("o", "u"):
        s = s.replace(f"{token} ", f"{token}")
    s = s.replace("o", " ").replace("u", " ")
    match = re.search(r"[-+]?\d+\.?\d*", s)
    return float(match.group()) if match else None


def parse_spread(row: dict, fallback_row: dict | None = None) -> float | None:
    raw = _get_value(row, "CLOSING SPREAD", "OPENING SPREAD")
    if raw is None and fallback_row is not None:
        raw = _get_value(fallback_row, "CLOSING SPREAD", "OPENING SPREAD")
    if raw is None:
        raw = _get_value(row, "LINE  MOVEMENT #3", "LINE  MOVEMENT #2", "LINE  MOVEMENT #1")
    spread = parse_market_value(raw)
    if spread is None or abs(spread) > 40:
        return None
    return spread


def parse_total(row: dict, fallback_row: dict | None = None) -> float | None:
    raw = _get_value(row, "CLOSING TOTAL")
    total = parse_market_value(raw)
    if total is None and fallback_row is not None:
        raw = _get_value(fallback_row, "CLOSING TOTAL")
        total = parse_market_value(raw)
    if total is None:
        raw = _get_value(row, "OPENING TOTAL")
        total = parse_market_value(raw)
    if total is None:
        odds = _get_value(row, "CLOSING ODDS", "OPENING ODDS")
        if isinstance(odds, str):
            match = re.search(r"(\d+\.?\d*)[ou]", odds.lower())
            if match:
                total = float(match.group(1))
    if total is None or not (150 <= total <= 280):
        return None
    return total


def parse_moneyline(row: dict, fallback_row: dict | None = None) -> int | None:
    raw = _get_value(row, "MONEYLINE", "CLOSING MONEYLINE", "OPENING MONEYLINE")
    if raw is None and fallback_row is not None:
        raw = _get_value(fallback_row, "MONEYLINE", "CLOSING MONEYLINE", "OPENING MONEYLINE")
    if raw is None:
        return None
    s = str(raw).strip().replace(" ", "")
    if s.upper() in {"PK", "PICK"}:
        return 0
    if s.lower() == "even":
        return 100
    match = re.search(r"^[-+]?\d+", s)
    if not match:
        return None
    return int(match.group())


def parse_venue_columns(df: pd.DataFrame) -> str:
    for cand in ["VENUE", "VENUE (R/H/N)", "VENUE_R_H_N", "VENUE_RH"]:
        if cand in df.columns:
            return cand
    raise KeyError("No venue column found for venue identification")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().replace("\n", " "))


SEASON_FALLBACK = {
    "2024-2025": "2023-2024",
    "2023-2024": "2022-2023",
    "2022-2023": "2021-2022",
    "2021-2022": "2020-2021",
}

LEAGUE_AVERAGES = {
    "off_rating": 112.0,
    "def_rating": 112.0,
    "pace": 99.0,
    "three_pt_rate": 0.38,
    "free_throw_rate": 0.22,
    "off_reb_rate": 0.26,
    "def_reb_rate": 0.74,
    "assist_rate": 0.62,
    "turnover_rate": 0.13,
    "rest_days": 2,
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def get_team_features(team_name: str, date_str: str, season: str) -> Tuple[Dict[str, float], Dict[str, object]]:
    stats = generate_team_stats(team_name, target_date=date_str, fallback_season=season)

    if stats is None:
        stats = {
            "OEFF": LEAGUE_AVERAGES["off_rating"],
            "DEFF": LEAGUE_AVERAGES["def_rating"],
            "PACE": LEAGUE_AVERAGES["pace"],
            "3PAr": LEAGUE_AVERAGES["three_pt_rate"],
            "FTr": LEAGUE_AVERAGES["free_throw_rate"],
            "ORr": LEAGUE_AVERAGES["off_reb_rate"],
            "DRr": LEAGUE_AVERAGES["def_reb_rate"],
            "ASTr": LEAGUE_AVERAGES["assist_rate"],
            "TOr": LEAGUE_AVERAGES["turnover_rate"],
            "REST_DAYS": LEAGUE_AVERAGES["rest_days"],
            "STAT_SOURCE": "league_average",
            "SEASON_USED": season,
            "PRIMARY_SEASON": season,
            "GAMES_PLAYED": 0,
            "GAMES_CURRENT_SEASON": 0,
        }

    rest_raw = stats.get("REST_DAYS")
    if rest_raw is None:
        rest_bucket = 2
    elif rest_raw >= 3:
        rest_bucket = 3
    else:
        rest_bucket = int(max(0, rest_raw))

    abbr = TEAM_NAME_TO_ABBR.get(team_name, team_name[:3].upper())

    features = {
        "team_id": abbr,
        "team_name": team_name,
        "off_rating": _clamp(stats["OEFF"], 80.0, 140.0),
        "def_rating": _clamp(stats["DEFF"], 80.0, 140.0),
        "pace": _clamp(stats["PACE"], 85.0, 115.0),
        "three_pt_rate": _clamp(stats["3PAr"], 0.2, 0.6),
        "free_throw_rate": _clamp(stats["FTr"], 0.1, 0.5),
        "off_reb_rate": _clamp(stats["ORr"], 0.15, 0.4),
        "def_reb_rate": _clamp(stats["DRr"], 0.65, 0.9),
        "assist_rate": _clamp(stats["ASTr"], 0.4, 0.7),
        "turnover_rate": _clamp(stats["TOr"], 0.1, 0.2),
        "rest_days": _clamp(rest_bucket, 0.0, 10.0),
    }

    metadata = {
        "stat_source": stats.get("STAT_SOURCE"),
        "season_used": stats.get("SEASON_USED"),
        "primary_season": stats.get("PRIMARY_SEASON"),
        "games_played_source": stats.get("GAMES_PLAYED"),
        "games_current_season": stats.get("GAMES_CURRENT_SEASON"),
        "rest_days_raw": rest_raw,
    }

    return features, metadata


def build_game_record(game_rows: pd.DataFrame, season: str, venue_col: str) -> Dict:
    rows = game_rows.to_dict("records")
    home_row = next((r for r in rows if str(r.get(venue_col, "")).upper().startswith("H")), None)
    away_row = next((r for r in rows if str(r.get(venue_col, "")).upper().startswith("R")), None)
    if home_row is None or away_row is None:
        logging.warning("Skipping game %s due to venue issue", rows[0].get("GAME-ID"))
        return {}

    date = pd.to_datetime(home_row.get("DATE"))
    date_str = date.strftime("%Y-%m-%d")
    away_team_name = away_row.get("TEAM")
    home_team_name = home_row.get("TEAM")

    away_features, away_meta = get_team_features(away_team_name, date_str, season)
    home_features, home_meta = get_team_features(home_team_name, date_str, season)

    spread_home = parse_spread(home_row, fallback_row=away_row)
    total = parse_total(home_row, fallback_row=away_row)
    moneyline_home = parse_moneyline(home_row)
    moneyline_away = parse_moneyline(away_row, fallback_row=home_row)
    if None in {spread_home, total, moneyline_home, moneyline_away}:
        logging.warning("Skipping game %s due to missing market", rows[0].get("GAME-ID"))
        return {}

    home_score = int(home_row.get("F", 0) or 0)
    away_score = int(away_row.get("F", 0) or 0)

    game_id = f"{date_str}-{away_features['team_id']}-{home_features['team_id']}"

    return {
        "game_id": game_id,
        "season": season,
        "date": date_str,
        "teams": {
            "A": away_features,
            "H": home_features,
        },
        "market": {
            "spread_home": spread_home,
            "total": total,
            "moneyline_home": moneyline_home,
            "moneyline_away": moneyline_away,
        },
        "outcome": {
            "home_final": home_score,
            "away_final": away_score,
        },
        "metadata": {
            "source_game_id": rows[0].get("GAME-ID"),
            "team_stats": {
                "away": away_meta,
                "home": home_meta,
            },
        },
    }


def process_file(path: Path, season: str) -> List[Dict]:
    logging.info("Processing %s", path)
    df = pd.read_excel(path)
    df = normalize_columns(df)
    venue_col = parse_venue_columns(df)
    games = []
    for game_id, group in df.groupby("GAME-ID"):
        record = build_game_record(group, season, venue_col)
        if record:
            games.append(record)
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare game-level JSONL using season aggregates")
    parser.add_argument("--team-boxscores-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seasons", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    team_dir = Path(args.team_boxscores_dir)
    if not team_dir.exists():
        parser.error(f"Team boxscores directory not found: {team_dir}")

    season_files = sorted(team_dir.glob("*.xlsx"))
    if args.seasons:
        selected = [f for f in season_files if any(season in f.stem for season in args.seasons)]
    else:
        selected = season_files

    all_games: List[Dict] = []
    for file_path in selected:
        season_match = None
        for candidate in ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]:
            if candidate in file_path.stem:
                season_match = candidate
                break
        if not season_match:
            logging.warning("Skipping %s: season not detected", file_path)
            continue
        all_games.extend(process_file(file_path, season_match))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_games:
            f.write(json.dumps(record) + "\n")

    logging.info("Wrote %d game records to %s", len(all_games), output_path)


if __name__ == "__main__":
    main()
