from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from features.availability import load_baselines_from_csv

TEAM_ABBREVIATIONS = {
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
    "Cleveland Cavaliers": "CLE",
    "Los Angeles Lakers": "LAL",
    "Los Angeles Clippers": "LAC",
    "San Antonio Spurs": "SAS",
    "New Jersey": "NJN",
}


def normalize_column(name: str) -> str:
    name = name.replace("\n", " ").strip()
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def parse_rest_days(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"b2b", "back_to_back"}:
        return 0.0
    digits = re.findall(r"\d+", s)
    if digits:
        return float(digits[0])
    return None


def parse_moneyline(value) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.upper() in {"PK", "PICK"}:
        return 0
    s = s.replace(" ", "")
    match = re.match(r"^[+-]?\d+", s)
    return int(match.group()) if match else None


def parse_market_value(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    for token in ("o", "u"):
        s = s.replace(f"{token} ", f"{token}")
    s = s.replace("o", " ").replace("u", " ")
    if "-" in s and not s.lstrip().startswith("-"):
        s = s.split("-")[0]
    if "+" in s and not s.lstrip().startswith("+"):
        s = s.split("+")[0]
    match = re.search(r"[-+]?\d+\.?\d*", s)
    return float(match.group()) if match else None


def parse_total(row: pd.Series) -> float | None:
    total_value = row.get("closing_total")
    total = parse_market_value(total_value)
    if total is None or total < 150:
        total = parse_market_value(row.get("opening_total"))
    if total is None or total < 150:
        total = parse_market_value(row.get("closing_odds"))
    return total if total is not None and 150 <= total <= 280 else None


def parse_spread(value) -> float | None:
    spread = parse_market_value(value)
    if spread is None:
        return None
    if abs(spread) > 40:
        return None
    return spread


def team_id_from_name(name: str) -> str:
    if pd.isna(name):
        raise ValueError("Missing team name")
    name = str(name).strip()
    if name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[name]
    return re.sub(r"[^A-Z]", "", name.upper())[:3] or name[:3].upper()


def compute_team_features(row: pd.Series, opponent: pd.Series) -> Dict[str, float | int | None]:
    fga = float(row.get("fga", 0) or 0)
    three_pa = float(row.get("3p", 0) or 0)
    fta = float(row.get("ft", 0) or 0)
    fta_attempts = float(row.get("fta", 0) or 0)
    assists = float(row.get("a", 0) or 0)
    turnovers = float(row.get("to", 0) or 0)
    poss = float(row.get("poss", 0) or 0)
    offensive_reb = float(row.get("or", 0) or 0)
    defensive_reb = float(row.get("dr", 0) or 0)
    opponent_defensive = float(opponent.get("dr", 0) or 0)
    opponent_offensive = float(opponent.get("or", 0) or 0)

    three_pt_rate = (three_pa / fga) if fga else 0.0
    free_throw_rate = (fta_attempts / fga) if fga else 0.0
    assist_rate = (assists / max(row.get("fg", 1), 1)) if row.get("fg", 0) else 0.0
    turnover_rate = (turnovers / poss) if poss else 0.0
    orb_rate = offensive_reb / (offensive_reb + max(opponent_defensive, 1)) if (offensive_reb + opponent_defensive) else 0.0
    drb_rate = defensive_reb / (defensive_reb + max(opponent_offensive, 1)) if (defensive_reb + opponent_offensive) else 0.0
    total_reb = orb_rate + drb_rate
    if total_reb > 0.98:
        scale = 0.98 / total_reb
        orb_rate *= scale
        drb_rate *= scale

    three_pt_rate = min(max(three_pt_rate, 0.2), 0.6)
    free_throw_rate = min(max(free_throw_rate, 0.1), 0.5)
    assist_rate = float(np.clip(assist_rate, 0.4, 0.7))
    turnover_rate = float(np.clip(turnover_rate, 0.1, 0.2))
    orb_rate = float(np.clip(orb_rate, 0.15, 0.4))
    drb_rate = float(np.clip(drb_rate, 0.65, 0.9))

    return {
        "team_id": team_id_from_name(row.get("team")),
        "off_rating": float(np.clip(row.get("oeff", 0) or 0.0, 80.0, 140.0)),
        "def_rating": float(np.clip(row.get("deff", 0) or 0.0, 80.0, 140.0)),
        "pace": float(np.clip(row.get("pace", 0) or 0.0, 85.0, 115.0)),
        "three_pt_rate": float(three_pt_rate),
        "free_throw_rate": float(free_throw_rate),
        "off_reb_rate": float(orb_rate),
        "def_reb_rate": float(drb_rate),
        "assist_rate": float(assist_rate),
        "turnover_rate": float(turnover_rate),
        "rest_days": parse_rest_days(row.get("team_rest_days", row.get("rest_days"))),
    }


def resolve_venue_column(df: pd.DataFrame) -> str:
    for cand in ["venue_r_h_n", "venue", "venue_r_h", "venue_rh"]:
        if cand in df.columns:
            return cand
    raise KeyError("No venue column found")


def build_game_record(game_rows: pd.DataFrame, season: str, venue_col: str) -> Dict:
    rows = game_rows.to_dict("records")
    home_row = next((r for r in rows if str(r.get(venue_col, "")).upper().startswith("H")), None)
    away_row = next((r for r in rows if str(r.get(venue_col, "")).upper().startswith("R")), None)
    if home_row is None or away_row is None:
        logging.warning("Skipping game %s due to missing home/away rows", rows[0].get("game_id"))
        return {}

    date = pd.to_datetime(home_row.get("date"))
    date_str = date.strftime("%Y-%m-%d")
    home_team = compute_team_features(pd.Series(home_row), pd.Series(away_row))
    away_team = compute_team_features(pd.Series(away_row), pd.Series(home_row))

    spread_home = parse_spread(home_row.get("closing_spread"))
    total = parse_total(pd.Series(home_row))
    moneyline_home = parse_moneyline(home_row.get("moneyline"))
    moneyline_away = parse_moneyline(away_row.get("moneyline"))

    if None in {spread_home, total, moneyline_home, moneyline_away}:
        logging.warning("Skipping game %s due to missing market data", rows[0].get("game_id"))
        return {}

    home_score = int(float(home_row.get("f", 0) or 0))
    away_score = int(float(away_row.get("f", 0) or 0))

    game_id = f"{date_str}-{away_team['team_id']}-{home_team['team_id']}"

    return {
        "game_id": game_id,
        "season": season,
        "date": date_str,
        "teams": {
            "A": {k: v for k, v in away_team.items() if k != "team_id"} | {"team_id": away_team["team_id"]},
            "H": {k: v for k, v in home_team.items() if k != "team_id"} | {"team_id": home_team["team_id"]},
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
            "source_game_id": rows[0].get("game_id"),
        },
    }


def process_file(path: Path, season: str) -> List[Dict]:
    logging.info("Processing %s", path)
    df = pd.read_excel(path)
    df.columns = [normalize_column(c) for c in df.columns]
    venue_col = resolve_venue_column(df)
    games = []
    for game_id, group in df.groupby("game_id"):
        if len(group) < 2:
            continue
        record = build_game_record(group, season, venue_col)
        if record:
            games.append(record)
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare game-level JSONL for direct prediction pipeline")
    parser.add_argument("--team-boxscores-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seasons", nargs="*", default=None, help="Seasons to include, e.g., 2022-2023 2023-2024")
    parser.add_argument("--availability-baselines", type=str, default=None, help="CSV file for player baseline minutes/TS/usage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    team_dir = Path(args.team_boxscores_dir)
    if not team_dir.exists():
        parser.error(f"Team boxscores directory not found: {team_dir}")

    if args.availability_baselines:
        baseline_path = Path(args.availability_baselines)
        if baseline_path.exists():
            logging.info("Loading availability baselines from %s", baseline_path)
            load_baselines_from_csv(str(baseline_path))
        else:
            logging.warning("Availability baseline file not found: %s", baseline_path)

    season_files = sorted(team_dir.glob("*.xlsx"))
    selected: Iterable[Path]
    if args.seasons:
        selected = [f for f in season_files if any(season in f.stem for season in args.seasons)]
    else:
        selected = season_files

    all_games: List[Dict] = []
    for file_path in selected:
        season_match = re.search(r"(\d{4}-\d{4})", file_path.stem)
        season = season_match.group(1) if season_match else "unknown"
        games = process_file(file_path, season)
        all_games.extend(games)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_games:
            f.write(json.dumps(record) + "\n")

    logging.info("Wrote %d game records to %s", len(all_games), output_path)


if __name__ == "__main__":
    main()
