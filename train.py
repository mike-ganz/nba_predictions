"""Training entry point for the direct prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from features.availability import load_baselines_from_csv

from calibration.calibrator import Calibrator, CalibrationConfig
from data.loaders import GameDataLoader
from training.dataset import TrainingDataset
from training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bivariate Poisson model")
    parser.add_argument("--data", type=str, required=True, help="Path to game jsonl")
    parser.add_argument("--output", type=str, required=True, help="Directory to save model")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config")
    return parser.parse_args()


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)

    dataset = TrainingDataset(collection.games)
    baselines_path = cfg.get("data", {}).get("availability_baselines")
    if baselines_path and Path(baselines_path).exists():
        load_baselines_from_csv(baselines_path)
    trainer_cfg = TrainerConfig(**cfg.get("training", {}))
    trainer = Trainer(dataset, config=trainer_cfg)
    model, validation_payload = trainer.fit()

    calibration_cfg = CalibrationConfig(**cfg.get("calibration", {}))
    calibrator = Calibrator(calibration_cfg)
    calibrator.fit(
        margin_cdf=validation_payload["margin_cdf"],
        joint_probs=validation_payload["joint_pmfs"],
        margin_outcomes=validation_payload["margin_outcomes"],
        score_outcomes_home=validation_payload["score_outcomes_home"],
        score_outcomes_away=validation_payload["score_outcomes_away"],
        market_home=validation_payload["market_home"],
        market_away=validation_payload["market_away"],
        sigma_home=validation_payload.get("sigma_home"),
        sigma_away=validation_payload.get("sigma_away"),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(calibrator, output_dir / "calibrator.joblib")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg))


if __name__ == "__main__":
    main()

