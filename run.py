"""Run an Agri-SCHOLAR experiment.

    python run.py --config configs/mandya.json
    python run.py --config configs/ablation_pop_bounds.json --years 2023 2024 --gpus 0 1
"""
import argparse

from agri_scholar.config import load_config
from agri_scholar.pipeline import run_experiment


def main():
    ap = argparse.ArgumentParser(description="Run Agri-SCHOLAR seasons.")
    ap.add_argument("--config", help="experiment file in configs/ (default settings if omitted)")
    ap.add_argument("--locations", nargs="+", help="override LOCATIONS")
    ap.add_argument("--years", nargs="+", type=int, help="override YEARS")
    ap.add_argument("--gpus", nargs="+", type=int, help="override GPU_IDS")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.locations or a.years:
        cfg["JOBS"] = None
        cfg["LOCATIONS"] = a.locations or cfg["LOCATIONS"]
        cfg["YEARS"] = a.years or cfg["YEARS"]
    if a.gpus:
        cfg["GPU_IDS"] = a.gpus
    run_experiment(cfg)


if __name__ == "__main__":
    main()
