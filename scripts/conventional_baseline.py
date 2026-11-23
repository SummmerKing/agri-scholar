"""Conventional-practice baseline: 150 kg N/ha in 3 splits and 6 x 40 mm irrigation, applied by the
managers already in the baseline .apsimx file. No agents and no search; the reward is computed the
same way as in the main pipeline for comparison.
"""
import os
import json
import shutil
import datetime

from agri_scholar.logging_utils import get_logger, section
from agri_scholar.simulation.apsim import ApsimController
from agri_scholar.simulation import reward as rm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "LOCATIONS": ["Haveri"],
    "YEARS": [2015],

    # baseline file with the conventional-practice managers enabled (150 kg N in 3 splits, 6 x 40 mm irrigation)
    "APSIM_BASE_FILE": os.path.join(ROOT, "apsimx", "Maize_Haveri_pop.apsimx"),

    "MET_DIR": os.path.join(ROOT, "data", "weather_maize_pilot"),
    "SCRATCH_DIR": os.path.join(ROOT, "scratch_sims_baseline"),
    "RESULTS_DIR": os.path.join(ROOT, "results", "conventional_baseline"),
    "LOG_DIR": os.path.join(ROOT, "logs", "conventional_baseline"),

    "SOIL_PROFILES": {
        "Mandya": {
            "order": "Alfisol", "type": "Sandy Loam", "lat": 12.52, "lon": 76.89,
            "bd": [1.45, 1.48, 1.50, 1.50, 1.50, 1.50, 1.50],
            "airdry": [0.05, 0.08, 0.11, 0.12, 0.12, 0.12, 0.12],
            "ll15": [0.10, 0.10, 0.11, 0.12, 0.12, 0.12, 0.12],
            "dul": [0.22, 0.22, 0.24, 0.25, 0.25, 0.25, 0.25],
            "sat": [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
            "ks": [100.0] * 7, "carbon": [0.50, 0.30, 0.20, 0.10, 0.05, 0.05, 0.05],
        },
        "Dharwad": {
            "order": "Vertisol", "type": "Clay", "lat": 15.4589, "lon": 75.0078,
            "bd": [1.15, 1.15, 1.15, 1.15, 1.20, 1.20, 1.20],
            "airdry": [0.15, 0.20, 0.25, 0.28, 0.28, 0.28, 0.28],
            "ll15": [0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28],
            "dul": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
            "sat": [0.54, 0.54, 0.54, 0.54, 0.53, 0.53, 0.53],
            "ks": [10.0] * 7, "carbon": [0.60, 0.40, 0.30, 0.20, 0.10, 0.10, 0.10],
        },
        "Davanagere": {
            "order": "Alfisol", "type": "Red Brown", "lat": 14.4644, "lon": 75.9218,
            "bd": [1.40, 1.40, 1.45, 1.45, 1.45, 1.45, 1.45],
            "airdry": [0.08, 0.10, 0.12, 0.14, 0.14, 0.14, 0.14],
            "ll15": [0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15],
            "dul": [0.28, 0.28, 0.30, 0.30, 0.30, 0.30, 0.30],
            "sat": [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
            "ks": [70.0] * 7, "carbon": [0.45, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05],
        },
        "Haveri": {
            "order": "Inceptisol", "type": "Transitional", "lat": 14.7937, "lon": 75.3983,
            "bd": [1.30, 1.30, 1.35, 1.35, 1.35, 1.35, 1.35],
            "airdry": [0.10, 0.15, 0.20, 0.20, 0.20, 0.20, 0.20],
            "ll15": [0.20, 0.20, 0.22, 0.22, 0.22, 0.22, 0.22],
            "dul": [0.38, 0.38, 0.40, 0.40, 0.40, 0.40, 0.40],
            "sat": [0.48, 0.48, 0.47, 0.47, 0.47, 0.47, 0.47],
            "ks": [50.0] * 7, "carbon": [0.55, 0.35, 0.25, 0.15, 0.05, 0.05, 0.05],
        },
    },

    # only used to rebuild C_F / C_I / C_L for the reward; APSIM applies these through its own managers
    "BASELINE_N_TOTAL_KG": 150.0,
    "BASELINE_N_SPLITS": 3,
    "BASELINE_IRRIG_EVENTS": 6,
    "BASELINE_IRRIG_MM_EACH": 40.0,

    "GRAIN_PRICE_PER_KG": 20.90,
    "FERTILIZER_PRICE_PER_KG": {
        "Urea": 13.04,      # pure elemental-N-equivalent price for UreaN applications
        "DAP": 27.00,
        "Mulch": 2.00,
    },
    "WATER_PRICE_PER_MM": 2.50,
    "LABOR_PRICE_PER_OPERATION": 350.0,

    "SHADOW_COST_WATERLOGGING": 5000.0,
    "SHADOW_COST_N_OVERSUPPLY": 2000.0,
    "SHADOW_COST_DEEP_DRAINAGE": 15.0,
    "SHADOW_COST_N_LEACHING": 25.0,
    "SHADOW_COST_SOC_LOSS": 150.0,

    "CROP_FAILURE_PENALTY": 50000.0,
}


def run_baseline_season(cfg: dict, location: str, year: int, logger) -> dict:
    met_path = os.path.join(cfg["MET_DIR"], f"{location}_{year}.met")
    section(logger, f"BASELINE SEASON START: {location} {year}")

    trial_dir = os.path.join(cfg["SCRATCH_DIR"], f"baseline_{location}_{year}")
    working_file = os.path.join(trial_dir, f"{location}_{year}_baseline.apsimx")

    # the baseline file's own managers stay enabled
    ctrl = ApsimController(cfg["APSIM_BASE_FILE"], working_file)
    ctrl.set_clock_and_weather(year, met_path)
    ctrl.set_location_soil_profile(location, cfg["SOIL_PROFILES"])
    logger.info(f"Baseline file configured for {location} {year}: "
                f"clock/weather + soil profile set, native fertiliser/"
                f"irrigation managers left enabled (no agent control).")

    try:
        # sowing date is not known here, so only season totals are reported
        diag = ctrl.run_and_get_yield()
        yield_val = diag.get("yield_kg_ha", 0)
        logger.info(f"Baseline run diagnostics ({location} {year}): "
                    f"status={diag.get('status')}, yield={yield_val} kg/ha")

        n_per_split = cfg["BASELINE_N_TOTAL_KG"] / cfg["BASELINE_N_SPLITS"]
        materials = [rm.MaterialApplication("Urea", n_per_split)
                     for _ in range(cfg["BASELINE_N_SPLITS"])]
        irrigations = [rm.IrrigationEvent(cfg["BASELINE_IRRIG_MM_EACH"])
                        for _ in range(cfg["BASELINE_IRRIG_EVENTS"])]

        prices = rm.Prices(
            grain_price_per_kg=cfg["GRAIN_PRICE_PER_KG"],
            fertilizer_price_per_kg=cfg["FERTILIZER_PRICE_PER_KG"],
            water_price_per_mm=cfg["WATER_PRICE_PER_MM"],
            labor_price_per_operation=cfg["LABOR_PRICE_PER_OPERATION"],
        )
        shadow_costs = rm.ShadowCosts(
            waterlogging=cfg["SHADOW_COST_WATERLOGGING"],
            n_oversupply=cfg["SHADOW_COST_N_OVERSUPPLY"],
            deep_drainage=cfg["SHADOW_COST_DEEP_DRAINAGE"],
            n_leaching=cfg["SHADOW_COST_N_LEACHING"],
            soc_loss=cfg["SHADOW_COST_SOC_LOSS"],
        )
        indicators = rm.SustainabilityIndicators(
            I_wl=diag.get("waterlogging_index", 0.0),
            I_no=diag.get("n_oversupply_index", 0.0),
            I_dd=diag.get("total_drainage_mm", 0.0),
            I_nl=diag.get("total_leach_no3_nh4", 0.0),
            I_cl=diag.get("soc_loss_index", 0.0),
        )

        reward_breakdown = rm.compute_reward(
            yield_val, materials, irrigations, indicators, prices, shadow_costs,
            crop_failure_penalty=cfg["CROP_FAILURE_PENALTY"],
        )

        logger.info(f"Baseline reward ({location} {year}): "
                    f"yield={yield_val} kg/ha, reward=Rs.{reward_breakdown['reward']:.2f} "
                    f"(R_rev={reward_breakdown['R_rev']:.2f}, "
                    f"C_prod={reward_breakdown['C_prod']:.2f}, "
                    f"P_sust={reward_breakdown['P_sust']:.2f})")

        return {
            "location": location, "year": year,
            "yield_kg_ha": yield_val,
            "reward_breakdown": reward_breakdown,
            "diagnostics": diag,
            "baseline_recipe": {
                "n_total_kg": cfg["BASELINE_N_TOTAL_KG"],
                "n_splits": cfg["BASELINE_N_SPLITS"],
                "irrig_events": cfg["BASELINE_IRRIG_EVENTS"],
                "irrig_mm_each": cfg["BASELINE_IRRIG_MM_EACH"],
            },
        }
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def main():
    cfg = CONFIG
    os.makedirs(cfg["SCRATCH_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for location in cfg["LOCATIONS"]:
        for year in cfg["YEARS"]:
            logger_name = f"agri_sigra_baseline_{location}_{year}"
            logger = get_logger(logger_name, run_tag=f"{run_tag}_{location}_{year}",
                                 log_dir=cfg["LOG_DIR"])
            try:
                result = run_baseline_season(cfg, location, year, logger)
                result["status"] = "ok"
            except Exception:
                logger.exception(f"Baseline season {location} {year} failed.")
                result = {"location": location, "year": year, "status": "failed"}
            all_results.append(result)

            out_path = os.path.join(cfg["RESULTS_DIR"],
                                     f"agri_sigra_baseline_{location}_{year}_{run_tag}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"{location} {year}: {result['status']} -> {out_path}")

    combined_path = os.path.join(cfg["RESULTS_DIR"], f"agri_sigra_baseline_combined_{run_tag}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nBASELINE RUN COMPLETE")
    for r in all_results:
        if r.get("status") == "ok":
            print(f"{r['location']} {r['year']}: yield={r['yield_kg_ha']:.0f} kg/ha, "
                  f"reward=Rs.{r['reward_breakdown']['reward']:.2f}")
        else:
            print(f"{r['location']} {r['year']}: FAILED")


if __name__ == "__main__":
    main()
