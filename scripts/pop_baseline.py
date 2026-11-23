"""Government Package-of-Practices baseline: the fixed recipe in the PoP .apsimx file, run with APSIM's
own managers over a range of years. No agents, no GPU.

    python -m scripts.pop_baseline --location Mandya --base-file apsimx/Maize_Mandya_pop2.apsimx \
        --years 2015 2016 2017 --weather observed
"""
import os
import sys
import json
import shutil
import logging
import argparse
import datetime
import re
from typing import Dict, List, Tuple

from agri_scholar.logging_utils import get_logger, section
from agri_scholar.simulation import reward as rm
from agri_scholar.simulation.apsim import ApsimController
import pandas as pd
from agri_scholar.simulation import weather as wa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "LOCATION": "Haveri",
    "YEARS": [2015],

    "APSIM_BASE_FILE": os.path.join(ROOT, "apsimx", "Maize_Haveri_pop2.apsimx"),
    "BASELINE_WEATHER_DIR": os.path.join(ROOT, "data", "weather_baseline_pop2"),
    "OBSERVED_WEATHER_DIR": os.path.join(ROOT, "data", "weather_maize_pilot"),
    "RESULTS_DIR": os.path.join(ROOT, "results", "pop_baseline"),
    "SCRATCH_DIR": os.path.join(ROOT, "scratch_sims_pop_baseline"),
    "LOG_DIR": os.path.join(ROOT, "logs", "pop_baseline"),

    # soil profiles (same as agri_scholar/config.py)
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

    # prices and shadow costs (same as agri_scholar/config.py)
    "GRAIN_PRICE_PER_KG": 20.90,
    "FERTILIZER_PRICE_PER_KG": {"Urea": 13.04, "DAP": 27.00, "Mulch": 2.00},
    "WATER_PRICE_PER_MM": 2.50,
    "LABOR_PRICE_PER_OPERATION": 350.0,
    "SHADOW_COST_WATERLOGGING": 5000.0,
    "SHADOW_COST_N_OVERSUPPLY": 2000.0,
    "SHADOW_COST_DEEP_DRAINAGE": 15.0,
    "SHADOW_COST_N_LEACHING": 25.0,
    "SHADOW_COST_SOC_LOSS": 150.0,
    "CROP_FAILURE_PENALTY": 50000.0,

    # PoP recipe, overridden by the values found in the APSIM file
    "POP_N_TOTAL_KG": 150.0,
    "POP_N_SPLITS": 3,
    "POP_IRRIG_EVENTS": 6,
    "POP_IRRIG_MM_EACH": 40.0,
}


def inspect_apsimx_recipe(base_file: str) -> dict:
    """Inspects Fertiliser, Irrigation, and Sowing rules in the base APSIM file to determine PoP schedule."""
    recipe = {
        "fert_splits": 3,
        "fert_total_kg": 150.0,
        "irrig_events": 6,
        "irrig_mm_each": 40.0,
        "cultivar": "MAH-14-138",
        "no _rule": {}
    }
    if not os.path.exists(base_file):
        return recipe
    try:
        with open(base_file, "r") as f:
            data = json.load(f)
        def walk(node):
            if isinstance(node, dict):
                name = node.get("Name", "")
                if "Fertiliser" in name or "Fertilizer" in name:
                    code = " ".join(node.get("CodeArray", []))
                    m = re.search(r"double\[\]\s+Amount\s*=\s*new\s+double\[\d+\]\s*\{([^}]+)\}", code)
                    if m:
                        vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
                        pos = [v for v in vals if v > 0]
                        if pos:
                            recipe["fert_splits"] = len(pos)
                            recipe["fert_total_kg"] = sum(pos)
                elif "Irrigation" in name:
                    code = " ".join(node.get("CodeArray", []))
                    m = re.search(r"double\[\]\s+Amount\s*=\s*new\s+double\[\d+\]\s*\{([^}]+)\}", code)
                    if m:
                        vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
                        pos = [v for v in vals if v > 0]
                        recipe["irrig_events"] = len(pos)
                        recipe["irrig_mm_each"] = (sum(pos) / len(pos)) if pos else 0.0
                elif name == "Sow using a variable rule":
                    params = {p["Key"]: p["Value"] for p in node.get("Parameters", [])}
                    recipe["sowing_rule"] = params
                    if "CultivarName" in params:
                        recipe["cultivar"] = params["CultivarName"]
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)
        walk(data)
    except Exception:
        pass
    return recipe


def _pop_materials_and_irrigations(cfg: dict) -> Tuple[List[rm.MaterialApplication], List[rm.IrrigationEvent]]:
    n_splits = cfg["POP_N_SPLITS"]
    n_total = cfg["POP_N_TOTAL_KG"]
    n_per_split = (n_total / n_splits) if n_splits > 0 else 0.0
    materials = [rm.MaterialApplication("Urea", n_per_split) for _ in range(n_splits)]
    irrigations = [rm.IrrigationEvent(cfg["POP_IRRIG_MM_EACH"]) for _ in range(cfg["POP_IRRIG_EVENTS"])]
    return materials, irrigations


def _prices_and_shadow(cfg: dict) -> Tuple[rm.Prices, rm.ShadowCosts]:
    prices = rm.Prices(
        grain_price_per_kg=cfg["GRAIN_PRICE_PER_KG"],
        fertilizer_price_per_kg=cfg["FERTILIZER_PRICE_PER_KG"],
        water_price_per_mm=cfg["WATER_PRICE_PER_MM"],
        labor_price_per_operation=cfg["LABOR_PRICE_PER_OPERATION"],
    )
    shadow = rm.ShadowCosts(
        waterlogging=cfg["SHADOW_COST_WATERLOGGING"],
        n_oversupply=cfg["SHADOW_COST_N_OVERSUPPLY"],
        deep_drainage=cfg["SHADOW_COST_DEEP_DRAINAGE"],
        n_leaching=cfg["SHADOW_COST_N_LEACHING"],
        soc_loss=cfg["SHADOW_COST_SOC_LOSS"],
    )
    return prices, shadow


def resolve_weather_file(cfg: dict, location: str, year: int, weather_mode: str, logger: logging.Logger) -> str:
    """Finds or constructs the weather file for the given season."""
    if weather_mode == "observed":
        met_path = os.path.join(cfg["OBSERVED_WEATHER_DIR"], f"{location}_{year}.met")
        if not os.path.exists(met_path):
            raise FileNotFoundError(f"Observed weather file not found: {met_path}")
        return os.path.abspath(met_path)

    met_path = os.path.join(cfg["BASELINE_WEATHER_DIR"], f"{location}_{year}.met")
    if not os.path.exists(met_path):
        logger.info(f"Baseline weather file not found at {met_path}. Generating via build_baseline_weather...")
        try:
            import build_baseline_weather as bbw
            cfg_bbw = dict(bbw.CONFIG)
            cfg_bbw["OUT_DIR"] = cfg["BASELINE_WEATHER_DIR"]
            out_path, _, _ = bbw.build_season_weather_file(cfg_bbw, location, year)
            met_path = out_path
        except Exception as e:
            raise FileNotFoundError(
                f"Could not build baseline weather for {location} {year}: {e}. "
                f"Ensure data/weather_maize_pilot/{location}_{year}.met exists."
            )

    return os.path.abspath(met_path)


def _to_plain_python(obj):
    """Recursively converts numpy/scalar types to pure Python types for clean JSON dump."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_python(x) for x in obj]
    return obj


def set_clock_and_weather_compat(ctrl: ApsimController, year: int, met_file: str,
                                start_date: str = None, end_date: str = None):
    """Set the weather file and simulation clock."""
    try:
        ctrl.set_clock_and_weather(year, met_file, start_date=start_date, end_date=end_date)
    except TypeError:
        ctrl.set_clock_and_weather(year, met_file)
        clock = ctrl._find_node(ctrl.data, "Clock")
        if clock:
            if start_date:
                clock["Start"] = start_date
            if end_date:
                clock["End"] = end_date
            ctrl._save()


def relax_sowing_in_controller(ctrl: ApsimController, min_rain: float = 10.0, min_esw: float = 10.0):
    """Ensures sowing criteria are relaxed with fallback on EndDate so sowing never fails."""
    modified = False
    def update(node):
        nonlocal modified
        if isinstance(node, dict):
            if node.get("Name") == "Sow using a variable rule":
                for p in node.get("Parameters", []):
                    try:
                        if p.get("Key") == "MinRain" and float(p.get("Value", 20)) > min_rain:
                            p["Value"] = str(min_rain)
                            modified = True
                        if p.get("Key") == "MinESW" and float(p.get("Value", 25)) > min_esw:
                            p["Value"] = str(min_esw)
                            modified = True
                        if p.get("Key") == "EndDate" and str(p.get("Value", "")).lower() in ("15-june", "15-jun"):
                            p["Value"] = "15-july"
                            modified = True
                    except (ValueError, TypeError):
                        pass
                code_array = node.get("CodeArray", [])
                code_text = " ".join(code_array)
                if "DateUtilities.DatesEqual(EndDate, Clock.Today)" not in code_text:
                    new_code = []
                    for line in code_array:
                        if "accumulatedRain.Sum > MinRain)" in line or "accumulatedRain.Sum >= MinRain)" in line:
                            new_code.append("                ((accumulatedRain.Sum >= MinRain && MathUtilities.Sum(waterBalance.ESW) >= MinESW) || DateUtilities.DatesEqual(EndDate, Clock.Today)))")
                            modified = True
                        elif "MathUtilities.Sum(waterBalance.ESW)" in line and "EndDate" not in line:
                            pass
                        else:
                            new_code.append(line)
                    node["CodeArray"] = new_code
            for v in node.values():
                update(v)
        elif isinstance(node, list):
            for x in node:
                update(x)
    update(ctrl.data)
    if modified:
        ctrl._save()


def calculate_sowing_date(met_path: str, year: int) -> str:
    df = wa.load_met_file(met_path)
    df = df[df["year"] == year].sort_values("day").reset_index(drop=True)
    rain_window = []
    for _, row in df.iterrows():
        day = int(row["day"])
        rain = float(row["rain"])
        date_obj = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1)
        if date_obj.month >= 6:
            rain_window.append((date_obj, rain))
            if len(rain_window) > 3:
                rain_window.pop(0)
            if len(rain_window) == 3:
                total_rain = sum(r for _, r in rain_window)
                all_above_2_5 = all(r > 2.5 for _, r in rain_window)
                if total_rain > 20.0 and all_above_2_5:
                    return rain_window[-1][0].strftime("%d-%b")
    return "15-Jul"

def force_sowing_date_in_controller(ctrl: ApsimController, sow_date_str: str):
    """Forces sowing on a specific date by setting StartDate=EndDate=sow_date_str and thresholds to 0."""
    modified = False
    def update(node):
        nonlocal modified
        if isinstance(node, dict):
            if node.get("Name") == "Sow using a variable rule":
                for p in node.get("Parameters", []):
                    try:
                        if p.get("Key") == "MinRain":
                            p["Value"] = "0"
                            modified = True
                        if p.get("Key") == "MinESW":
                            p["Value"] = "0"
                            modified = True
                        if p.get("Key") == "StartDate":
                            p["Value"] = sow_date_str
                            modified = True
                        if p.get("Key") == "EndDate":
                            p["Value"] = sow_date_str
                            modified = True
                    except (ValueError, TypeError):
                        pass
                code_array = node.get("CodeArray", [])
                code_text = " ".join(code_array)
                if "DateUtilities.DatesEqual(EndDate, Clock.Today)" not in code_text:
                    new_code = []
                    for line in code_array:
                        if "accumulatedRain.Sum > MinRain)" in line or "accumulatedRain.Sum >= MinRain)" in line:
                            new_code.append("                ((accumulatedRain.Sum >= MinRain && MathUtilities.Sum(waterBalance.ESW) >= MinESW) || DateUtilities.DatesEqual(EndDate, Clock.Today)))")
                            modified = True
                        elif "MathUtilities.Sum(waterBalance.ESW)" in line and "EndDate" not in line:
                            pass
                        else:
                            new_code.append(line)
                    node["CodeArray"] = new_code
            for v in node.values():
                update(v)
        elif isinstance(node, list):
            for x in node:
                update(x)
    update(ctrl.data)
    if modified:
        ctrl._save()


def run_pop_season(cfg: dict, location: str, year: int, met_file: str, logger: logging.Logger) -> dict:
    """Runs a single PoP baseline season in APSIM as-is and calculates yield and rewards."""
    trial_dir = os.path.join(cfg["SCRATCH_DIR"], f"pop_{location}_{year}")
    working_file = os.path.join(trial_dir, f"{location}_{year}_pop.apsimx")
    os.makedirs(trial_dir, exist_ok=True)

    try:
        ctrl = ApsimController(cfg["APSIM_BASE_FILE"], working_file)
        set_clock_and_weather_compat(ctrl, year, met_file,
                                     start_date=f"{year}-05-01T00:00:00",
                                     end_date=f"{year}-12-31T00:00:00")
        ctrl.set_location_soil_profile(location, cfg["SOIL_PROFILES"])
        if cfg.get("RELAX_SOWING", True):
            sow_date_str = calculate_sowing_date(met_file, year)
            logger.info(f"Calculated custom sowing date: {sow_date_str}")
            force_sowing_date_in_controller(ctrl, sow_date_str)

        raw_diag = ctrl.run_and_get_yield()
        diag = _to_plain_python(raw_diag)
        yield_val = float(diag.get("yield_kg_ha", 0.0))
        status = diag.get("status", "failed")

        materials, irrigations = _pop_materials_and_irrigations(cfg)
        prices, shadow = _prices_and_shadow(cfg)
        indicators = rm.SustainabilityIndicators(
            I_wl=float(diag.get("waterlogging_index", 0.0)),
            I_no=float(diag.get("n_oversupply_index", 0.0)),
            I_dd=float(diag.get("total_drainage_mm", 0.0)),
            I_nl=float(diag.get("total_leach_no3_nh4", 0.0)),
            I_cl=float(diag.get("soc_loss_index", 0.0)),
        )
        reward_breakdown = rm.compute_reward(
            yield_val, materials, irrigations, indicators, prices, shadow,
            crop_failure_penalty=cfg["CROP_FAILURE_PENALTY"],
        )

        return {
            "location": location,
            "year": year,
            "status": status,
            "yield_kg_ha": yield_val,
            "reward": float(reward_breakdown.get("reward", 0.0)),
            "reward_breakdown": _to_plain_python(reward_breakdown),
            "diagnostics": {k: v for k, v in diag.items() if k not in ("weekly_stress",)},
            "weather_file": met_file,
            "cultivar": cfg.get("CULTIVAR", "MAH-14-138"),
            "recipe": {
                "n_total_kg": cfg["POP_N_TOTAL_KG"],
                "n_splits": cfg["POP_N_SPLITS"],
                "irrig_events": cfg["POP_IRRIG_EVENTS"],
                "irrig_mm_each": cfg["POP_IRRIG_MM_EACH"],
            },
        }
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def parse_years(year_args: List[str]) -> List[int]:
    """Parses year arguments like ['2015', '2016'] or ['2015-2025']."""
    years = []
    for arg in year_args:
        if "-" in arg:
            start, end = map(int, arg.split("-"))
            years.extend(range(start, end + 1))
        else:
            years.append(int(arg))
    return sorted(list(set(years)))


def main():
    parser = argparse.ArgumentParser(description="Run Package-of-Practices (PoP) baseline across years.")
    parser.add_argument("--location", default=None, help="Location name (default: inferred from base file or Mandya)")
    parser.add_argument("--base-file", default=None,
                        help="Path to base .apsimx file (default: inferred from location or Maize_{location}_pop2.apsimx / Maize_{location}_pop.apsimx)")
    parser.add_argument("--years", nargs="+", default=None,
                        help="Years to run, e.g. --years 2015 2016 or --years 2015-2025 (default: 2015-2025)")
    parser.add_argument("--weather", choices=["baseline", "observed"], default="observed",
                        help="Weather series: 'observed' (raw pilot met) or 'baseline' (spliced weather array). Default: observed")
    parser.add_argument("--out", default=None, help="Custom output JSON path")
    parser.add_argument("--no-relax-sowing", dest="relax_sowing", action="store_false", default=True,
                        help="Disable automatic relaxation of sowing trigger (default: relax enabled with EndDate fallback)")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["RELAX_SOWING"] = args.relax_sowing

    if args.base_file:
        cfg["APSIM_BASE_FILE"] = os.path.abspath(args.base_file)
        if not os.path.exists(cfg["APSIM_BASE_FILE"]):
            raise FileNotFoundError(f"Base APSIM file not found: {cfg['APSIM_BASE_FILE']}")
        if args.location:
            location = args.location
        else:
            base_lower = os.path.basename(cfg["APSIM_BASE_FILE"]).lower()
            for loc in ["haveri", "mandya", "dharwad", "davanagere"]:
                if loc in base_lower:
                    location = loc.capitalize()
                    break
            else:
                location = cfg["LOCATION"]
    else:
        location = args.location or cfg["LOCATION"]
        pop2_candidate = os.path.join(ROOT, "apsimx", f"Maize_{location}_pop2.apsimx")
        pop_candidate = os.path.join(ROOT, "apsimx", f"Maize_{location}_pop.apsimx")
        if os.path.exists(pop2_candidate):
            cfg["APSIM_BASE_FILE"] = pop2_candidate
        elif os.path.exists(pop_candidate):
            cfg["APSIM_BASE_FILE"] = pop_candidate

    recipe_info = inspect_apsimx_recipe(cfg["APSIM_BASE_FILE"])
    cfg["POP_N_TOTAL_KG"] = recipe_info["fert_total_kg"]
    cfg["POP_N_SPLITS"] = recipe_info["fert_splits"]
    cfg["POP_IRRIG_EVENTS"] = recipe_info["irrig_events"]
    cfg["POP_IRRIG_MM_EACH"] = recipe_info["irrig_mm_each"]
    cfg["CULTIVAR"] = recipe_info["cultivar"]

    years = parse_years(args.years) if args.years else cfg["YEARS"]

    os.makedirs(cfg["SCRATCH_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    os.makedirs(cfg["LOG_DIR"], exist_ok=True)

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_logger = get_logger(f"pop_baseline_{location}_{run_tag}",
                                run_tag=f"{run_tag}_{location}", log_dir=cfg["LOG_DIR"])

    section(overall_logger, f"RUNNING POP BASELINE: {location} ({len(years)} seasons, weather={args.weather})")
    overall_logger.info(f"Base file: {cfg['APSIM_BASE_FILE']}")
    overall_logger.info(f"Cultivar: {cfg['CULTIVAR']} | Recipe: N={cfg['POP_N_TOTAL_KG']}kg ({cfg['POP_N_SPLITS']} splits), "
                        f"Irrig={cfg['POP_IRRIG_EVENTS']} events x {cfg['POP_IRRIG_MM_EACH']}mm")

    results = []
    for year in years:
        logger = get_logger(f"pop_baseline_{location}_{year}",
                            run_tag=f"{run_tag}_{location}_{year}", log_dir=cfg["LOG_DIR"])
        apsim_log = logging.getLogger("agri_sigra")
        apsim_log.handlers = logger.handlers
        apsim_log.setLevel(logger.level)
        apsim_log.propagate = False

        try:
            met_file = resolve_weather_file(cfg, location, year, args.weather, logger)
            res = run_pop_season(cfg, location, year, met_file, logger)
            logger.info(f"{location} {year}: status={res['status']} | yield={res['yield_kg_ha']:.0f} kg/ha | "
                        f"reward=Rs.{res['reward']:.2f}")
        except Exception as e:
            logger.exception(f"Execution failed for {location} {year}: {e}")
            res = {
                "location": location,
                "year": year,
                "status": "failed",
                "yield_kg_ha": 0.0,
                "reward": 0.0,
                "error": str(e),
            }
        results.append(res)

    out_path = args.out or os.path.join(cfg["RESULTS_DIR"], f"pop_baseline_{location}_{run_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"PACKAGE OF PRACTICES (PoP) BASELINE RESULTS: {location} (Weather: {args.weather})")
    print(f"Cultivar: {cfg['CULTIVAR']} | Base File: {os.path.basename(cfg['APSIM_BASE_FILE'])}")
    print(f"Recipe: N={cfg['POP_N_TOTAL_KG']:.0f} kg ({cfg['POP_N_SPLITS']} splits), Irrig={cfg['POP_IRRIG_EVENTS']} events x {cfg['POP_IRRIG_MM_EACH']:.0f} mm")
    print("=" * 78)
    print(f"{'Year':<8} | {'Status':<10} | {'Yield (kg/ha)':<15} | {'Reward (Rs/ha)':<16} | {'Drainage (mm)':<14}")
    print("-" * 78)

    yields = []
    rewards = []
    for r in results:
        status = r.get("status", "failed")
        y = r.get("yield_kg_ha", 0.0)
        rew = r.get("reward", 0.0)
        drainage = r.get("diagnostics", {}).get("total_drainage_mm", 0.0)

        if status == "success":
            yields.append(y)
            rewards.append(rew)
            print(f"{r['year']:<8} | {status:<10} | {y:<15.0f} | Rs. {rew:<12.2f} | {drainage:<14.1f}")
        else:
            print(f"{r['year']:<8} | {status:<10} | {'0 (failed)':<15} | Rs. {rew:<12.2f} | {'N/A':<14}")

    print("-" * 78)
    if yields:
        avg_y = sum(yields) / len(yields)
        avg_rew = sum(rewards) / len(rewards)
        print(f"Summary ({len(yields)}/{len(results)} successful seasons):")
        print(f"  Mean Yield : {avg_y:.1f} kg/ha (Min: {min(yields):.0f}, Max: {max(yields):.0f})")
        print(f"  Mean Reward: Rs. {avg_rew:.2f}")
    else:
        print("No seasons completed successfully.")
    print(f"Detailed results saved to: {out_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
