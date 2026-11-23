"""Haveri reruns (2017, 2019, 2021, 2022). Same pipeline as run.py."""
import os
import gc
import json
import copy
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from logging_utils import get_logger, section
import weather_array as wa
import reward_model as rm
from context_retrieval_agent import ContextRetrievalAgent
from llm_backend import LocalLLM
from preseason_agent import PreseasonAgent
from advisory_bounding_agent import AdvisoryBoundingAgent
from optimization_agent import OptimizationAgent
from verification_feedback_agent import VerificationFeedbackAgent
from simulation_engine import SimulationEngine, ActionCaps
from seeding import derive_seed

CONFIG = {
    "LOCATIONS": ["Haveri"],
    "YEARS": [2017, 2019, 2021, 2022],
    "TOTAL_WEEKS": 18,
    "CULTIVAR": "MAH-14-138",
    "POPULATION_PLANTS_M2": 8.5,
    "SEED": 42,

    "GPU_IDS": [5, 6, 7],
    "INSTANCES_PER_GPU": 1,

    "APSIM_BASE_FILE": os.path.join(os.path.dirname(__file__), "apsimx", "Maize_Haveri_pop2.apsimx"),
    "MET_DIR": os.path.join(os.path.dirname(__file__), "data", "weather_maize_pilot"),
    "RAG_CORPUS_PATH": os.path.join(os.path.dirname(__file__), "rag_corpus.json"),
    "SCRATCH_DIR": os.path.join(os.path.dirname(__file__), "scratch_sims"),
    "RESULTS_DIR": os.path.join(os.path.dirname(__file__), "results", "haveri_rerun"),
    "LOG_DIR": os.path.join(os.path.dirname(__file__), "logs", "haveri_rerun"),

    "ABA_MODEL_ID": "google/gemma-4-E4B-it",
    "ABA_GPU_ID": 0,
    "ABA_MAX_NEW_TOKENS": 1500,
    "ABA_TEMPERATURE": 0.5,

    "RAG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RAG_ALPHA": 0.5,
    "RAG_TOP_K": 3,
    "RAG_LOCATION_BOOST": 0.25,
    "RAG_REGION_BOOST": 0.10,
    "RAG_MAX_CHUNKS_PER_FILE": 2,

    "N_PRESEASON_REFLECTION_ITERATIONS": 2,
    "N_TRIALS_PRESEASON": 100,   # mulch / DAP / tillage search per PSA iteration (sowing date is locked first)

    "SOW_WINDOW_START": "06-01",          # MM-DD, first allowed sowing day
    "SOW_WINDOW_END": "07-15",            # MM-DD, last allowed sowing day
    "SOW_SEARCH_STRIDE_DAYS": 1,
    "SOW_SEARCH_N_ANALOGS": 3,            # weather ensemble size (top-K analog years)
    "SOW_SEARCH_LAMBDA_RISK": 0.5,
    "SOW_SEARCH_SMOOTH_HALFWIDTH": 1,     # +/-1 candidate neighbours -> 3-point smoothing of the score curve
    "SOW_SEARCH_BASAL_DAP_KG": 60.0,      # neutral management used only while ranking sowing dates
    "SOW_SEARCH_TILLAGE_FRACTION": 0.5,
    "SOW_SEARCH_N_JOBS": 4,               # parallel APSIM runs during the sowing search

    "N_REFLECTION_ITERATIONS": 3,

    "N_TRIALS_LATER_WEEKS": 25,   # every in-season week (week 0 included) searches urea/irrigation only
    "OPTUNA_N_JOBS": 24,
    # receding horizon: plan urea for weeks w..T-1, commit week w
    "N_TRIALS_PER_HORIZON_WEEK": 4,
    "HORIZON_WEEKS": None,
    "FUTURE_UREA_STEP_KG": 5.0,     # coarse dose step for future (uncommitted) weeks
    "MAX_SINGLE_UREA_KG": None,     # per-split ceiling for future weeks; None = derived from season cap / splits

    "GRAIN_PRICE_PER_KG": 20.90,
    "FERTILIZER_PRICE_PER_KG": {
        "Urea": 13.04,      # pure elemental-N-equivalent price for UreaN applications
        "DAP": 27.00,
        "Mulch": 2.00,      # nominal per-kg cost of sourcing/spreading residue mulch
    },
    "WATER_PRICE_PER_MM": 2.50,
    "LABOR_PRICE_PER_OPERATION": 350.0,

    "SHADOW_COST_WATERLOGGING": 5000.0,
    "SHADOW_COST_N_OVERSUPPLY": 2000.0,
    "SHADOW_COST_DEEP_DRAINAGE": 15.0,
    "SHADOW_COST_N_LEACHING": 25.0,
    "SHADOW_COST_SOC_LOSS": 150.0,

    "CROP_FAILURE_PENALTY": 50000.0,

    "ANALOG_FEATURES_WEEK0": ["ONI", "Rain_Jan", "Rain_Feb", "Rain_Mar", "Rain_Apr", "Rain_May", "Tmax_Jan_May"],
    "ANALOG_WEIGHTS_WEEK0": [0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1],
    "ANALOG_FEATURES_LATER": ["ONI", "CumRain", "RainyDays", "MaxDrySpell", "Tmax_Mean"],
    "ANALOG_WEIGHTS_LATER": [0.2, 0.3, 0.2, 0.2, 0.1],

    # calibrated .apsimx per location
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
}


def sow_window_doys(year: int, start_md: str, end_md: str):
    """('06-01', '07-15') -> (DOY of 1 Jun, DOY of 15 Jul) for `year` (leap-year safe)."""
    def doy(md):
        m, d = (int(x) for x in md.split("-"))
        return datetime.date(year, m, d).timetuple().tm_yday
    return doy(start_md), doy(end_md)


def run_season(cfg: dict, location: str, year: int, logger) -> dict:
    met_path = os.path.join(cfg["MET_DIR"], f"{location}_{year}.met")
    season_seed = derive_seed(cfg["SEED"], location, year)

    section(logger, f"SEASON START: {location} {year} (seed={season_seed})")

    all_weather = wa.load_all_weather(cfg["MET_DIR"], location)
    oni = wa.get_enso_data()
    hist_features_full = wa.build_historical_feature_table(all_weather, oni)

    def narrative_as_of(day: int) -> str:
        """Rainfall narrative using only weather observed through `day` (no look-ahead)."""
        return wa.summarize_kharif_season(met_path, year, int(day))["narrative"]

    df = all_weather[year].sort_values("day").reset_index(drop=True)
    onset_doy = 166
    rain_window = []
    for _, row in df.iterrows():
        day, rain = int(row["day"]), float(row["rain"])
        if day >= 152:
            rain_window.append((day, rain))
            if len(rain_window) > 3:
                rain_window.pop(0)
            if len(rain_window) == 3 and sum(r for _, r in rain_window) >= 25.0 and rain >= 2.5:
                onset_doy = day
                break

    win_start = onset_doy
    win_end = onset_doy + 15

    logger.info(f"Sowing window DOY {win_start}..{win_end}")

    prices = rm.Prices(
        grain_price_per_kg=cfg["GRAIN_PRICE_PER_KG"],
        fertilizer_price_per_kg=cfg["FERTILIZER_PRICE_PER_KG"],
        water_price_per_mm=cfg["WATER_PRICE_PER_MM"],
        labor_price_per_operation=cfg["LABOR_PRICE_PER_OPERATION"],
    )
    shadow_costs = rm.ShadowCosts(
        waterlogging=cfg["SHADOW_COST_WATERLOGGING"], n_oversupply=cfg["SHADOW_COST_N_OVERSUPPLY"],
        deep_drainage=cfg["SHADOW_COST_DEEP_DRAINAGE"], n_leaching=cfg["SHADOW_COST_N_LEACHING"],
        soc_loss=cfg["SHADOW_COST_SOC_LOSS"],
    )

    # placeholder caps for the pre-season search, replaced by the PSA caps
    placeholder_caps = ActionCaps(max_urea_splits=cfg["TOTAL_WEEKS"], max_irrigation_events=cfg["TOTAL_WEEKS"],
                                  min_urea_kg=0.0, min_irrigation_mm=0.0)

    engine = SimulationEngine(
        location=location, year=year, apsim_base_file=cfg["APSIM_BASE_FILE"], default_met_file=met_path,
        sow_window_start_doy=win_start, sow_window_end_doy=win_end,
        total_weeks=cfg["TOTAL_WEEKS"], prices=prices, shadow_costs=shadow_costs,
        soil_profiles=cfg["SOIL_PROFILES"], action_caps=placeholder_caps, scratch_dir=cfg["SCRATCH_DIR"],
        cultivar=cfg["CULTIVAR"], population=cfg["POPULATION_PLANTS_M2"],
    )
    vfa = VerificationFeedbackAgent(engine)
    cra = ContextRetrievalAgent(cfg["RAG_CORPUS_PATH"], cfg["RAG_EMBED_MODEL"], cfg["RAG_ALPHA"], cfg["RAG_TOP_K"],
                                 location_boost=cfg["RAG_LOCATION_BOOST"],
                                 region_boost=cfg["RAG_REGION_BOOST"],
                                 max_chunks_per_file=cfg["RAG_MAX_CHUNKS_PER_FILE"])
    # one LLM shared by the PSA and the ABA
    llm = LocalLLM(cfg["ABA_MODEL_ID"], cfg["ABA_GPU_ID"], cfg["ABA_MAX_NEW_TOKENS"], cfg["ABA_TEMPERATURE"],
                   seed=season_seed)
    psa = PreseasonAgent(llm)
    aba = AdvisoryBoundingAgent(llm)
    oa = OptimizationAgent(
        vfa=vfa, total_weeks=cfg["TOTAL_WEEKS"], n_trials_preseason=cfg["N_TRIALS_PRESEASON"],
        n_trials_later=cfg["N_TRIALS_LATER_WEEKS"], n_jobs=cfg["OPTUNA_N_JOBS"], seed=season_seed,
        trials_per_horizon_week=cfg["N_TRIALS_PER_HORIZON_WEEK"], horizon_weeks=cfg["HORIZON_WEEKS"],
        future_dose_step=cfg["FUTURE_UREA_STEP_KG"], max_single_urea_kg=cfg["MAX_SINGLE_UREA_KG"],
    )

    try:
        locked_actions, weekly_log = {}, {}
        sow_doy = win_start
        previous_yield = 0.0
        rng = np.random.default_rng(season_seed)
        last_week_drainage = 0.0
        prev_cumulative_leaching = 0.0
        week_cache = {}   # week_idx -> (weather_file, context_bundle, analog_year)

        oni_val = float(hist_features_full.loc[hist_features_full.Year == year, "ONI"].values[0]) \
            if year in hist_features_full.Year.values else 0.0

        # CRA + weather array for an in-season week
        def build_week_inputs(week_idx: int):
            if week_idx in week_cache:
                return week_cache[week_idx]
            current_day = sow_doy + week_idx * 7
            if week_idx == 0:
                target_vector = wa.extract_preseason_features(met_path)
                target_vector["ONI"] = oni_val
                features, weights = cfg["ANALOG_FEATURES_WEEK0"], cfg["ANALOG_WEIGHTS_WEEK0"]
                hist_features = hist_features_full[hist_features_full.Year < year]
            else:
                ins = wa.extract_in_season_features(all_weather[year], sow_doy, current_day)
                target_vector = {"ONI": oni_val, **ins}
                features, weights = cfg["ANALOG_FEATURES_LATER"], cfg["ANALOG_WEIGHTS_LATER"]
                records = []
                for y in all_weather:
                    if y >= year:
                        continue
                    h_ins = wa.extract_in_season_features(all_weather[y], sow_doy, current_day)
                    h_oni = float(hist_features_full.loc[hist_features_full.Year == y, "ONI"].values[0]) \
                        if y in hist_features_full.Year.values else 0.0
                    records.append({"Year": y, "ONI": h_oni, **h_ins})
                hist_features = pd.DataFrame(records)

            w_t, analog_year_used = wa.build_weekly_weather_array(
                year, sow_doy, current_day, 366,
                all_weather, hist_features, target_vector, features, weights, rng=rng,
            )
            week_weather_file = os.path.join(cfg["SCRATCH_DIR"], f"weather_{location}_{year}_wk{week_idx}.met")
            wa.save_weather_to_met(w_t, met_path, week_weather_file)

            recent_days = all_weather[year][(all_weather[year].day > current_day - 7) & (all_weather[year].day <= current_day)]
            mean_radn = float(recent_days["radn"].mean()) if not recent_days.empty else 20.0
            total_recent_rain = float(recent_days["rain"].sum()) if not recent_days.empty else 0.0
            context_bundle = cra.build_context_bundle(
                location, year, week_idx, narrative_as_of(current_day),
                extra_query_terms=f"analog year {analog_year_used}",
                mean_radn=mean_radn, total_recent_rain=total_recent_rain,
                drainage_mm=last_week_drainage, is_overcast=bool(mean_radn < 12.0),
            )
            logger.info(f"CRA context bundle (week {week_idx}):\n{context_bundle}")
            week_cache[week_idx] = (week_weather_file, context_bundle, analog_year_used)
            return week_cache[week_idx]

        # pre-season inputs: observed weather to the day before the window, degraded forecast, analog tail
        def build_preseason_inputs():
            decision_day = win_start - 1
            target_vector = wa.extract_preseason_features(met_path)
            target_vector["ONI"] = oni_val
            hist_features = hist_features_full[hist_features_full.Year < year]
            members = wa.build_preseason_ensemble(
                year, win_start, 366, all_weather, hist_features, target_vector,
                cfg["ANALOG_FEATURES_WEEK0"], cfg["ANALOG_WEIGHTS_WEEK0"],
                cfg["SOW_SEARCH_N_ANALOGS"], rng=rng,
            )
            files, analogs = [], []
            for i, (w_t, ay) in enumerate(members):
                f = os.path.join(cfg["SCRATCH_DIR"], f"weather_{location}_{year}_pre{i}.met")
                wa.save_weather_to_met(w_t, met_path, f)
                files.append(f)
                analogs.append(ay)
            recent = all_weather[year][(all_weather[year].day > decision_day - 7) & (all_weather[year].day <= decision_day)]
            mean_radn = float(recent["radn"].mean()) if not recent.empty else 20.0
            ctx = cra.build_context_bundle(
                location, year, 0, narrative_as_of(decision_day),
                extra_query_terms=f"analog years {analogs}",
                mean_radn=mean_radn, total_recent_rain=float(recent["rain"].sum()) if not recent.empty else 0.0,
                drainage_mm=0.0, is_overcast=bool(mean_radn < 12.0),
            )
            logger.info(f"CRA context bundle (pre-season, analog years {analogs}):\n{ctx}")
            return files, ctx, analogs

        # pre-season: sowing search, then PSA <-> OA/VFA for land prep and season caps
        section(logger, f"PRE-SEASON STAGE | {location} {year}")
        pre_files, pre_context, pre_analogs = build_preseason_inputs()
        pre_weather_file = pre_files[0]

        pre_feedback, pre_trace = None, []
        best_pre = {"reward": -float("inf"), "action": None, "result": None, "bounds": None}
        for it in range(1, cfg["N_PRESEASON_REFLECTION_ITERATIONS"] + 1):
            logger.info(f"--- Pre-season reflection {it}/{cfg['N_PRESEASON_REFLECTION_ITERATIONS']} ---")
            psa_bounds = psa.propose(location, year, cfg["TOTAL_WEEKS"], pre_context,
                                     vfa_feedback=pre_feedback, reflection_trace=pre_trace)
            pre_action, pre_result, pre_reward = oa.search_preseason(psa_bounds, pre_weather_file, win_start, win_end)
            if pre_result is None:
                logger.warning("OA returned no valid pre-season candidate; skipping.")
                pre_trace.append({"iteration": it, "reward": None, "yield_kg_ha": None,
                                  "psa_bounds": psa_bounds, "best_action": None})
                continue
            pre_feedback = vfa.build_critique(pre_result)
            logger.info(f"VFA critique (pre-season, iteration {it}):\n{pre_feedback}")
            pre_trace.append({"iteration": it, "reward": pre_reward, "yield_kg_ha": pre_result.yield_kg_ha,
                              "psa_bounds": psa_bounds, "best_action": pre_action,
                              "vfa_critique": pre_feedback})
            if pre_reward > best_pre["reward"]:
                best_pre.update(reward=pre_reward, action=pre_action, result=pre_result, bounds=psa_bounds)

        if best_pre["result"] is None:
            logger.error("No pre-season candidate from any iteration -- using PSA fallback structure and "
                         "neutral land prep (no mulch, no basal DAP).")
            from preseason_agent import make_fallback
            best_pre["bounds"] = make_fallback()
            best_pre["action"] = {"sow_doy": win_start, "mulch_mass": 0.0, "basal_dap_kg": 0.0, "tillage_fraction": 0.5}

        sow_doy = best_pre["action"].get("sow_doy", win_start)
        locked_actions["sow_doy"] = sow_doy
        for key in ("mulch_mass", "basal_dap_kg", "tillage_fraction"):
            locked_actions[key] = best_pre["action"].get(key, 0.0)
        mg = best_pre["bounds"]["management_frequency_caps"]
        engine.caps = ActionCaps(
            max_urea_splits=int(mg["max_urea_splits"]),
            max_irrigation_events=int(mg["max_irrigation_events"]),
            min_urea_kg=float(mg["min_urea_dose_kg"]),
            min_irrigation_mm=float(mg["min_irrigation_dose_mm"]),
            season_max_urea_kg=float(mg["season_max_urea_kg"]),
            season_max_irrigation_mm=float(mg["season_max_irrigation_mm"]),
        )
        caps = engine.caps
        logger.info(f"PRE-SEASON LOCKED: sow DOY {sow_doy}, "
                    f"mulch={locked_actions['mulch_mass']}, DAP={locked_actions['basal_dap_kg']}, "
                    f"tillage={locked_actions['tillage_fraction']}\nSeason structure: {caps}")

        preseason_log = {"locked": dict(locked_actions), "sowing_search": None,
                         "analog_years": pre_analogs, "caps": mg, "bounds": best_pre["bounds"],
                         "reflection_trace": pre_trace,
                         "reward_breakdown": best_pre["result"].reward_breakdown if best_pre["result"] else None}

        # weekly loop
        for week_idx in range(cfg["TOTAL_WEEKS"]):
            section(logger, f"WEEK {week_idx} | {location} {year}")
            current_day = sow_doy + week_idx * 7
            week_weather_file, context_bundle, analog_year_used = build_week_inputs(week_idx)

            # ABA <-> OA/VFA reflection
            vfa_feedback = None
            best_of_week = {"reward": -float("inf"), "action": None, "result": None, "strategy": None}
            reflection_trace = []

            for reflection_iter in range(1, cfg["N_REFLECTION_ITERATIONS"] + 1):
                logger.info(f"--- Reflection iteration {reflection_iter}/{cfg['N_REFLECTION_ITERATIONS']} "
                            f"(week {week_idx}) ---")

                bounds = aba.propose_bounds(location, year, week_idx, cfg["TOTAL_WEEKS"],
                                            context_bundle, locked_actions, caps, vfa_feedback,
                                            reflection_trace=reflection_trace, weekly_log=weekly_log)

                candidate_action, candidate_result, best_reward = oa.search(
                    bounds, week_idx, locked_actions, week_weather_file, caps)

                if candidate_result is None:
                    logger.warning("OA returned no valid candidate this reflection iteration; skipping.")
                    reflection_trace.append({
                        "iteration": reflection_iter, "reward": None, "yield_kg_ha": None,
                        "aba_bounds": bounds, "vfa_critique": None,
                        "aba_used_fallback": bounds.get("strategy_name") == "Fallback",
                    })
                    continue

                vfa_feedback = vfa.build_critique(candidate_result, previous_yield=previous_yield)
                logger.info(f"VFA critique (week {week_idx}, reflection {reflection_iter}):\n{vfa_feedback}")

                reflection_trace.append({
                    "iteration": reflection_iter,
                    "reward": best_reward,
                    "yield_kg_ha": candidate_result.yield_kg_ha,
                    "aba_bounds": bounds,
                    "chosen_doses": {"urea": float(candidate_action.get(f"n_dose_{week_idx}", 0.0)),
                                     "irrig": float(candidate_action.get(f"w_dose_{week_idx}", 0.0))},
                    "vfa_critique": vfa_feedback,
                    "aba_used_fallback": bounds.get("strategy_name") == "Fallback",
                })

                if best_reward > best_of_week["reward"]:
                    best_of_week.update(reward=best_reward, action=candidate_action,
                                        result=candidate_result, strategy=bounds)

            if best_of_week["result"] is None:
                logger.error(f"Week {week_idx}: no candidate produced by any reflection iteration. "
                             "Issuing a zero-intervention advisory.")
                final_action = dict(locked_actions)
                final_action[f"n_dose_{week_idx}"] = 0.0
                final_action[f"w_dose_{week_idx}"] = 0.0
                best_result = None
            else:
                final_action = dict(best_of_week["action"])
                best_result = best_of_week["result"]
                if best_result.crop_failure:
                    logger.warning(f"Week {week_idx}: even the best candidate reported crop failure -- "
                                   "issuing a zero-intervention advisory for this week.")
                    final_action[f"n_dose_{week_idx}"] = 0.0
                    final_action[f"w_dose_{week_idx}"] = 0.0

            # issued action must satisfy the season caps
            final_action, gate_violations = caps.enforce(final_action, cfg["TOTAL_WEEKS"])
            if gate_violations:
                logger.error(f"Week {week_idx}: final advisory needed cap enforcement (this should not "
                             f"happen; check OA): {gate_violations}")

            locked_actions[f"n_dose_{week_idx}"] = final_action.get(f"n_dose_{week_idx}", 0.0)
            locked_actions[f"w_dose_{week_idx}"] = final_action.get(f"w_dose_{week_idx}", 0.0)
            if best_result is not None:
                previous_yield = best_result.yield_kg_ha
                last_week_drainage = best_result.diagnostics.get("total_drainage_mm", 0.0) if best_result.diagnostics else 0.0

                # top-up after heavy leaching
                current_leaching = best_result.diagnostics.get("total_leach_no3_nh4", 0.0) if best_result.diagnostics else 0.0
                if week_idx < 10:
                    delta_leaching = current_leaching - prev_cumulative_leaching
                    if delta_leaching >= 15.0:
                        extra_splits = int(delta_leaching // 15.0)
                        caps.max_urea_splits += extra_splits
                        caps.season_max_urea_kg += (20.0 * extra_splits)
                        logger.warning(f"URGENT TOP-UP: {delta_leaching:.1f}kg NO3 leached in week {week_idx}. "
                                       f"Adding {extra_splits} extra splits and {20.0 * extra_splits}kg Urea to budget!")
                prev_cumulative_leaching = current_leaching
            else:
                last_week_drainage = 0.0

            weekly_log[f"week_{week_idx}"] = {
                "doy": current_day,
                "analog_year": analog_year_used,
                "context_bundle": context_bundle,
                "issued_advisory": {
                    "urea_kg": locked_actions[f"n_dose_{week_idx}"],
                    "irrigation_mm": locked_actions[f"w_dose_{week_idx}"],
                },
                "season_budget_after": caps.remaining(locked_actions, week_idx + 1),
                "tentative_urea_plan": ({w: float(best_of_week["action"].get(f"n_dose_{w}", 0.0))
                                         for w in range(week_idx + 1, cfg["TOTAL_WEEKS"])
                                         if best_of_week["action"].get(f"n_dose_{w}", 0.0) > 0}
                                        if best_of_week["action"] else {}),
                "best_strategy": best_of_week["strategy"],
                "vfa_critique": vfa_feedback,
                "reward_breakdown": best_result.reward_breakdown if best_result else None,
                "diagnostics": best_result.diagnostics if best_result else None,
                "reflection_trace": reflection_trace,
            }
            logger.info(f"Week {week_idx} advisory locked: "
                        f"Urea={locked_actions[f'n_dose_{week_idx}']:.1f}kg, "
                        f"Irrigation={locked_actions[f'w_dose_{week_idx}']:.1f}mm")

        # end of season: rerun on observed weather and check the caps
        section(logger, f"END-OF-SEASON VERIFICATION: {location} {year}")
        final_result = engine.run(locked_actions, met_file=met_path)
        logger.info(f"Final realized yield: {final_result.yield_kg_ha:.0f} kg/ha | "
                    f"Final reward: Rs.{final_result.reward:.2f}")

        _, audit_violations = caps.enforce(locked_actions, cfg["TOTAL_WEEKS"])
        usage = caps.usage(locked_actions, range(cfg["TOTAL_WEEKS"]))
        logger.info(f"CAP AUDIT: urea {usage['urea_events']}/{caps.max_urea_splits} splits, "
                    f"{usage['urea_kg']:.1f}/{caps.season_max_urea_kg} kg | irrigation "
                    f"{usage['irrig_events']}/{caps.max_irrigation_events} events, "
                    f"{usage['irrig_mm']:.1f}/{caps.season_max_irrigation_mm} mm | "
                    f"violations={audit_violations or 'none'}")

        total_aba_calls = cfg["TOTAL_WEEKS"] * cfg["N_REFLECTION_ITERATIONS"]
        logger.info(
            f"ABA fallback summary: {aba.fallback_count}/{total_aba_calls} calls used fallback bounds "
            f"({100 * aba.fallback_count / max(total_aba_calls, 1):.1f}%). "
            f"PSA fallbacks: {psa.fallback_count}/{cfg['N_PRESEASON_REFLECTION_ITERATIONS']}.")

        return {
            "location": location, "year": year,
            "sow_window_doy": [win_start, win_end], "sow_doy": sow_doy,
            "preseason": preseason_log,
            "locked_actions": locked_actions,
            "cap_audit": {"usage": usage, "violations": audit_violations,
                          "caps": {"max_urea_splits": caps.max_urea_splits,
                                   "max_irrigation_events": caps.max_irrigation_events,
                                   "min_urea_kg": caps.min_urea_kg, "min_irrigation_mm": caps.min_irrigation_mm,
                                   "season_max_urea_kg": caps.season_max_urea_kg,
                                   "season_max_irrigation_mm": caps.season_max_irrigation_mm}},
            "final_yield_kg_ha": final_result.yield_kg_ha,
            "final_reward_breakdown": final_result.reward_breakdown,
            "final_diagnostics": final_result.diagnostics,
            "weekly_log": weekly_log,
        }
    finally:
        del aba, psa, llm, oa, vfa, engine
        gc.collect()
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# one (location, year) season per worker, pinned to a GPU slot
def _run_one_job(job: dict) -> dict:
    cfg = job["cfg"]
    location = job["location"]
    year = job["year"]
    gpu_id = job["gpu_id"]
    run_tag = job["run_tag"]

    cfg = copy.deepcopy(cfg)
    cfg["LOCATION"] = location
    cfg["ABA_GPU_ID"] = gpu_id

    os.makedirs(cfg["SCRATCH_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)

    logger_name = f"agri_sigra_{location}_{year}"
    job_run_tag = f"{run_tag}_{location}_{year}"
    logger = get_logger(logger_name, run_tag=job_run_tag, log_dir=cfg["LOG_DIR"])
    log_path = logger.handlers[0].baseFilename if logger.handlers else None

    result = {"location": location, "year": year, "gpu_id": gpu_id, "log_path": log_path,
              "status": "failed", "season_result": None}

    logger.info(f"[GPU {gpu_id}] Starting season {location} {year}")
    logger.info(f"CONFIG:\n{json.dumps({k: v for k, v in cfg.items() if k != 'SOIL_PROFILES'}, indent=2, default=str)}")
    try:
        season_result = run_season(cfg, location, year, logger)
        result["season_result"] = season_result
        result["status"] = "ok"
    except Exception:
        logger.exception(f"Season {location} {year} failed with an unhandled exception.")
    finally:
        gc.collect()
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # per-job checkpoint
    out_path = os.path.join(cfg["RESULTS_DIR"], f"agri_sigra_{location}_{year}_{run_tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {out_path}")

    return result


def main():
    cfg = CONFIG
    import sys
    if "--years" in sys.argv:
        idx = sys.argv.index("--years")
        cfg["YEARS"] = [int(y) for y in sys.argv[idx+1:]]

    os.makedirs(cfg["SCRATCH_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs_spec = [(location, year) for location in cfg["LOCATIONS"] for year in cfg["YEARS"]]

    gpu_ids = cfg["GPU_IDS"]
    instances_per_gpu = cfg["INSTANCES_PER_GPU"]

    jobs = [
        {"cfg": cfg, "location": location, "year": year,
         "gpu_id": gpu_ids[i % len(gpu_ids)],
         "run_tag": run_tag}
        for i, (location, year) in enumerate(jobs_spec)
    ]

    print(f"Running {len(jobs)} season(s) in parallel on GPUs {gpu_ids}:")
    for j in jobs:
        print(f"  {j['location']} {j['year']} -> GPU {j['gpu_id']}")

    all_results = []

    max_workers = len(gpu_ids) * instances_per_gpu
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        futures = {executor.submit(_run_one_job, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Job {job['location']} {job['year']} (GPU {job['gpu_id']}) "
                      f"raised an exception: {exc}")
                result = {"location": job["location"], "year": job["year"],
                          "gpu_id": job["gpu_id"], "status": "failed", "season_result": None}

            all_results.append(result)

            combined_path = os.path.join(cfg["RESULTS_DIR"], f"agri_sigra_combined_{run_tag}.json")
            with open(combined_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"[{len(all_results)}/{len(jobs)}] {job['location']} {job['year']} "
                  f"(GPU {job['gpu_id']}) -> {result['status']} | checkpoint: {combined_path}")

    print("\nRUN COMPLETE")
    for r in all_results:
        if r["status"] == "ok" and r["season_result"] is not None:
            sr = r["season_result"]
            print(f"{r['location']} {r['year']} (GPU {r['gpu_id']}): "
                  f"yield={sr['final_yield_kg_ha']:.0f} kg/ha, "
                  f"reward=Rs.{sr['final_reward_breakdown']['reward']:.2f} | log: {r.get('log_path')}")
        else:
            print(f"{r['location']} {r['year']} (GPU {r['gpu_id']}): FAILED | log: {r.get('log_path')}")


if __name__ == "__main__":
    main()
