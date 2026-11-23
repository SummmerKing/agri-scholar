"""Checks that run without a GPU, APSIM or the LLM.

    python -m pytest tests -q
"""
import json
import os

import pytest

from agri_scholar.config import DEFAULTS, ROOT, load_config
from agri_scholar.simulation import reward as rm
from agri_scholar.simulation import weather as wa
from agri_scholar.simulation.engine import ActionCaps

pipeline = pytest.importorskip("agri_scholar.pipeline", reason="needs torch, transformers, optuna")

PRICES = rm.Prices(grain_price_per_kg=20.0, fertilizer_price_per_kg={"Urea": 10.0, "DAP": 30.0},
                   water_price_per_mm=2.0, labor_price_per_operation=100.0)
SHADOW = rm.ShadowCosts(waterlogging=10.0, n_oversupply=20.0, deep_drainage=1.0, n_leaching=2.0, soc_loss=3.0)


def test_reward_decomposition():
    r = rm.compute_reward(
        5000.0,
        [rm.MaterialApplication("Urea", 50.0), rm.MaterialApplication("DAP", 100.0)],
        [rm.IrrigationEvent(40.0)],
        rm.SustainabilityIndicators(I_wl=1.0, I_no=0.5, I_dd=10.0, I_nl=5.0, I_cl=1.0),
        PRICES, SHADOW,
    )
    assert r["R_rev"] == 100000.0
    assert r["C_prod"] == 500.0 + 3000.0 + 80.0 + 3 * 100.0
    assert r["P_sust"] == 10.0 + 10.0 + 10.0 + 10.0 + 3.0
    assert r["reward"] == r["R_rev"] - r["C_prod"] - r["P_sust"]


def test_crop_failure_penalty():
    r = rm.compute_reward(0.0, [], [], rm.SustainabilityIndicators(), PRICES, SHADOW, crop_failure_penalty=123.0)
    assert r["crop_failure"] and r["reward"] == -123.0


def test_caps_enforce_split_and_season_limits():
    caps = ActionCaps(max_urea_splits=2, max_irrigation_events=1, min_urea_kg=20.0, min_irrigation_mm=10.0,
                      season_max_urea_kg=100.0, season_max_irrigation_mm=50.0)
    action = {"n_dose_0": 60.0, "n_dose_1": 60.0, "n_dose_2": 30.0, "w_dose_0": 40.0, "w_dose_3": 40.0}
    clean, violations = caps.enforce(action, total_weeks=6)
    used = caps.usage(clean, range(6))
    assert used["urea_events"] <= 2 and used["urea_kg"] <= 100.0
    assert used["irrig_events"] <= 1 and used["irrig_mm"] <= 50.0
    assert violations


@pytest.mark.parametrize("name", ["main", "mandya", "haveri", "haveri_rerun",
                                  "ablation_fixed_bounds", "ablation_pop_bounds"])
def test_experiment_configs_load(name):
    path = os.path.join(ROOT, "configs", f"{name}.json")
    with open(path) as f:
        unknown = set(json.load(f)) - set(DEFAULTS)
    assert not unknown, unknown
    cfg = load_config(path)
    assert cfg["BOUNDS"] in ("llm", "fixed", "pop")
    assert all(os.path.exists(p) for p in cfg["APSIM_BASE_FILES"].values())


def test_sowing_window_modes():
    weather = wa.load_all_weather(os.path.join(ROOT, "data", "weather_maize_pilot"), "Mandya")[2020]
    pop = pipeline.sowing_window(load_config(SOWING="pop_date"), "Mandya", 2020, weather)
    assert pop == (182, 182)
    lo, hi = pipeline.sowing_window(load_config(SOWING="onset"), "Mandya", 2020, weather)
    assert hi - lo == 15 and lo >= 152
    assert pipeline.sowing_window(load_config(SOWING="window"), "Mandya", 2020, weather) == (153, 197)


def test_pop_bounds_one_urea_dose_per_window():
    from agri_scholar.agents.fixed_bounds import PopBounds
    agent = PopBounds(total_weeks=18)
    caps = ActionCaps(max_urea_splits=3, max_irrigation_events=6, min_urea_kg=25.0, min_irrigation_mm=20.0,
                      season_max_urea_kg=225.0, season_max_irrigation_mm=360.0)
    week0 = agent.propose_bounds("Mandya", 2020, 0, 18, "", {}, caps, None, [], {})["in_season_weeks"][0]
    week1 = agent.propose_bounds("Mandya", 2020, 1, 18, "", {"n_dose_0": 50.0}, caps, None, [], {})["in_season_weeks"][0]
    assert (week0["urea_min"], week0["urea_max"]) == (25.0, 75.0)
    assert week1["urea_max"] == 0.0
