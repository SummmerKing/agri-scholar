"""Reward of Section 3.3: R = revenue - production cost - sustainability penalties (Eqs. 2-14)."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Prices:
    """p_g, p_i^F (per fertilizer/material type), p_w, p_L from the methodology."""
    grain_price_per_kg: float                 # p_g
    fertilizer_price_per_kg: Dict[str, float]  # p_i^F, keyed by material name (e.g. "Urea", "DAP", "Mulch")
    water_price_per_mm: float                  # p_w
    labor_price_per_operation: float           # p_L


@dataclass
class ShadowCosts:
    """S_wl, S_no, S_dd, S_nl, S_cl -- monetary shadow costs for each sustainability indicator."""
    waterlogging: float       # S_wl  (Rs per seasonal saturation-ratio unit)
    n_oversupply: float       # S_no  (Rs per seasonal excess-N-ratio unit)
    deep_drainage: float      # S_dd  (Rs per mm cumulative drainage)
    n_leaching: float         # S_nl  (Rs per kg N leached)
    soc_loss: float           # S_cl  (Rs per % SOC lost)


@dataclass
class MaterialApplication:
    """One discrete fertilizer/material application event -> one C_F term."""
    material: str       # key into Prices.fertilizer_price_per_kg
    amount_kg: float


@dataclass
class IrrigationEvent:
    """One discrete irrigation application event -> one C_I term."""
    amount_mm: float


@dataclass
class SustainabilityIndicators:
    """APSIM-derived indicators I_wl, I_no, I_dd, I_nl, I_cl (Eqs. 9-14)."""
    I_wl: float = 0.0   # waterlogging index                (Eq. 9)
    I_no: float = 0.0   # nitrogen oversupply index          (Eq. 10)
    I_dd: float = 0.0   # cumulative deep drainage, mm        (Eq. 11)
    I_nl: float = 0.0   # cumulative N leaching, kg/ha         (Eq. 12)
    I_cl: float = 0.0   # soil organic carbon loss, %            (Eq. 13)


def compute_production_cost(
    materials: List[MaterialApplication],
    irrigations: List[IrrigationEvent],
    prices: Prices,
) -> Dict[str, float]:
    """Implements Eq. 4: C_prod = C_F + C_I + C_L."""
    c_f = sum(m.amount_kg * prices.fertilizer_price_per_kg.get(m.material, 0.0) for m in materials)
    c_i = sum(irr.amount_mm * prices.water_price_per_mm for irr in irrigations)
    n_f, n_i = len(materials), len(irrigations)
    c_l = (n_f + n_i) * prices.labor_price_per_operation
    return {"C_F": c_f, "C_I": c_i, "C_L": c_l, "C_prod": c_f + c_i + c_l, "n_F": n_f, "n_I": n_i}


def compute_sustainability_penalty(indicators: SustainabilityIndicators, shadow: ShadowCosts) -> Dict[str, float]:
    """Implements Eq. 5-7: P_sust = P_season + P_post."""
    p_season = shadow.waterlogging * indicators.I_wl + shadow.n_oversupply * indicators.I_no
    p_post = (shadow.deep_drainage * indicators.I_dd
              + shadow.n_leaching * indicators.I_nl
              + shadow.soc_loss * indicators.I_cl)
    return {"P_season": p_season, "P_post": p_post, "P_sust": p_season + p_post}


def compute_reward(
    yield_kg_ha: float,
    materials: List[MaterialApplication],
    irrigations: List[IrrigationEvent],
    indicators: SustainabilityIndicators,
    prices: Prices,
    shadow: ShadowCosts,
    crop_failure_penalty: float = 50000.0,
) -> Dict[str, float]:
    """Eq. 2: R = R_rev - C_prod - P_sust (per hectare), with the full breakdown."""
    if yield_kg_ha <= 0:
        return {
            "yield_kg_ha": 0.0, "R_rev": 0.0, "C_prod": 0.0, "P_sust": 0.0,
            "reward": -crop_failure_penalty, "crop_failure": True,
        }

    r_rev = yield_kg_ha * prices.grain_price_per_kg           # Eq. 3
    cost = compute_production_cost(materials, irrigations, prices)  # Eq. 4
    penalty = compute_sustainability_penalty(indicators, shadow)     # Eq. 5-7
    reward = r_rev - cost["C_prod"] - penalty["P_sust"]              # Eq. 2

    return {
        "yield_kg_ha": yield_kg_ha,
        "R_rev": r_rev,
        **cost,
        **penalty,
        "reward": reward,
        "crop_failure": False,
    }
