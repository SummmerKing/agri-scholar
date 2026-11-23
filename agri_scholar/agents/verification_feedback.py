"""Verification and Feedback Agent (VFA), Section 3.4.

score_candidate runs a candidate through APSIM and returns the reward to the OA. build_critique writes
a short natural-language critique that the ABA uses to tighten or relax its bounds.
"""
import logging
from typing import List, Optional

from agri_scholar.simulation.engine import SimulationEngine, SimulationResult

log = logging.getLogger("agri_sigra")


class VerificationFeedbackAgent:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    # inner loop: OA -> VFA -> APSIM -> reward
    def score_candidate(self, action: dict, met_file: str) -> SimulationResult:
        return self.engine.run(action, met_file=met_file)

    # middle loop: VFA critique -> ABA
    def build_critique(self, result: SimulationResult, previous_yield: float = 0.0) -> str:
        diag = result.diagnostics
        rb = result.reward_breakdown

        if result.crop_failure:
            return ("CROP FAILURE: APSIM returned zero yield for this candidate action -- the crop never "
                    "established or died during the season. Widen sowing-window and irrigation bounds; "
                    "the previous bounds may have starved the crop of water or nitrogen at establishment.")

        lines = [
            f"Yield: {result.yield_kg_ha:.0f} kg/ha. Reward (Eq. 2, R = R_rev - C_prod - P_sust): "
            f"Rs.{rb['reward']:.2f} (Revenue Rs.{rb['R_rev']:.2f}, Production cost Rs.{rb['C_prod']:.2f}, "
            f"Sustainability penalty Rs.{rb['P_sust']:.2f}).",
            f"Sown on: {result.sow_date_str}.",
        ]

        repro_w = diag.get("repro_water_stress", 0.0)
        repro_n = diag.get("repro_n_stress", 0.0)
        veg_w = diag.get("veg_water_stress", 0.0)
        veg_n = diag.get("veg_n_stress", 0.0)
        lines.append(f"[STRESS] Water: {veg_w:.1f} vegetative-days + {repro_w:.1f} REPRODUCTIVE-days. "
                     f"Nitrogen: {veg_n:.1f} vegetative-days + {repro_n:.1f} REPRODUCTIVE-days.")

        if repro_w > 3.0:
            lines.append("CRITICAL: severe water stress during flowering/grain-fill -- this directly reduces "
                          "grain set. Increase irrigation bounds around weeks 8-12 or shift the sowing window "
                          "to align silking with peak monsoon rainfall.")
        if repro_n > 3.0:
            lines.append("CRITICAL: nitrogen deficiency during the reproductive phase -- ensure a urea "
                          "top-dress bound is available before tasseling (around week 7-8).")

        hi = diag.get("harvest_index", 0.0)
        lines.append(f"Harvest Index: {hi:.3f} (agronomic target >= 0.45).")
        if 0 < hi < 0.30:
            lines.append("WARNING: very low harvest index -- the crop grew vegetatively but failed to convert "
                          "biomass into grain. Often indicates late sowing or reproductive-stage stress.")

        drainage = diag.get("total_drainage_mm", 0.0)
        leaching = diag.get("total_leach_no3_nh4", 0.0)
        env_msg = f"Environmental: {drainage:.0f}mm cumulative drainage, {leaching:.1f}kg/ha NO3 leached."
        if drainage > 50 or leaching > 15.0:
            env_msg += (
                " Suggest an extra split or foliar nitrogen application after heavy percolation events to compensate for leached NO3. "
                "Also, consider narrowing the upper irrigation bound or withholding scheduled irrigation if rainfall already exceeds 200 mm/week."
            )
        lines.append(env_msg)

        if previous_yield and result.yield_kg_ha > previous_yield:
            lines.append(f"Improved by {result.yield_kg_ha - previous_yield:.0f} kg/ha over the previous "
                         "reflection iteration.")
        elif previous_yield and result.yield_kg_ha <= previous_yield:
            lines.append("No improvement over the previous reflection iteration -- reconsider whether the "
                         "bounds are actually binding (i.e. whether the optimizer is choosing values at the "
                         "edge of the range you gave it).")

        return "\n".join(lines)

    def weekly_stress_hotspots(self, result: SimulationResult) -> str:
        weekly = result.diagnostics.get("weekly_stress", {})
        hotspots = []
        for key, vals in sorted(weekly.items(), key=lambda kv: int(kv[0].split("_")[1])):
            if vals.get("water_stress", 0) > 0.1 or vals.get("n_stress", 0) > 0.1:
                hotspots.append(f"{key}(W={vals['water_stress']:.1f},N={vals['n_stress']:.1f})")
        return ", ".join(hotspots) if hotspots else "None detected."
