"""Translate a candidate action into APSIM calls for one (location, year), run it and compute the reward.
Also holds the season caps (ActionCaps) and the logic that enforces them.
"""
import os
import uuid
import shutil
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

from agri_scholar.simulation.apsim import ApsimController
from agri_scholar.simulation import reward as rm

log = logging.getLogger("agri_sigra")


# smallest dose counted as an application
MIN_PHYSICAL_UREA_KG = 1.0
MIN_PHYSICAL_IRRIG_MM = 1.0


def resolve_dose_window(bound_min: float, bound_max: float, min_dose: float,
                        events_used: int, max_events: int,
                        amount_used: float, season_max: float,
                        physical_floor: float):
    """Allowed (lo, hi) dose for this week, or None if no application is allowed."""
    if events_used >= max_events:
        return None
    remaining = season_max - amount_used
    lo = max(float(bound_min), float(min_dose), float(physical_floor))
    hi = min(float(bound_max), float(remaining))
    if hi < lo:
        return None
    return lo, hi


@dataclass
class ActionCaps:
    """Season caps: max applications, minimum dose and season totals for urea and irrigation."""
    max_urea_splits: int
    max_irrigation_events: int
    min_urea_kg: float
    min_irrigation_mm: float
    season_max_urea_kg: float = float("inf")
    season_max_irrigation_mm: float = float("inf")

    def usage(self, action: dict, weeks) -> dict:
        """Counts applications and totals over the given week indices."""
        n_ev = w_ev = 0
        n_kg = w_mm = 0.0
        for w in weeks:
            n = float(action.get(f"n_dose_{w}", 0.0) or 0.0)
            i = float(action.get(f"w_dose_{w}", 0.0) or 0.0)
            if n > 0:
                n_ev += 1
                n_kg += n
            if i > 0:
                w_ev += 1
                w_mm += i
        return {"urea_events": n_ev, "irrig_events": w_ev, "urea_kg": n_kg, "irrig_mm": w_mm}

    def remaining(self, locked_actions: dict, upto_week: int) -> dict:
        """What is left of the season budget after weeks [0, upto_week)."""
        u = self.usage(locked_actions, range(upto_week))
        return {
            "urea_splits_left": max(0, self.max_urea_splits - u["urea_events"]),
            "irrig_events_left": max(0, self.max_irrigation_events - u["irrig_events"]),
            "urea_kg_left": max(0.0, self.season_max_urea_kg - u["urea_kg"]),
            "irrig_mm_left": max(0.0, self.season_max_irrigation_mm - u["irrig_mm"]),
            **u,
        }

    def urea_window(self, bound_min, bound_max, locked_actions, week_idx):
        u = self.usage(locked_actions, range(week_idx))
        return resolve_dose_window(bound_min, bound_max, self.min_urea_kg, u["urea_events"],
                                   self.max_urea_splits, u["urea_kg"], self.season_max_urea_kg,
                                   MIN_PHYSICAL_UREA_KG)

    def irrigation_window(self, bound_min, bound_max, locked_actions, week_idx):
        u = self.usage(locked_actions, range(week_idx))
        return resolve_dose_window(bound_min, bound_max, self.min_irrigation_mm, u["irrig_events"],
                                   self.max_irrigation_events, u["irrig_mm"],
                                   self.season_max_irrigation_mm, MIN_PHYSICAL_IRRIG_MM)

    def enforce(self, action: dict, total_weeks: int):
        """Return (clean_action, violations) after clipping every dose that breaks a cap."""
        clean = dict(action)
        violations = []
        n_ev = w_ev = 0
        n_kg = w_mm = 0.0
        for w in range(total_weeks):
            for kind, key, min_d, max_ev, season_max, floor in (
                ("urea", f"n_dose_{w}", self.min_urea_kg, self.max_urea_splits,
                 self.season_max_urea_kg, MIN_PHYSICAL_UREA_KG),
                ("irrigation", f"w_dose_{w}", self.min_irrigation_mm, self.max_irrigation_events,
                 self.season_max_irrigation_mm, MIN_PHYSICAL_IRRIG_MM),
            ):
                amt = float(clean.get(key, 0.0) or 0.0)
                if amt <= 0:
                    clean[key] = 0.0
                    continue
                used_ev, used_amt = (n_ev, n_kg) if kind == "urea" else (w_ev, w_mm)
                lo = max(min_d, floor)
                if used_ev >= max_ev:
                    violations.append(f"{key}={amt:.1f}: exceeds max {kind} events ({max_ev}) -> 0")
                    clean[key] = 0.0
                    continue
                room = season_max - used_amt
                if amt > room:
                    violations.append(f"{key}={amt:.1f}: exceeds season {kind} ceiling "
                                      f"(room {max(room, 0):.1f}) -> clipped")
                    amt = max(room, 0.0)
                if amt < lo:
                    violations.append(f"{key}={amt:.1f}: below min {kind} dose ({lo:.1f}) -> 0")
                    clean[key] = 0.0
                    continue
                clean[key] = amt
                if kind == "urea":
                    n_ev += 1
                    n_kg += amt
                else:
                    w_ev += 1
                    w_mm += amt
        return clean, violations


@dataclass
class SimulationResult:
    action: dict
    yield_kg_ha: float
    reward: float
    reward_breakdown: dict
    diagnostics: dict
    sow_date_str: str
    materials_applied: list
    irrigation_applied: list
    crop_failure: bool = False
    cap_violations: list = field(default_factory=list)


class SimulationEngine:
    """Runs candidate actions in APSIM for one (location, year)."""

    def __init__(self, location: str, year: int, apsim_base_file: str,
                 default_met_file: str, sow_window_start_doy: int, sow_window_end_doy: int,
                 total_weeks: int, prices: rm.Prices, shadow_costs: rm.ShadowCosts,
                 soil_profiles: dict, action_caps: ActionCaps, scratch_dir: str,
                 cultivar: str = "MAH-14-138", population: float = 8.5):
        self.location = location
        self.year = year
        self.apsim_base_file = apsim_base_file
        self.default_met_file = default_met_file
        self.sow_window_start_doy = int(sow_window_start_doy)
        self.sow_window_end_doy = int(sow_window_end_doy)
        self.total_weeks = total_weeks
        self.prices = prices
        self.shadow_costs = shadow_costs
        self.soil_profiles = soil_profiles
        self.caps = action_caps
        self.scratch_dir = scratch_dir
        self.cultivar = cultivar
        self.population = population
        os.makedirs(self.scratch_dir, exist_ok=True)

    def effective_sow_doy(self, sow_doy) -> int:
        """Sowing DOY APSIM will actually use (clipped into the sowing window)."""
        return max(self.sow_window_start_doy, min(int(sow_doy), self.sow_window_end_doy))

    def run(self, action: Dict, met_file: Optional[str] = None) -> SimulationResult:
        met_file = met_file or self.default_met_file
        # simulate only a cap-compliant action
        action, cap_violations = self.caps.enforce(action, self.total_weeks)
        if cap_violations:
            log.warning("Engine enforced caps on candidate:\n  " + "\n  ".join(cap_violations))
        trial_id = uuid.uuid4().hex[:8]
        trial_dir = os.path.join(self.scratch_dir, f"trial_{trial_id}")
        working_file = os.path.join(trial_dir, f"{self.location}_{self.year}.apsimx")

        try:
            ctrl = ApsimController(self.apsim_base_file, working_file)
            ctrl.set_clock_and_weather(self.year, met_file)
            ctrl.set_location_soil_profile(self.location, self.soil_profiles)
            ctrl.disable_default_managers()

            # sowing
            sow_doy = self.effective_sow_doy(action.get("sow_doy", self.sow_window_start_doy))
            sow_date = datetime.datetime(self.year, 1, 1) + datetime.timedelta(days=sow_doy - 1)
            sow_str = sow_date.strftime("%d-%b")
            end_str = (sow_date + datetime.timedelta(days=7)).strftime("%d-%b")
            ctrl.set_sowing_rules(sow_str, end_str, self.population, self.cultivar)

            materials_applied, irrigation_applied = [], []

            # land preparation (mulch is one C_F event)
            tillage_date = (sow_date - datetime.timedelta(days=3)).strftime("%d-%b")
            ctrl.add_tillage_event(tillage_date, fraction_incorporated=action.get("tillage_fraction", 0.5))
            mulch_kg = float(action.get("mulch_mass", 0.0))
            if mulch_kg > 0:
                ctrl.set_mulch(mulch_kg)
                materials_applied.append(rm.MaterialApplication("Mulch", mulch_kg))

            # basal DAP at sowing
            basal_dap = float(action.get("basal_dap_kg", 0.0))
            if basal_dap >= 10.0:
                ctrl.add_fertilizer(sow_str, basal_dap, fert_type="DAP")
                materials_applied.append(rm.MaterialApplication("DAP", basal_dap))

            # in-season urea and irrigation
            for w in range(self.total_weeks):
                action_date = (sow_date + datetime.timedelta(weeks=w)).strftime("%d-%b")
                n_amt = float(action.get(f"n_dose_{w}", 0.0))
                w_amt = float(action.get(f"w_dose_{w}", 0.0))
                if n_amt > 0:
                    ctrl.add_fertilizer(action_date, n_amt, fert_type="UreaN")
                    materials_applied.append(rm.MaterialApplication("Urea", n_amt))
                if w_amt > 0:
                    ctrl.apply_irrigation(action_date, w_amt)
                    irrigation_applied.append(rm.IrrigationEvent(w_amt))

            # run APSIM
            diag = ctrl.run_and_get_yield(sow_doy=sow_doy, total_weeks=self.total_weeks)
            yield_val = diag.get("yield_kg_ha", 0)

            indicators = rm.SustainabilityIndicators(
                I_wl=diag.get("waterlogging_index", 0.0),
                I_no=diag.get("n_oversupply_index", 0.0),
                I_dd=diag.get("total_drainage_mm", 0.0),
                I_nl=diag.get("total_leach_no3_nh4", 0.0),
                I_cl=diag.get("soc_loss_index", 0.0),
            )
            reward_breakdown = rm.compute_reward(
                yield_val, materials_applied, irrigation_applied, indicators, self.prices, self.shadow_costs
            )

            return SimulationResult(
                action=action, yield_kg_ha=yield_val, reward=reward_breakdown["reward"],
                reward_breakdown=reward_breakdown, diagnostics=diag, sow_date_str=sow_str,
                materials_applied=materials_applied, irrigation_applied=irrigation_applied,
                crop_failure=reward_breakdown.get("crop_failure", False),
                cap_violations=cap_violations,
            )
        except Exception:
            log.exception(f"SimulationEngine.run failed for trial {trial_id}")
            failure = rm.compute_reward(0, [], [], rm.SustainabilityIndicators(), self.prices, self.shadow_costs)
            return SimulationResult(action=action, yield_kg_ha=0, reward=failure["reward"],
                                     reward_breakdown=failure, diagnostics={"status": "error"},
                                     sow_date_str="", materials_applied=[], irrigation_applied=[], crop_failure=True)
        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)
