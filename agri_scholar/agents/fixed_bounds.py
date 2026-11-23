"""Rule-based replacements for the Preseason Agent and the ABA, used in the no-LLM ablations.

FixedBounds: 0-150 kg/ha urea and 0-150 mm irrigation every week, effectively no caps.
PopBounds: bounds derived from the regional Package of Practices. PoP doses become +/-50% ranges,
each PoP urea week a +/-1 week window (one dose per window), season totals capped at 1.5x PoP.
"""
from agri_scholar.agents.advisory_bounding import _sanitize_weekly
from agri_scholar.agents.preseason import sanitize_preseason


class _RuleAgent:
    strategy = "Rule-based"

    def __init__(self, total_weeks):
        self.total_weeks = total_weeks
        self.fallback_count = 0

    def _weekly(self, week_idx, caps, locked_actions, u_lo, u_hi, i_lo, i_hi, reasoning):
        return _sanitize_weekly({
            "reflection": f"{self.strategy} bounds.",
            "strategy_name": self.strategy,
            "in_season_weeks": [{"week_index": week_idx, "urea_min": u_lo, "urea_max": u_hi,
                                 "irrig_min": i_lo, "irrig_max": i_hi, "reasoning": reasoning}],
        }, week_idx, caps, locked_actions)


class FixedBounds(_RuleAgent):
    strategy = "Unrestricted"
    MAX_UREA_KG = 150.0
    MAX_IRRIGATION_MM = 150.0

    def propose(self, *args, **kwargs):
        return sanitize_preseason({
            "reflection": "Unrestricted bounds.",
            "strategy_name": self.strategy,
            "land_prep_bounds": {"mulch_kg_min": 0.0, "mulch_kg_max": 2000.0,
                                 "basal_dap_min": 0.0, "basal_dap_max": 150.0},
            "management_frequency_caps": {
                "max_urea_splits": 9999,
                "max_irrigation_events": 9999,
                "min_urea_dose_kg": 0.0,
                "min_irrigation_dose_mm": 0.0,
                "season_max_urea_kg": self.MAX_UREA_KG * self.total_weeks,
                "season_max_irrigation_mm": self.MAX_IRRIGATION_MM * self.total_weeks,
            },
        })

    def propose_bounds(self, location, year, week_idx, total_weeks, context_bundle, locked_actions, caps,
                       vfa_feedback, reflection_trace, weekly_log):
        return self._weekly(week_idx, caps, locked_actions, 0.0, self.MAX_UREA_KG, 0.0,
                            self.MAX_IRRIGATION_MM, "Unrestricted")


class PopBounds(_RuleAgent):
    strategy = "PoP-derived"
    N_KG, N_WEEKS = 50.0, (0, 4, 8)
    IRRIG_MM, IRRIG_WEEKS = 40.0, (0, 2, 4, 6, 8, 10)
    DOSE_RANGE, WINDOW_WEEKS = 0.5, 1
    DAP_KG = 75.0 / 0.46            # 75 kg P2O5/ha as DAP (46% P2O5)

    def propose(self, *args, **kwargs):
        r = self.DOSE_RANGE
        return sanitize_preseason({
            "reflection": "PoP-derived bounds.",
            "strategy_name": self.strategy,
            "land_prep_bounds": {"mulch_kg_min": 0.0, "mulch_kg_max": 2000.0,
                                 "basal_dap_min": self.DAP_KG * (1 - r),
                                 "basal_dap_max": self.DAP_KG * (1 + r)},
            "management_frequency_caps": {
                "max_urea_splits": len(self.N_WEEKS),
                "max_irrigation_events": len(self.IRRIG_WEEKS),
                "min_urea_dose_kg": self.N_KG * (1 - r),
                "min_irrigation_dose_mm": self.IRRIG_MM * (1 - r),
                "season_max_urea_kg": self.N_KG * len(self.N_WEEKS) * (1 + r),
                "season_max_irrigation_mm": self.IRRIG_MM * len(self.IRRIG_WEEKS) * (1 + r),
            },
        })

    def _urea_window(self, week_idx, locked_actions):
        for pw in self.N_WEEKS:
            lo_w, hi_w = max(0, pw - self.WINDOW_WEEKS), pw + self.WINDOW_WEEKS
            if lo_w <= week_idx <= hi_w:
                used = any(float(locked_actions.get(f"n_dose_{w}", 0.0) or 0.0) > 0
                           for w in range(lo_w, week_idx))
                if used:
                    return 0.0, 0.0
                return self.N_KG * (1 - self.DOSE_RANGE), self.N_KG * (1 + self.DOSE_RANGE)
        return 0.0, 0.0

    def propose_bounds(self, location, year, week_idx, total_weeks, context_bundle, locked_actions, caps,
                       vfa_feedback, reflection_trace, weekly_log):
        u_lo, u_hi = self._urea_window(week_idx, locked_actions)
        # PoP irrigations are two weeks apart, so the windows would overlap: allow irrigation anywhere
        # in the PoP irrigation period instead (event cap = 6)
        if week_idx <= max(self.IRRIG_WEEKS) + self.WINDOW_WEEKS:
            i_lo, i_hi = self.IRRIG_MM * (1 - self.DOSE_RANGE), self.IRRIG_MM * (1 + self.DOSE_RANGE)
        else:
            i_lo, i_hi = 0.0, 0.0
        return self._weekly(week_idx, caps, locked_actions, u_lo, u_hi, i_lo, i_hi,
                            "PoP schedule +/-50% dose, +/-1 week")
