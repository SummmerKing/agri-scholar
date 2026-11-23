"""Preseason Agent (PSA).

Decides once per season, before the first in-season week: land-prep bounds (mulch, basal DAP) and the
season structure (max urea splits and irrigation events, minimum doses, season totals). The sowing
date is fixed beforehand by an APSIM-scored search.
"""
import datetime
import json
import logging

from llm_backend import LocalLLM

log = logging.getLogger("agri_sigra")

# fallback when the LLM output cannot be parsed
_FALLBACK_CAPS = {
    "max_urea_splits": 3, "max_irrigation_events": 4,
    "min_urea_dose_kg": 20.0, "min_irrigation_dose_mm": 20.0,
    "season_max_urea_kg": 150.0, "season_max_irrigation_mm": 200.0,
}


def _f(x, default):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except (TypeError, ValueError):
        return default


def sanitize_preseason(parsed: dict) -> dict:
    """Coerce types, fix inversions, and make the land-prep bounds and caps
    internally consistent. Every correction is logged.
    """
    fixes = []
    parsed.pop("sowing_window", None)   # sowing is locked upstream; ignore any the LLM invents

    lp = parsed.setdefault("land_prep_bounds", {})
    for lo_k, hi_k, dlo, dhi in (("mulch_kg_min", "mulch_kg_max", 0.0, 2000.0),
                                 ("basal_dap_min", "basal_dap_max", 50.0, 150.0)):
        lo = max(0.0, _f(lp.get(lo_k), dlo))
        hi = max(0.0, _f(lp.get(hi_k), dhi))
        if hi < lo:
            fixes.append(f"land_prep {hi_k}={hi} < {lo_k}={lo} -- max raised to min")
            hi = lo
        lp[lo_k], lp[hi_k] = lo, hi

    c = parsed.setdefault("management_frequency_caps", {})
    caps = {
        "max_urea_splits": max(0, int(_f(c.get("max_urea_splits"), _FALLBACK_CAPS["max_urea_splits"]))),
        "max_irrigation_events": max(0, int(_f(c.get("max_irrigation_events"), _FALLBACK_CAPS["max_irrigation_events"]))),
        "min_urea_dose_kg": max(0.0, _f(c.get("min_urea_dose_kg"), _FALLBACK_CAPS["min_urea_dose_kg"])),
        "min_irrigation_dose_mm": max(0.0, _f(c.get("min_irrigation_dose_mm"), _FALLBACK_CAPS["min_irrigation_dose_mm"])),
        "season_max_urea_kg": max(0.0, _f(c.get("season_max_urea_kg"), _FALLBACK_CAPS["season_max_urea_kg"])),
        "season_max_irrigation_mm": max(0.0, _f(c.get("season_max_irrigation_mm"), _FALLBACK_CAPS["season_max_irrigation_mm"])),
    }
    # a ceiling below one minimum dose disables the operation
    for n_k, min_k, tot_k, label in (("max_urea_splits", "min_urea_dose_kg", "season_max_urea_kg", "urea"),
                                     ("max_irrigation_events", "min_irrigation_dose_mm", "season_max_irrigation_mm", "irrigation")):
        if caps[n_k] > 0 and caps[tot_k] < caps[min_k]:
            fixes.append(f"{label}: season ceiling {caps[tot_k]} < min dose {caps[min_k]} -- ceiling raised to min dose")
            caps[tot_k] = caps[min_k]
    parsed["management_frequency_caps"] = caps

    if fixes:
        log.warning("PSA sanitize corrections:\n  " + "\n  ".join(fixes))
    return parsed


def make_fallback() -> dict:
    return sanitize_preseason({
        "reflection": "LLM output could not be parsed -- using a conservative fallback.",
        "strategy_name": "Fallback",
        "land_prep_bounds": {"mulch_kg_min": 0.0, "mulch_kg_max": 2000.0,
                             "basal_dap_min": 50.0, "basal_dap_max": 150.0},
        "management_frequency_caps": dict(_FALLBACK_CAPS),
    })


PROMPT_TEMPLATE = """You are an elite agronomist AI planning the PRE-SEASON decisions for a maize farmer in {location} for the {year} Kharif season.
These decisions are made ONCE, before sowing, and then held fixed for the whole season:
  1. Land preparation at sowing: surface mulch (moisture retention) and basal DAP (root phosphorus).
  2. Season-wide management structure: how many urea top-dressings and irrigation events are agronomically
     defensible, the smallest single dose worth applying, and the season-total ceilings.

You do NOT choose exact values -- you choose DEFENSIBLE BOUNDS for 1. A numerical optimizer will jointly search these bounds ALONG WITH the sowing date (which is searched universally from Mid-May to Mid-August) to find the global optimum.
Every candidate is verified in the APSIM crop simulator. The structure in 2 is a HARD CONTRACT: every later
weekly advisory will be forced to respect it (no more splits than you allow, no dose below your minimum, no total above
your ceilings), so make it realistic for a smallholder in this location.

CRITICAL CLIMATE-RESILIENCE GUIDELINES:
- Do not budget irrigation heavily when the monsoon outlook is wet; keep some urea splits available for leaching events.

TOTAL SEASON LENGTH: {total_weeks} weeks

CONTEXT BUNDLE FROM THE RETRIEVAL AGENT:
{context_bundle}
{feedback_block}
Respond with ONLY a JSON object (no markdown fences, no commentary outside the JSON) in exactly this schema:
{{
  "reflection": "<your agronomic reasoning>",
  "strategy_name": "<short name>",
  "land_prep_bounds": {{"mulch_kg_min": <float>, "mulch_kg_max": <float>, "basal_dap_min": <float>, "basal_dap_max": <float>}},
  "management_frequency_caps": {{
    "max_urea_splits": <int, number of urea top-dressings justified for the full season>,
    "max_irrigation_events": <int, number of irrigation events justified for the full season>,
    "min_urea_dose_kg": <float, minimum meaningful single urea dose in kg/ha>,
    "min_irrigation_dose_mm": <float, minimum meaningful single irrigation dose in mm>,
    "season_max_urea_kg": <float, total urea ceiling for the season in kg/ha>,
    "season_max_irrigation_mm": <float, total irrigation ceiling for the season in mm>
  }}
}}
"""


class PreseasonAgent:
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.fallback_count = 0

    def _build_prompt(self, location, year, total_weeks, context_bundle, vfa_feedback, reflection_trace):
        fb = ""
        if reflection_trace:
            fb += "\nWHAT YOU TRIED EARLIER IN THE PRE-SEASON REFLECTION:\n"
            for t in reflection_trace:
                if t.get("reward") is None:
                    fb += f"  Iteration {t['iteration']}: optimizer found no valid candidate.\n"
                    continue
                b = t.get("psa_bounds") or {}
                a = t.get("best_action") or {}
                fb += (f"  Iteration {t['iteration']}: reward=Rs.{t['reward']:.0f}, yield={t['yield_kg_ha']:.0f} kg/ha "
                       f"| "
                       f"mulch {a.get('mulch_mass')}, DAP {a.get('basal_dap_kg')}\n")
        if vfa_feedback:
            fb += (f"\nVFA (SIMULATOR) FEEDBACK ON YOUR LAST PROPOSAL:\n{vfa_feedback}\n"
                   "Note: in-season urea/irrigation were held at zero during this test, so judge "
                   "land-prep from establishment/stress signals, not from missing top-dressing.\n")
        return PROMPT_TEMPLATE.format(location=location, year=year,
                                      total_weeks=total_weeks, context_bundle=context_bundle,
                                      feedback_block=fb)

    def propose(self, location, year, total_weeks, context_bundle, vfa_feedback=None, reflection_trace=None) -> dict:
        prompt = self._build_prompt(location, year, total_weeks, context_bundle,
                                    vfa_feedback, reflection_trace or [])
        log.debug(f"PSA prompt:\n{prompt}")
        response = self.llm.generate_text(prompt)
        log.info(f"PSA raw response:\n{response}")
        parsed = self.llm.extract_json(response)
        if parsed is not None:
            return sanitize_preseason(parsed)
        self.fallback_count += 1
        log.warning(f"PSA returned unparsable output; using fallback (total fallbacks: {self.fallback_count}).")
        return make_fallback()
