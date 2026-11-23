"""Advisory Bounding Agent (ABA), Section 3.2.

Each week the ABA proposes literature-grounded bounds for that week's urea top-dressing and
irrigation. Season-level decisions (sowing, land prep, split and event caps) come from the
Preseason Agent and are passed in as fixed constraints.
"""
import json
import logging

from llm_backend import LocalLLM
from simulation_engine import ActionCaps

log = logging.getLogger("agri_sigra")


def _f(x, default=0.0):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _fmt(x):
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return "?"


def _sanitize_weekly(parsed: dict, week_idx: int, caps: ActionCaps, locked_actions: dict) -> dict:
    fixes = []
    weeks = parsed.get("in_season_weeks")
    if not isinstance(weeks, list):
        weeks = []

    entry = None
    for wb in weeks:
        if isinstance(wb, dict) and int(_f(wb.get("week_index"), -1)) == week_idx:
            entry = wb
            break
    if entry is None:
        fixes.append(f"no in_season_weeks entry for week {week_idx} -- inserted conservative default")
        entry = {"week_index": week_idx, "urea_min": 0.0, "urea_max": 30.0,
                 "irrig_min": 0.0, "irrig_max": 30.0, "reasoning": "Default: ABA omitted this week."}
    # keep only the current week
    parsed["in_season_weeks"] = [entry]
    entry["week_index"] = week_idx

    for lo_k, hi_k in (("urea_min", "urea_max"), ("irrig_min", "irrig_max")):
        lo, hi = max(0.0, _f(entry.get(lo_k))), max(0.0, _f(entry.get(hi_k)))
        if hi < lo:
            fixes.append(f"{hi_k}={hi} < {lo_k}={lo} -- max raised to min")
            hi = lo
        entry[lo_k], entry[hi_k] = lo, hi

    uw = caps.urea_window(entry["urea_min"], entry["urea_max"], locked_actions, week_idx)
    iw = caps.irrigation_window(entry["irrig_min"], entry["irrig_max"], locked_actions, week_idx)
    entry["urea_window"] = list(uw) if uw else None
    entry["irrig_window"] = list(iw) if iw else None
    if uw is None and entry["urea_max"] > 0:
        fixes.append("urea not permitted this week by season caps (splits/ceiling/min dose) -- OA will skip urea")
    if iw is None and entry["irrig_max"] > 0:
        fixes.append("irrigation not permitted this week by season caps -- OA will skip irrigation")

    if fixes:
        log.warning(f"ABA sanitize (week {week_idx}):\n  " + "\n  ".join(fixes))
    return parsed


def _make_fallback_strategy(week_idx: int, caps: ActionCaps, locked_actions: dict) -> dict:
    return _sanitize_weekly({
        "reflection": "LLM output could not be parsed -- using a conservative fallback bound set.",
        "strategy_name": "Fallback",
        "in_season_weeks": [{"week_index": week_idx, "urea_min": 0.0, "urea_max": 30.0,
                             "irrig_min": 0.0, "irrig_max": 30.0,
                             "reasoning": "Fallback: conservative bounds."}],
    }, week_idx, caps, locked_actions)


PROMPT_TEMPLATE = """You are an elite agronomist AI advising a maize farmer in {location} for the {year} Kharif season.
Pre-season decisions (sowing, land preparation, and the season's management structure) are ALREADY LOCKED. Your job now is
the WEEKLY decision: bounds for this week's urea top-dressing and supplemental irrigation.

You do NOT choose exact values -- you choose DEFENSIBLE BOUNDS. A downstream numerical optimizer searches only inside them
(it may also choose to skip the operation this week) and every candidate is verified in the APSIM crop simulator.

CRITICAL CLIMATE-RESILIENCE GUIDELINES:
- Withhold Scheduled Irrigation: do not issue irrigation bounds when rainfall already exceeds 200 mm/week.
- Dynamic Nitrogen Top-Dressing: after heavy percolation events, widen urea bounds to compensate for leached NO3.

LOCKED SEASON STRUCTURE (you cannot change this; the simulator will reject anything that breaks it):
  - Max urea applications for the season: {max_urea_splits}  | used so far: {urea_events} | LEFT: {urea_splits_left}
  - Max irrigation events for the season: {max_irrigation_events} | used so far: {irrig_events} | LEFT: {irrig_events_left}
  - Minimum single urea dose: {min_urea_kg} kg/ha | Minimum single irrigation dose: {min_irrigation_mm} mm
  - Season urea ceiling: {season_max_urea_kg} kg/ha | used {urea_kg:.1f} | LEFT {urea_kg_left:.1f}
  - Season irrigation ceiling: {season_max_irrigation_mm} mm | used {irrig_mm:.1f} | LEFT {irrig_mm_left:.1f}
If a "LEFT" value is 0, set that operation's bounds to 0 this week. Spend scarce splits where they matter most
(e.g. before tasseling, or after leaching rain) rather than early by default.

CURRENT DECISION WEEK: {week_idx} (0-indexed, out of {total_weeks})

CONTEXT BUNDLE FROM THE RETRIEVAL AGENT:
{context_bundle}

SEASON HISTORY -- what has been issued and its simulated outcome so far:
{season_history_block}

ACTIONS ALREADY LOCKED IN THIS SEASON (pre-season decisions and weeks 0 to {week_idx_minus_1}):
{locked_actions}
{within_week_trace_block}{feedback_block}
Respond with ONLY a JSON object (no markdown fences, no commentary outside the JSON) in exactly this schema:
{{
  "reflection": "<your agronomic reasoning, referencing the weather context, the remaining budget and any feedback above>",
  "strategy_name": "<short name for this week's approach>",
  "in_season_weeks": [
    {{"week_index": {week_idx}, "urea_min": <float>, "urea_max": <float>, "irrig_min": <float>, "irrig_max": <float>, "reasoning": "<why these bounds>"}}
  ]
}}
"""


def _build_season_history(locked_actions: dict, weekly_log: dict, week_idx: int) -> str:
    if not weekly_log or week_idx == 0:
        return "(no history yet -- this is the first decision week)"
    lines = ["Week | Urea issued (kg/ha) | Irrigation issued (mm) | Simulated yield (kg/ha)", "-" * 72]
    for w in range(week_idx):
        urea = locked_actions.get(f"n_dose_{w}", 0.0)
        irrig = locked_actions.get(f"w_dose_{w}", 0.0)
        rb = (weekly_log.get(f"week_{w}", {}) or {}).get("reward_breakdown") or {}
        y = rb.get("yield_kg_ha")
        lines.append(f"  {w:>2}  | {urea:>19.1f} | {irrig:>22.1f} | {(f'{y:.0f}' if y is not None else 'n/a'):>23}")
    return "\n".join(lines)


def _build_within_week_trace(reflection_trace: list) -> str:
    if not reflection_trace:
        return ""
    lines = ["\nWHAT YOU TRIED EARLIER IN THIS WEEK'S REFLECTION ITERATIONS:",
             "Use this to avoid repeating what didn't work and to build on what did."]
    for entry in reflection_trace:
        it = entry.get("iteration", "?")
        reward, y = entry.get("reward"), entry.get("yield_kg_ha")
        if reward is None:
            lines.append(f"  Iteration {it}: OA returned no valid candidate (search failed).")
            continue
        note = " [FALLBACK BOUNDS USED -- LLM output was unparsable]" if entry.get("aba_used_fallback") else ""
        lines.append(f"  Iteration {it}: reward=Rs.{reward:.0f}, yield={(f'{y:.0f}' if y is not None else 'n/a')} kg/ha{note}")
        wbs = (entry.get("aba_bounds") or {}).get("in_season_weeks", [])
        if wbs:
            wb = wbs[0]
            lines.append(f"    Bounds you gave: urea [{_fmt(wb.get('urea_min'))}, {_fmt(wb.get('urea_max'))}] kg/ha | "
                         f"irrig [{_fmt(wb.get('irrig_min'))}, {_fmt(wb.get('irrig_max'))}] mm")
        chosen = entry.get("chosen_doses")
        if chosen:
            lines.append(f"    Optimizer applied: urea {chosen['urea']:.1f} kg/ha, irrigation {chosen['irrig']:.1f} mm")
    lines.append("")
    return "\n".join(lines)


class AdvisoryBoundingAgent:
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.fallback_count = 0

    def _build_prompt(self, location, year, week_idx, total_weeks, context_bundle, locked_actions,
                      caps, vfa_feedback, reflection_trace, weekly_log):
        feedback_block = ""
        if vfa_feedback:
            feedback_block = (
                f"\nVFA (SIMULATOR) FEEDBACK FROM THE PREVIOUS REFLECTION ITERATION:\n{vfa_feedback}\n"
                "Use this feedback to tighten or relax your bounds -- e.g. widen irrigation bounds if "
                "reproductive-stage water stress was reported.\n")
        rem = caps.remaining(locked_actions, week_idx)
        return PROMPT_TEMPLATE.format(
            location=location, year=year, week_idx=week_idx, week_idx_minus_1=max(week_idx - 1, 0),
            total_weeks=total_weeks, context_bundle=context_bundle,
            locked_actions=json.dumps(locked_actions, indent=2) if locked_actions else "(none yet)",
            feedback_block=feedback_block,
            season_history_block=_build_season_history(locked_actions, weekly_log, week_idx),
            within_week_trace_block=_build_within_week_trace(reflection_trace),
            max_urea_splits=caps.max_urea_splits, max_irrigation_events=caps.max_irrigation_events,
            min_urea_kg=caps.min_urea_kg, min_irrigation_mm=caps.min_irrigation_mm,
            season_max_urea_kg=caps.season_max_urea_kg, season_max_irrigation_mm=caps.season_max_irrigation_mm,
            **rem,
        )

    def propose_bounds(self, location: str, year: int, week_idx: int, total_weeks: int,
                       context_bundle: str, locked_actions: dict, caps: ActionCaps,
                       vfa_feedback: str = None, reflection_trace: list = None,
                       weekly_log: dict = None) -> dict:
        prompt = self._build_prompt(location, year, week_idx, total_weeks, context_bundle, locked_actions,
                                    caps, vfa_feedback, reflection_trace or [], weekly_log or {})
        log.debug(f"ABA prompt (week {week_idx}):\n{prompt}")
        response = self.llm.generate_text(prompt)
        log.info(f"ABA raw response (week {week_idx}):\n{response}")

        parsed = self.llm.extract_json(response)
        if parsed is not None:
            return _sanitize_weekly(parsed, week_idx, caps, locked_actions)

        self.fallback_count += 1
        log.warning(f"ABA returned unparsable output (week {week_idx}); using conservative fallback bounds. "
                    f"Total fallbacks this season so far: {self.fallback_count}.")
        return _make_fallback_strategy(week_idx, caps, locked_actions)
