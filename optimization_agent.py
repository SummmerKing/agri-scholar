
"""Optimization Agent (OA), Section 3.3.

Optuna search inside the ABA bounds. search_sowing scans the sowing window on constructed weather;
search plans urea over the remaining weeks (receding horizon) and irrigation for the current week.
Each trial is scored by the VFA through APSIM.
"""
import logging
import optuna

from verification_feedback_agent import VerificationFeedbackAgent
from simulation_engine import (SimulationResult, ActionCaps, resolve_dose_window,
                               MIN_PHYSICAL_UREA_KG)
from seeding import derive_seed

log = logging.getLogger("agri_sigra")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptimizationAgent:
    def __init__(self, vfa: VerificationFeedbackAgent, total_weeks: int,
                 n_trials_preseason: int, n_trials_later: int, n_jobs: int, seed: int = None,
                 trials_per_horizon_week: int = 4, horizon_weeks: int = None,
                 future_dose_step: float = 5.0, max_single_urea_kg: float = None):
        self.vfa = vfa
        self.total_weeks = total_weeks
        self.n_trials_preseason = n_trials_preseason
        self.n_trials_later = n_trials_later
        self.n_jobs = n_jobs
        # Per-call sub-seeds (see seeding.py). Use n_jobs=1 for bit-for-bit reproducibility.
        self.seed = seed
        self._call_count = 0
        # receding horizon
        self.trials_per_horizon_week = trials_per_horizon_week  # extra trials per future week planned
        self.horizon_weeks = horizon_weeks
        self.future_dose_step = future_dose_step                # coarse kg step for future weeks
        self.max_single_urea_kg = max_single_urea_kg            # per-split ceiling for future weeks
        self._prev_plan_params = {}                             # warm start from last search()

    def _evaluate_candidate(self, action: dict, met_file: str) -> SimulationResult:
        return self.vfa.score_candidate(action, met_file=met_file)

    def _make_study(self):
        sampler = None
        if self.seed is not None:
            self._call_count += 1
            sampler = optuna.samplers.TPESampler(seed=derive_seed(self.seed, self._call_count))
        return optuna.create_study(direction="maximize", sampler=sampler)

    def _run_study(self, objective, n_trials, best_holder, label):
        study = self._make_study()
        log.info(f"OA: searching {label} with {n_trials} trials (maximizing profit R(pi), Eq. 1)")
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)
        best_result = best_holder["result"]
        if best_result is None:
            log.warning(f"OA: no valid candidate found for {label}")
            return None, None, best_holder["reward"]
        log.info(f"OA: {label} best reward = Rs.{best_holder['reward']:.2f} "
                 f"(yield {best_result.yield_kg_ha:.0f} kg/ha)")
        return dict(best_result.action), best_result, best_holder["reward"]

    def search_sowing(self, window_start: int, window_end: int, met_files: list, total_weeks: int,
                      stride: int = 1, lambda_risk: float = 0.5, smooth_halfwidth: int = 1,
                      neutral_dap_kg: float = 60.0, neutral_tillage: float = 0.5, n_jobs: int = 1):
        """Full-window sowing-date search, scored by APSIM."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        days = list(range(int(window_start), int(window_end) + 1, max(1, int(stride))))
        if days[-1] != int(window_end):
            days.append(int(window_end))   # always evaluate the last allowed day

        def neutral(doy):
            a = {"sow_doy": doy, "mulch_mass": 0.0, "basal_dap_kg": float(neutral_dap_kg),
                 "tillage_fraction": float(neutral_tillage)}
            for w in range(total_weeks):
                a[f"n_dose_{w}"] = 0.0
                a[f"w_dose_{w}"] = 0.0
            return a

        jobs = [(d, k) for d in days for k in range(len(met_files))]
        log.info(f"OA: sowing search over DOY {days[0]}..{days[-1]} ({len(days)} dates) x {len(met_files)} "
                 f"weather members = {len(jobs)} APSIM runs")

        def run_one(job):
            d, k = job
            r = self._evaluate_candidate(neutral(d), met_files[k])
            return d, k, float(r.reward), float(r.yield_kg_ha)

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                out = list(ex.map(run_one, jobs))
        else:
            out = [run_one(j) for j in jobs]

        rewards = {d: [] for d in days}
        yields = {d: [] for d in days}
        for d, k, rew, y in out:
            rewards[d].append(rew)
            yields[d].append(y)

        table = []
        for d in days:
            r = np.array(rewards[d])
            table.append({"doy": d, "mean_reward": float(r.mean()), "std_reward": float(r.std()),
                          "score": float(r.mean() - lambda_risk * r.std()),
                          "mean_yield_kg_ha": float(np.mean(yields[d])), "rewards": rewards[d]})
        scores = np.array([t["score"] for t in table])
        h = max(0, int(smooth_halfwidth))
        for i, t in enumerate(table):
            t["smoothed_score"] = float(scores[max(0, i - h): i + h + 1].mean())
        best = max(table, key=lambda t: (t["smoothed_score"], -t["doy"]))
        for t in table:
            log.info(f"  sow DOY {t['doy']}: mean=Rs.{t['mean_reward']:.0f} std={t['std_reward']:.0f} "
                     f"score={t['score']:.0f} smoothed={t['smoothed_score']:.0f} yield={t['mean_yield_kg_ha']:.0f}")
        log.info(f"OA: sowing search picked DOY {best['doy']} (smoothed score Rs.{best['smoothed_score']:.0f})")
        return int(best["doy"]), table

    def search_preseason(self, bounds: dict, met_file: str, win_start: int, win_end: int):
        """Land-prep search (mulch, basal DAP, tillage) jointly optimized with sowing date."""
        prep = bounds.get("land_prep_bounds", {"mulch_kg_min": 0, "mulch_kg_max": 1000,
                                               "basal_dap_min": 0, "basal_dap_max": 60})
        best_holder = {"result": None, "reward": -float("inf")}

        def objective(trial: optuna.Trial):
            action = {
                "sow_doy": trial.suggest_int("sow_doy", win_start, win_end),
                "mulch_mass": trial.suggest_float("mulch_mass", prep["mulch_kg_min"],
                                                  max(prep["mulch_kg_max"], prep["mulch_kg_min"]), step=1.0),
                "basal_dap_kg": trial.suggest_float("basal_dap_kg", prep["basal_dap_min"],
                                                    max(prep["basal_dap_max"], prep["basal_dap_min"]), step=1.0),
                "tillage_fraction": trial.suggest_float("tillage_fraction", 0.2, 0.8, step=0.05),
            }
            for w in range(self.total_weeks):
                action[f"n_dose_{w}"] = 0.0
                action[f"w_dose_{w}"] = 0.0
            result = self._evaluate_candidate(action, met_file)
            if result.reward > best_holder["reward"]:
                best_holder["reward"], best_holder["result"] = result.reward, result
            return result.reward

        return self._run_study(objective, self.n_trials_preseason, best_holder, "pre-season decisions")

    @staticmethod
    def _snap(value: float, lo: float, hi: float, step: float) -> float:
        """Round to `step`, then clamp into [lo, hi] (so the dose is always inside the window)."""
        v = round(value / step) * step
        if v > hi:
            v = (hi // step) * step
        if v < lo:
            v = lo
        return round(min(v, hi), 4)

    def search(self, bounds: dict, week_idx: int, locked_actions: dict, met_file: str, caps: ActionCaps):
        week_bounds = {int(wb["week_index"]): wb for wb in bounds.get("in_season_weeks", [])}
        b = week_bounds.get(week_idx, {"urea_min": 0, "urea_max": 0, "irrig_min": 0, "irrig_max": 0})
        n_weeks = self.total_weeks
        last_plan_week = n_weeks if self.horizon_weeks is None else min(n_weeks, week_idx + self.horizon_weeks)

        # current week uses the ABA bounds; future weeks get a wide envelope
        rem0 = caps.remaining(locked_actions, week_idx)
        single_max = self.max_single_urea_kg
        if single_max is None:
            per_split = caps.season_max_urea_kg / max(caps.max_urea_splits, 1)
            single_max = max(float(b.get("urea_max", 0.0)), 2.0 * per_split)
        single_max = min(single_max, caps.season_max_urea_kg)
        fut_lo, fut_hi = 0.0, single_max

        u_win_now = caps.urea_window(b.get("urea_min", 0.0), b.get("urea_max", 0.0), locked_actions, week_idx)
        i_win = caps.irrigation_window(b.get("irrig_min", 0.0), b.get("irrig_max", 0.0), locked_actions, week_idx)
        urea_possible_later = (rem0["urea_splits_left"] > 0 and rem0["urea_kg_left"] >= max(caps.min_urea_kg, 1.0)
                               and last_plan_week > week_idx)
        log.info(f"OA week {week_idx}: urea window now={u_win_now}, irrigation window={i_win} | "
                 f"budget left={rem0} | planning weeks {week_idx}..{last_plan_week - 1}")

        best_holder = {"result": None, "reward": -float("inf"), "params": None}

        def objective(trial: optuna.Trial):
            action = dict(locked_actions)
            used_ev, used_kg = rem0["urea_events"], rem0["urea_kg"]
            irrig = 0.0
            if i_win is not None and trial.suggest_categorical("apply_irrigation", [False, True]):
                irrig = min(trial.suggest_float("w_dose", i_win[0], i_win[1], step=0.1), i_win[1])
            action[f"w_dose_{week_idx}"] = irrig
            for w in range(week_idx + 1, n_weeks):
                action[f"w_dose_{w}"] = 0.0

            for w in range(week_idx, n_weeks):
                u = 0.0
                if w < last_plan_week:
                    if w == week_idx:
                        lo_b, hi_b, step = b.get("urea_min", 0.0), b.get("urea_max", 0.0), 0.1
                    else:
                        lo_b, hi_b, step = fut_lo, fut_hi, self.future_dose_step
                    win = resolve_dose_window(lo_b, hi_b, caps.min_urea_kg, used_ev, caps.max_urea_splits,
                                              used_kg, caps.season_max_urea_kg, MIN_PHYSICAL_UREA_KG)
                    if win is not None and trial.suggest_categorical(f"apply_n_{w}", [False, True]):
                        frac = trial.suggest_float(f"n_frac_{w}", 0.0, 1.0)
                        u = self._snap(win[0] + frac * (win[1] - win[0]), win[0], win[1], step)
                        used_ev += 1
                        used_kg += u
                action[f"n_dose_{w}"] = u

            result = self._evaluate_candidate(action, met_file)   # yield through harvest
            if result.reward > best_holder["reward"]:
                best_holder.update(reward=result.reward, result=result, params=dict(trial.params))
            return result.reward

        if u_win_now is None and i_win is None and not urea_possible_later:
            action = dict(locked_actions)
            for w in range(week_idx, n_weeks):
                action[f"n_dose_{w}"] = 0.0
                action[f"w_dose_{w}"] = 0.0
            result = self._evaluate_candidate(action, met_file)
            log.info(f"OA week {week_idx}: no operation permitted now or later -> single zero-input evaluation")
            return dict(result.action), result, result.reward

        weeks_ahead = max(0, last_plan_week - week_idx - 1)
        n_trials = self.n_trials_later + self.trials_per_horizon_week * weeks_ahead
        label = f"week {week_idx} (receding horizon, {weeks_ahead} future weeks)"

        study = self._make_study()
        # warm starts: no N now, and the previous plan
        study.enqueue_trial({f"apply_n_{w}": False for w in range(week_idx, last_plan_week)})
        prev = {k: v for k, v in self._prev_plan_params.items()
                if k.startswith(("apply_n_", "n_frac_")) and week_idx <= int(k.rsplit("_", 1)[1]) < last_plan_week}
        if prev:
            study.enqueue_trial(prev)
        log.info(f"OA: searching {label} with {n_trials} trials (maximizing profit R(pi), Eq. 1)")
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        best_result = best_holder["result"]
        if best_result is None:
            log.warning(f"OA: no valid candidate found for {label}")
            return None, None, best_holder["reward"]
        self._prev_plan_params = best_holder["params"] or {}
        plan = {w: best_result.action.get(f"n_dose_{w}", 0.0) for w in range(week_idx, n_weeks)
                if best_result.action.get(f"n_dose_{w}", 0.0) > 0}
        log.info(f"OA: {label} best reward = Rs.{best_holder['reward']:.2f} (yield {best_result.yield_kg_ha:.0f} kg/ha) "
                 f"| tentative urea plan (kg by week; only week {week_idx} is committed): {plan or 'none'}")
        return dict(best_result.action), best_result, best_holder["reward"]
