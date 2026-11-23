"""Plot Agri-SCHOLAR yields against government baseline yields, one figure per location.

Set RESULTS_DIR to the results folder to plot (default: results/mandya).
"""

import glob
import json
import os
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
rng = random.randint(1000, 9999)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mandya")
OUTPUT_PATH = os.path.join(RESULTS_DIR, f"baseline_comparison_{rng}.png")

GOVT_BASELINE_DATA = {
    2014: 4090, 2015: 3620, 2016: 870, 2017: 2830,
    2018: 3040, 2019: 3400, 2020: 2650, 2021: 2180, 2022: 2550,
}

OUR_BASELINE_DATA = {
    2015: 9241, 2016: 5343, 2017: 8880, 2018: 6720,
    2019: 6726, 2020: 4878, 2021: 7833, 2022: 7288,
    2023: 8559, 2024: 4780, 2025: 6551
}
GOVT_BASELINE_DATA = OUR_BASELINE_DATA

TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")

COLOR_BASELINE = "#2a78d6"
COLOR_OURS = "#eb6834"


def load_latest_yields():
    """Returns {(location, year): (yield_kg_ha, source_file)} using the
    newest-timestamped 'ok' entry for each (location, year).
    """
    best = {}
    best_ts = {}

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        match = TIMESTAMP_RE.search(os.path.basename(path))
        if not match:
            continue
        timestamp = match.group(1)

        with open(path) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]

        for entry in entries:
            if entry.get("status") != "ok":
                continue
            season_result = entry.get("season_result") or {}
            yield_kg_ha = season_result.get("final_yield_kg_ha")
            location = entry.get("location")
            year = entry.get("year")
            if yield_kg_ha is None or location is None or year is None:
                continue

            key = (location, year)
            if key not in best_ts or timestamp > best_ts[key]:
                best_ts[key] = timestamp
                best[key] = (yield_kg_ha, os.path.basename(path))

    return best


def main():
    latest = load_latest_yields()
    if not latest:
        raise SystemExit(f"No successful results found in {RESULTS_DIR}")

    by_location = defaultdict(dict)
    for (location, year), (yield_kg_ha, source) in latest.items():
        by_location[location][year] = (yield_kg_ha, source)

    for location, year_data in by_location.items():
        compared_years = sorted(y for y in year_data if y in GOVT_BASELINE_DATA)
        extra_years = sorted(y for y in year_data if y not in GOVT_BASELINE_DATA)
        missing_years = sorted(y for y in GOVT_BASELINE_DATA if y not in year_data)

        print(f"\n[{location}]")
        for year in compared_years:
            yld, source = year_data[year]
            print(f"  {year}: ours={yld} kg/ha, baseline={GOVT_BASELINE_DATA[year]} kg/ha  ({source})")
        if missing_years:
            print(f"  no simulation result for baseline year(s): {missing_years}")
        if extra_years:
            print(f"  simulation results with no baseline to compare: {extra_years}")

        if not compared_years:
            continue

        our_yields = [year_data[y][0] for y in compared_years]
        baseline_yields = [GOVT_BASELINE_DATA[y] for y in compared_years]

        x = np.arange(len(compared_years))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(7, len(compared_years) * 1.1), 5.5))

        bars_baseline = ax.bar(x - width / 2, baseline_yields, width,
                                label="Government baseline", color=COLOR_BASELINE)
        bars_ours = ax.bar(x + width / 2, our_yields, width,
                            label="Agri-SCHOLAR", color=COLOR_OURS)

        ax.bar_label(bars_baseline, padding=3, fontsize=9, color="#3a3a3a")
        ax.bar_label(bars_ours, padding=3, fontsize=9, color="#3a3a3a")

        ax.set_xticks(x)
        ax.set_xticklabels(compared_years)
        ax.set_ylabel("Yield (kg/ha)")
        ax.set_ylim(0, max(our_yields + baseline_yields) * 1.18)
        ax.set_title(f"Agri-SCHOLAR vs. government baseline yield, {location}", pad=36)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path = OUTPUT_PATH if len(by_location) == 1 else os.path.join(
            RESULTS_DIR, f"baseline_comparison_{location}_{rng}.png"
        )
        fig.savefig(out_path, dpi=150)
        print(f"  saved chart -> {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
