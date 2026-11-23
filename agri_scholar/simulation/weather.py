"""Weather array construction (operational forecast emulation).

    W_t = C_pre || O_{1:7t} || F_{7t+1:7t+7} || A_hat_{7t+8:D}

C_pre is real weather up to sowing, O the observed weeks so far, F a degraded 7-day forecast and
A_hat an analog tail taken from the closest historical years.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ONI values, used as an analog-tail feature
def get_enso_data() -> pd.DataFrame:
    oni_data = {
        1980: [0.59, 0.46, 0.34, 0.38, 0.48, 0.46], 1981: [-0.26, -0.5, -0.47, -0.37, -0.26, -0.29],
        1982: [-0.05, 0.07, 0.19, 0.47, 0.66, 0.72], 1983: [2.18, 1.92, 1.54, 1.29, 1.06, 0.72],
        1984: [-0.6, -0.42, -0.34, -0.43, -0.51, -0.45], 1985: [-1.04, -0.85, -0.77, -0.78, -0.78, -0.63],
        1986: [-0.49, -0.47, -0.31, -0.2, -0.12, -0.04], 1987: [1.23, 1.19, 1.06, 0.95, 0.97, 1.22],
        1988: [0.81, 0.54, 0.14, -0.31, -0.88, -1.3], 1989: [-1.69, -1.43, -1.08, -0.83, -0.58, -0.4],
        1990: [0.14, 0.21, 0.28, 0.29, 0.29, 0.31], 1991: [0.41, 0.26, 0.22, 0.26, 0.45, 0.64],
        1992: [1.71, 1.63, 1.48, 1.29, 1.06, 0.73], 1993: [0.09, 0.3, 0.5, 0.67, 0.7, 0.57],
        1994: [0.06, 0.07, 0.17, 0.31, 0.42, 0.41], 1995: [0.96, 0.72, 0.53, 0.3, 0.14, -0.03],
        1996: [-0.9, -0.75, -0.59, -0.39, -0.31, -0.3], 1997: [-0.5, -0.36, -0.1, 0.28, 0.75, 1.22],
        1998: [2.24, 1.93, 1.44, 0.99, 0.45, -0.13], 1999: [-1.55, -1.3, -1.07, -0.98, -1.02, -1.04],
        2000: [-1.66, -1.41, -1.07, -0.81, -0.71, -0.64], 2001: [-0.68, -0.52, -0.44, -0.34, -0.25, -0.12],
        2002: [-0.15, 0.03, 0.09, 0.2, 0.43, 0.65], 2003: [0.92, 0.63, 0.38, -0.04, -0.26, -0.16],
        2004: [0.37, 0.31, 0.23, 0.17, 0.17, 0.28], 2005: [0.64, 0.58, 0.45, 0.43, 0.29, 0.11],
        2006: [-0.85, -0.77, -0.57, -0.37, -0.14, -0.03], 2007: [0.66, 0.22, -0.12, -0.32, -0.38, -0.47],
        2008: [-1.64, -1.52, -1.29, -1.01, -0.84, -0.61], 2009: [-0.85, -0.79, -0.61, -0.33, 0.01, 0.28],
        2010: [1.5, 1.22, 0.84, 0.35, -0.17, -0.66], 2011: [-1.31, -1.04, -0.8, -0.62, -0.46, -0.37],
        2012: [-0.72, -0.57, -0.46, -0.36, -0.17, 0.06], 2013: [-0.29, -0.29, -0.21, -0.19, -0.27, -0.34],
        2014: [-0.28, -0.32, -0.14, 0.15, 0.31, 0.23], 2015: [0.69, 0.61, 0.65, 0.81, 1.02, 1.25],
        2016: [2.63, 2.28, 1.71, 1.05, 0.49, 0.0], 2017: [-0.19, -0.02, 0.18, 0.31, 0.4, 0.39],
        2018: [-0.77, -0.71, -0.57, -0.39, -0.13, 0.06], 2019: [0.89, 0.86, 0.84, 0.77, 0.64, 0.52],
        2020: [0.64, 0.63, 0.53, 0.3, 0.01, -0.23], 2021: [-0.91, -0.79, -0.71, -0.55, -0.39, -0.3],
        2022: [-0.82, -0.79, -0.86, -0.95, -0.9, -0.78], 2023: [-0.54, -0.29, -0.02, 0.27, 0.57, 0.84],
        2024: [1.92, 1.62, 1.26, 0.82, 0.49, 0.22], 2025: [-0.45, -0.24, -0.06, 0.02, -0.02, -0.04],
    }
    records = [
        {"Year": yr, "DJF": v[0], "JFM": v[1], "FMA": v[2], "MAM": v[3], "AMJ": v[4], "MJJ": v[5]}
        for yr, v in oni_data.items()
    ]
    return pd.DataFrame(records)


def load_met_file(met_path: str) -> pd.DataFrame:
    """Parse an APSIM .met file into a DataFrame with columns year, day, radn, maxt, mint, rain."""
    with open(met_path, "r") as f:
        lines = f.readlines()
    start_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("()")) + 1
    records = []
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 6:
            records.append({
                "year": int(parts[0]), "day": int(parts[1]),
                "radn": float(parts[2]), "maxt": float(parts[3]),
                "mint": float(parts[4]), "rain": float(parts[5]),
            })
    return pd.DataFrame(records)

def load_all_weather(met_dir: str, location: str) -> Dict[int, pd.DataFrame]:
    """Load every `{location}_{year}.met` file in `met_dir` into a {year: DataFrame} map."""
    import glob
    all_weather = {}
    for f in glob.glob(os.path.join(met_dir, f"{location}_*.met")):
        year_str = os.path.basename(f).split("_")[1].split(".")[0]
        if year_str.isdigit():
            all_weather[int(year_str)] = load_met_file(f)
    return all_weather


def save_weather_to_met(df: pd.DataFrame, orig_met_path: str, out_met_path: str):
    """Write a spliced weather DataFrame back out as a valid APSIM .met file (reusing the original header)."""
    with open(orig_met_path, "r") as f:
        lines = f.readlines()
    header = []
    for line in lines:
        header.append(line)
        if line.strip().startswith("()"):
            break
    with open(out_met_path, "w") as f:
        f.writelines(header)
        for _, row in df.iterrows():
            f.write(f"{int(row['year'])} {int(row['day'])} {row['radn']:.2f} "
                     f"{row['maxt']:.1f} {row['mint']:.1f} {row['rain']:.1f}\n")


def summarize_kharif_season(met_path: str, year: int, as_of_day: int) -> Dict:
    """Rainfall narrative using only days up to and including as_of_day."""
    df = load_met_file(met_path)
    df = df[(df["year"] == year) & (df["day"] <= int(as_of_day))].sort_values("day").reset_index(drop=True)
    as_of_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(as_of_day) - 1)
    as_of_str = as_of_date.strftime("%d-%b")

    monthly_rain = {6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0}
    heavy_events, longest_dry, current_dry, kharif_total, premonsoon_total = [], 0, 0, 0.0, 0.0
    kharif_days = 0

    for _, row in df.iterrows():
        day, rain = int(row["day"]), float(row["rain"])
        date_obj = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1)
        month = date_obj.month
        if month <= 5:
            premonsoon_total += rain
        if 6 <= month <= 10:
            kharif_days += 1
            kharif_total += rain
            monthly_rain[month] += rain
            if rain < 2.5:
                current_dry += 1
                longest_dry = max(longest_dry, current_dry)
            else:
                current_dry = 0
            if rain >= 30.0:
                heavy_events.append(f"{date_obj.strftime('%d-%b')} ({rain:.1f}mm)")

    if kharif_days == 0:
        narrative = (f"Observed through {as_of_str}: Pre-monsoon (Jan-May) rainfall {premonsoon_total:.0f}mm. "
                     f"The Kharif season has not started yet; no in-season rainfall observed.")
    else:
        month_names = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
        monthly_str = ", ".join(f"{month_names[m]}: {monthly_rain[m]:.0f}mm"
                                for m in range(6, 11) if m <= as_of_date.month)
        heavy_str = ", ".join(heavy_events[:4]) if heavy_events else "None"
        if len(heavy_events) > 4:
            heavy_str += " (and others)"
        narrative = (
            f"Observed through {as_of_str}: Pre-monsoon (Jan-May) rainfall {premonsoon_total:.0f}mm. "
            f"Kharif-to-date total: {kharif_total:.0f}mm. Monthly distribution so far: [{monthly_str}]. "
            f"Longest dry spell so far: {longest_dry} consecutive days. Major storms so far: {heavy_str}."
        )
    return {"seasonal_rain_total": kharif_total, "premonsoon_rain": premonsoon_total,
            "as_of_day": int(as_of_day), "narrative": narrative}


# analog tail selection
def extract_preseason_features(met_path: str) -> Dict[str, float]:
    """Jan-May rainfall + Tmax features used for the week-0 analog search."""
    df = load_met_file(met_path)
    if df.empty:
        return None
    year = int(df["year"].iloc[0])
    df["month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["day"].astype(str), format="%Y-%j").dt.month
    return {
        "Year": year,
        "Rain_Jan": float(df[df.month == 1]["rain"].sum()), "Rain_Feb": float(df[df.month == 2]["rain"].sum()),
        "Rain_Mar": float(df[df.month == 3]["rain"].sum()), "Rain_Apr": float(df[df.month == 4]["rain"].sum()),
        "Rain_May": float(df[df.month == 5]["rain"].sum()), "Tmax_Jan_May": float(df[df.month <= 5]["maxt"].mean()),
    }


def extract_in_season_features(weather_df: pd.DataFrame, sow_day: int, current_day: int) -> Dict[str, float]:
    """In-season rainfall / dry-spell features used for the analog search from week 1 onward."""
    season = weather_df[(weather_df.day >= sow_day) & (weather_df.day <= current_day)].copy()
    if season.empty:
        return {"CumRain": 0.0, "RainyDays": 0, "MaxDrySpell": 0, "Tmax_Mean": 0.0}
    season["is_dry"] = season["rain"] == 0
    season["dry_streak"] = season["is_dry"] * (season.groupby((~season["is_dry"]).cumsum()).cumcount() + 1)
    return {
        "CumRain": float(season["rain"].sum()),
        "RainyDays": int((season["rain"] > 0).sum()),
        "MaxDrySpell": int(season["dry_streak"].max()),
        "Tmax_Mean": float(season["maxt"].mean()),
    }


def build_historical_feature_table(all_weather: Dict[int, pd.DataFrame], oni: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year, df in all_weather.items():
        df = df.copy()
        df["month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["day"].astype(str), format="%Y-%j").dt.month
        oni_val = 0.0
        match = oni[oni.Year == year]
        if not match.empty and not pd.isna(match["MAM"].values[0]):
            oni_val = float(match["MAM"].values[0])
        records.append({
            "Year": year, "ONI": oni_val,
            "Rain_Jan": float(df[df.month == 1]["rain"].sum()), "Rain_Feb": float(df[df.month == 2]["rain"].sum()),
            "Rain_Mar": float(df[df.month == 3]["rain"].sum()), "Rain_Apr": float(df[df.month == 4]["rain"].sum()),
            "Rain_May": float(df[df.month == 5]["rain"].sum()), "Tmax_Jan_May": float(df[df.month <= 5]["maxt"].mean()),
        })
    return pd.DataFrame(records)


def rank_analog_years(historical: pd.DataFrame, target_vector: Dict[str, float],
                      features: List[str], weights: List[float]) -> List[int]:
    """Rank historical years by weighted Euclidean distance over z-scored climate features."""
    hist = historical[features]
    mu, sigma = hist.mean(), hist.std().replace(0, 1e-8)
    z_hist = (hist - mu) / sigma
    z_target = (pd.Series(target_vector)[features] - mu) / sigma
    distances = np.sqrt(((z_hist - z_target) ** 2 * weights).sum(axis=1))
    order = distances.sort_values(kind="stable").index
    return [int(historical.loc[i, "Year"]) for i in order]


def select_analog_year(historical: pd.DataFrame, target_vector: Dict[str, float],
                        features: List[str], weights: List[float]) -> int:
    """Single nearest analog year (rank 1 of rank_analog_years)."""
    return rank_analog_years(historical, target_vector, features, weights)[0]


# short-term forecast construction
def degrade_forecast(
    true_forecast: pd.DataFrame,
    rad_cs: pd.Series = None,
    sigma_p: float = 0.5, b: float = 1.0,
    sigma_t: float = 0.93, beta: float = 0.36, mu_t_base: float = 0.5,
    sigma_k: float = 0.1, mu_k_base: float = 0.1,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """Turn a 7-day observed window into a forecast whose error grows with lead time."""
    rng = rng or np.random.default_rng()
    degraded = true_forecast.reset_index(drop=True).copy()
    for col in ("rain", "maxt", "mint", "radn"):
        if col in degraded.columns:
            degraded[col] = degraded[col].astype(float)

    for i in range(len(degraded)):
        tau = i + 1
        rain_true = degraded["rain"].iloc[i]
        tmax_true, tmin_true = degraded["maxt"].iloc[i], degraded["mint"].iloc[i]
        rad_true = degraded["radn"].iloc[i]
        rad_cs_tau = rad_cs.iloc[i] if rad_cs is not None else (rad_true + 1.0)

        is_wet_true = rain_true > 0
        p_flip = 0.05 * (tau ** 0.7)
        flip = rng.random() < p_flip
        is_wet_f = (not is_wet_true) if flip else is_wet_true

        rain_f, delta_rain = 0.0, 0
        if is_wet_f:
            base_rain = rain_true if is_wet_true else 1.0
            variance_factor = (sigma_p ** 2) * (tau ** b)
            shape = 1.0 / variance_factor
            scale = base_rain * variance_factor
            rain_f = rng.gamma(shape, scale)
            if not is_wet_true:
                delta_rain = 1
            else:
                delta_rain = 1 if rain_f > rain_true else (-1 if rain_f < rain_true else 0)
        else:
            delta_rain = -1 if is_wet_true else 0
        degraded.loc[i, "rain"] = rain_f

        mu_t = mu_t_base * tau / 7.0
        eps_tmax = rng.normal(0, sigma_t * (tau ** beta))
        eps_tmin = rng.normal(0, sigma_t * (tau ** beta))
        degraded.loc[i, "maxt"] = tmax_true - delta_rain * mu_t + eps_tmax
        degraded.loc[i, "mint"] = tmin_true - delta_rain * mu_t + eps_tmin

        k_true = (rad_true / rad_cs_tau) if rad_cs_tau > 0 else 0.0
        mu_k = mu_k_base * tau / 7.0
        eps_k = rng.normal(0, sigma_k * (tau ** beta))
        k_f = float(np.clip(k_true - delta_rain * mu_k + eps_k, 0.0, 1.1))
        degraded.loc[i, "radn"] = k_f * rad_cs_tau

    return degraded


# weather array assembly
def build_weekly_weather_array(
    target_year: int,
    sowing_day: int,
    current_day: int,
    season_end_day: int,
    all_weather: Dict[int, pd.DataFrame],
    historical_features: pd.DataFrame,
    target_vector: Dict[str, float],
    features: List[str],
    weights: List[float],
    rad_cs_data: pd.Series = None,
    rng: np.random.Generator = None,
    analog_year: int = None,
) -> Tuple[pd.DataFrame, int]:
    """Assemble W_t = C_pre || O || F || A_hat for the current week; returns the data and the analog year."""
    target_data = all_weather[target_year].copy()

    c_pre = target_data[target_data.day < sowing_day]
    observed = target_data[(target_data.day >= sowing_day) & (target_data.day <= current_day)]

    forecast_start, forecast_end = current_day + 1, current_day + 7
    true_forecast = target_data[(target_data.day >= forecast_start) & (target_data.day <= forecast_end)]
    if not true_forecast.empty:
        rad_cs_slice = rad_cs_data.iloc[forecast_start - 1: forecast_end] if rad_cs_data is not None else None
        forecast_segment = degrade_forecast(true_forecast, rad_cs_slice, rng=rng)
    else:
        forecast_segment = pd.DataFrame()

    if analog_year is None:
        analog_year = select_analog_year(historical_features, target_vector, features, weights)
    analog_start = current_day + 8
    analog_full = all_weather[analog_year].copy()
    analog_tail = analog_full[(analog_full.day >= analog_start) & (analog_full.day <= season_end_day)].copy()
    analog_tail["year"] = target_year  # spoof year for simulator clock continuity

    w_t = pd.concat([c_pre, observed, forecast_segment, analog_tail]).reset_index(drop=True)

    # leap-year mismatch between target and analog years
    target_max_day = target_data.day.max()
    if not w_t.empty and w_t.day.max() < target_max_day:
        pad = w_t.iloc[-1:].copy()
        pad["day"] = target_max_day
        w_t = pd.concat([w_t, pad]).reset_index(drop=True)
    elif not w_t.empty and w_t.day.max() > target_max_day:
        w_t = w_t[w_t.day <= target_max_day].reset_index(drop=True)

    return w_t, analog_year

def build_preseason_ensemble(
    target_year: int,
    window_start_doy: int,
    season_end_day: int,
    all_weather: Dict[int, pd.DataFrame],
    historical_features: pd.DataFrame,
    target_vector: Dict[str, float],
    features: List[str],
    weights: List[float],
    n_members: int,
    rng: np.random.Generator = None,
) -> List[Tuple[pd.DataFrame, int]]:
    """Weather ensemble for the sowing-date search, one member per analog year."""
    ranked = rank_analog_years(historical_features, target_vector, features, weights)[:max(1, n_members)]
    members = []
    for ay in ranked:
        w_t, ay_used = build_weekly_weather_array(
            target_year, window_start_doy, window_start_doy - 1, season_end_day,
            all_weather, historical_features, target_vector, features, weights,
            rng=rng, analog_year=ay,
        )
        members.append((w_t, ay_used))
    return members
