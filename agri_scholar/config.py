"""Default settings for Agri-SCHOLAR runs. Experiment files in configs/ override these."""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "LOCATIONS": ["Mandya"],
    "YEARS": [2023, 2024, 2025],
    "JOBS": None,               # explicit [[location, year], ...]; overrides LOCATIONS x YEARS
    "TOTAL_WEEKS": 18,
    "CULTIVAR": "MAH-14-138",
    "POPULATION_PLANTS_M2": 8.5,
    "SEED": 42,

    "GPU_IDS": [0],
    "INSTANCES_PER_GPU": 1,

    "BOUNDS": "llm",            # llm | fixed | pop  (fixed and pop are the no-LLM ablations)
    "SOWING": "pop_date",       # pop_date: PoP baseline date | onset: monsoon onset + 15 days | window

    "APSIM_BASE_FILES": {
        "Mandya": os.path.join(ROOT, "apsimx", "Maize_Mandya_Caliberated.apsimx"),
        "Haveri": os.path.join(ROOT, "apsimx", "Maize_Haveri_pop2.apsimx"),
    },
    "MET_DIR": os.path.join(ROOT, "data", "weather_maize_pilot"),
    "RAG_CORPUS_PATH": os.path.join(ROOT, "rag_corpus.json"),
    "SCRATCH_DIR": os.path.join(ROOT, "scratch_sims"),
    "RESULTS_DIR": os.path.join(ROOT, "results", "main"),
    "LOG_DIR": os.path.join(ROOT, "logs", "main"),

    "ABA_MODEL_ID": "google/gemma-4-E4B-it",
    "ABA_GPU_ID": 0,
    "ABA_MAX_NEW_TOKENS": 1500,
    "ABA_TEMPERATURE": 0.5,

    "RAG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RAG_ALPHA": 0.5,
    "RAG_TOP_K": 3,
    "RAG_LOCATION_BOOST": 0.25,
    "RAG_REGION_BOOST": 0.10,
    "RAG_MAX_CHUNKS_PER_FILE": 2,

    "N_PRESEASON_REFLECTION_ITERATIONS": 2,
    "N_TRIALS_PRESEASON": 100,   # mulch / DAP / tillage search per PSA iteration (sowing date is locked first)

    "SOW_WINDOW_START": "06-01",          # MM-DD, first allowed sowing day
    "SOW_WINDOW_END": "07-15",            # MM-DD, last allowed sowing day
    "SOW_SEARCH_STRIDE_DAYS": 1,
    "SOW_SEARCH_N_ANALOGS": 3,            # weather ensemble size (top-K analog years)
    "SOW_SEARCH_LAMBDA_RISK": 0.5,
    "SOW_SEARCH_SMOOTH_HALFWIDTH": 1,     # +/-1 candidate neighbours -> 3-point smoothing of the score curve
    "SOW_SEARCH_BASAL_DAP_KG": 60.0,      # neutral management used only while ranking sowing dates
    "SOW_SEARCH_TILLAGE_FRACTION": 0.5,
    "SOW_SEARCH_N_JOBS": 4,               # parallel APSIM runs during the sowing search

    "N_REFLECTION_ITERATIONS": 3,

    "N_TRIALS_LATER_WEEKS": 25,   # every in-season week (week 0 included) searches urea/irrigation only
    "OPTUNA_N_JOBS": 24,
    # receding horizon: plan urea for weeks w..T-1, commit week w
    "N_TRIALS_PER_HORIZON_WEEK": 4,
    "HORIZON_WEEKS": None,
    "FUTURE_UREA_STEP_KG": 5.0,     # coarse dose step for future (uncommitted) weeks
    "MAX_SINGLE_UREA_KG": None,     # per-split ceiling for future weeks; None = derived from season cap / splits

    "GRAIN_PRICE_PER_KG": 20.90,
    "FERTILIZER_PRICE_PER_KG": {
        "Urea": 13.04,      # pure elemental-N-equivalent price for UreaN applications
        "DAP": 27.00,
        "Mulch": 2.00,      # nominal per-kg cost of sourcing/spreading residue mulch
    },
    "WATER_PRICE_PER_MM": 2.50,
    "LABOR_PRICE_PER_OPERATION": 350.0,

    "SHADOW_COST_WATERLOGGING": 5000.0,
    "SHADOW_COST_N_OVERSUPPLY": 2000.0,
    "SHADOW_COST_DEEP_DRAINAGE": 15.0,
    "SHADOW_COST_N_LEACHING": 25.0,
    "SHADOW_COST_SOC_LOSS": 150.0,

    "CROP_FAILURE_PENALTY": 50000.0,

    "ANALOG_FEATURES_WEEK0": ["ONI", "Rain_Jan", "Rain_Feb", "Rain_Mar", "Rain_Apr", "Rain_May", "Tmax_Jan_May"],
    "ANALOG_WEIGHTS_WEEK0": [0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1],
    "ANALOG_FEATURES_LATER": ["ONI", "CumRain", "RainyDays", "MaxDrySpell", "Tmax_Mean"],
    "ANALOG_WEIGHTS_LATER": [0.2, 0.3, 0.2, 0.2, 0.1],

    "SOIL_PROFILES": {
        "Mandya": {
            "order": "Alfisol", "type": "Sandy Loam", "lat": 12.52, "lon": 76.89,
            "bd": [1.45, 1.48, 1.50, 1.50, 1.50, 1.50, 1.50],
            "airdry": [0.05, 0.08, 0.11, 0.12, 0.12, 0.12, 0.12],
            "ll15": [0.10, 0.10, 0.11, 0.12, 0.12, 0.12, 0.12],
            "dul": [0.22, 0.22, 0.24, 0.25, 0.25, 0.25, 0.25],
            "sat": [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
            "ks": [100.0] * 7, "carbon": [0.50, 0.30, 0.20, 0.10, 0.05, 0.05, 0.05],
        },
        "Dharwad": {
            "order": "Vertisol", "type": "Clay", "lat": 15.4589, "lon": 75.0078,
            "bd": [1.15, 1.15, 1.15, 1.15, 1.20, 1.20, 1.20],
            "airdry": [0.15, 0.20, 0.25, 0.28, 0.28, 0.28, 0.28],
            "ll15": [0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28],
            "dul": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
            "sat": [0.54, 0.54, 0.54, 0.54, 0.53, 0.53, 0.53],
            "ks": [10.0] * 7, "carbon": [0.60, 0.40, 0.30, 0.20, 0.10, 0.10, 0.10],
        },
        "Davanagere": {
            "order": "Alfisol", "type": "Red Brown", "lat": 14.4644, "lon": 75.9218,
            "bd": [1.40, 1.40, 1.45, 1.45, 1.45, 1.45, 1.45],
            "airdry": [0.08, 0.10, 0.12, 0.14, 0.14, 0.14, 0.14],
            "ll15": [0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15],
            "dul": [0.28, 0.28, 0.30, 0.30, 0.30, 0.30, 0.30],
            "sat": [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
            "ks": [70.0] * 7, "carbon": [0.45, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05],
        },
        "Haveri": {
            "order": "Inceptisol", "type": "Transitional", "lat": 14.7937, "lon": 75.3983,
            "bd": [1.30, 1.30, 1.35, 1.35, 1.35, 1.35, 1.35],
            "airdry": [0.10, 0.15, 0.20, 0.20, 0.20, 0.20, 0.20],
            "ll15": [0.20, 0.20, 0.22, 0.22, 0.22, 0.22, 0.22],
            "dul": [0.38, 0.38, 0.40, 0.40, 0.40, 0.40, 0.40],
            "sat": [0.48, 0.48, 0.47, 0.47, 0.47, 0.47, 0.47],
            "ks": [50.0] * 7, "carbon": [0.55, 0.35, 0.25, 0.15, 0.05, 0.05, 0.05],
        },
    },
}

# PoP baseline sowing dates, used when SOWING == "pop_date"
POP_SOWING_DATES = {
    ("Mandya", 2015): "06-19", ("Mandya", 2016): "06-03", ("Mandya", 2017): "06-11",
    ("Mandya", 2018): "06-03", ("Mandya", 2019): "06-09", ("Mandya", 2020): "06-30",
    ("Mandya", 2021): "06-04", ("Mandya", 2022): "06-19", ("Mandya", 2023): "07-04",
    ("Mandya", 2024): "06-07", ("Mandya", 2025): "06-12",
    ("Haveri", 2015): "06-19", ("Haveri", 2016): "06-04", ("Haveri", 2017): "06-11",
    ("Haveri", 2018): "06-03", ("Haveri", 2019): "06-12", ("Haveri", 2020): "07-04",
    ("Haveri", 2021): "06-04", ("Haveri", 2022): "06-22", ("Haveri", 2023): "07-04",
    ("Haveri", 2024): "06-05", ("Haveri", 2025): "06-11",
}

_PATH_KEYS = ("MET_DIR", "RAG_CORPUS_PATH", "SCRATCH_DIR", "RESULTS_DIR", "LOG_DIR")


def load_config(path=None, **overrides):
    """DEFAULTS updated with a JSON experiment file and keyword overrides. Relative paths are taken
    from the repository root."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for key in _PATH_KEYS:
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(ROOT, cfg[key])
    cfg["APSIM_BASE_FILES"] = {loc: p if os.path.isabs(p) else os.path.join(ROOT, p)
                               for loc, p in cfg["APSIM_BASE_FILES"].items()}
    return cfg
