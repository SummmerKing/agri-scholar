# Agri-SCHOLAR

Multi-agent, LLM-guided crop advisories for maize in Karnataka, India, grounded in APSIM NextGen crop simulations. Built at the Game Theory Lab, IISc.

Paper: [arXiv:2610.31003](https://arxiv.org/abs/2610.31003) (PDF also in [`paper/`](paper/agri_scholar.pdf)).

Inside the code the system still uses its working name, Agri-SIGRA (for example in result file names). Both names refer to the same pipeline.

## Overview

Five agents run on a rolling weekly cycle over one growing season:

- **Preseason Agent (PSA)** sets land preparation (tillage, mulch, basal DAP) and the season structure: how many urea splits and irrigation events are allowed, minimum doses and season totals.
- **Context Retrieval Agent (CRA)** builds a weather narrative and retrieves Package-of-Practices (PoP) guidance with hybrid BM25 + dense retrieval.
- **Advisory Bounding Agent (ABA)** uses an LLM (Gemma) to propose bounds for this week's urea and irrigation.
- **Optimization Agent (OA)** searches inside those bounds with Optuna, using a risk-adjusted reward.
- **Verification and Feedback Agent (VFA)** simulates each candidate in APSIM, scores it and writes a critique for the ABA.

```
pre-season (once)
    sowing-date search over the window, scored by APSIM
    PSA <-> OA/VFA: land prep and season caps
for each week:
    CRA: weather array + context bundle
    repeat N_REFLECTION_ITERATIONS times:
        ABA proposes bounds
        OA searches inside them; each Optuna trial -> VFA -> APSIM -> reward
        VFA critique goes back to the ABA
    lock the best action that satisfies the season caps
```

Weather for each week is assembled as real weather up to the decision day, a degraded 7-day forecast, and an analog tail from the closest historical years, so the agents never see the realized future.

## Requirements

| | |
| --- | --- |
| Python | 3.10+ on Linux |
| GPU | one per parallel season; Gemma-4-E4B in bf16 needs about 16 GB |
| [APSIM NextGen](https://www.apsim.info/download-apsim/) | the `Models` executable on `PATH`, or `APSIM_MODELS_EXE=/path/to/Models` |
| Hugging Face access to `google/gemma-4-E4B-it` | gated model: accept the licence, then `huggingface-cli login` |
| Disk | about 20 GB, mostly the Gemma weights |

## Setup

```bash
git clone https://github.com/SummmerKing/agri-scholar.git
cd agri-scholar
conda create -n agri-scholar python=3.11 -y
conda activate agri-scholar
pip install -r requirements.txt
```

Run every script from the repository root.

## Running

1. In `run.py`, set `CONFIG["GPU_IDS"]` to GPUs that exist on your machine (for example `[0]`).
2. Edit `jobs_spec` in `main()` to choose the seasons, for example `[("Mandya", 2015), ("Haveri", 2017)]`. Weather is available for Mandya and Haveri in `data/weather_maize_pilot/`.
3. Run it. A season takes several hours, so use `tmux` or `nohup`:

```bash
python run.py
```

Each (location, year) runs in its own process. Results are written to `results/main/`: one JSON per season with the locked actions, weekly advisories, VFA critiques, final yield and reward breakdown, plus a combined JSON. Logs go to `logs/`.

The RAG corpus (`rag_corpus.json`) and its embedding cache are included. To rebuild them, put the PoP PDFs in `Package_Of_Practices/`, run `python build_rag.py` and delete `rag_corpus_embeddings.npy`.

All settings live in the `CONFIG` dict at the top of each run script: season length, cultivar and seed, LLM and retrieval settings, search budgets, sowing window, prices (Rs.) and the shadow costs used in the sustainability penalty.

## Experiments

| Experiment | Script | Results |
| --- | --- | --- |
| Main pipeline | `run.py` | `results/main/` |
| Mandya, 2015-2025 | `run_mandya.py` | `results/mandya/` |
| Haveri, 2015-2025 | `run_haveri.py` | `results/haveri/` |
| Haveri reruns | `run_haveri_rerun.py`, `run_haveri_rerun_2025.py` | `results/haveri_rerun/` |
| Ablation: no LLM, fixed wide bounds | `run_no_aba.py` | `results/ablation_fixed_bounds/` |
| Ablation: no LLM, PoP-derived bounds | `run_pop_bounds.py` | `results/ablation_pop_bounds/` |
| Government PoP baseline | `run_pop_baseline.py` | `results/pop_baseline/` |

The PoP baseline needs no GPU:

```bash
python run_pop_baseline.py --location Mandya --base-file apsimx/Maize_Mandya_pop2.apsimx \
    --years 2015 2016 2017 --weather observed
```

`make_ablation_tables.py` prints the LaTeX tables for the ABA ablation, and `plot_baseline_comparison.py` plots Agri-SCHOLAR yields against government baseline yields.

## Layout

| Path | Contents |
| --- | --- |
| `run.py` | pipeline and experiment settings |
| `preseason_agent.py`, `context_retrieval_agent.py`, `advisory_bounding_agent.py`, `optimization_agent.py`, `verification_feedback_agent.py` | the five agents |
| `reward_model.py` | reward (Eqs. 2-14): revenue, production cost, sustainability penalties |
| `simulation_engine.py` | action -> APSIM calls -> outcome, and the season caps |
| `apsim_controller.py` | APSIM NextGen file editing and execution |
| `weather_array.py` | weather array construction and forecast degradation |
| `llm_backend.py` | shared, seeded Gemma instance |
| `build_rag.py` | PoP PDFs -> `rag_corpus.json` |
| `apsimx/` | calibrated and PoP APSIM models for Mandya and Haveri |
| `data/weather_maize_pilot/` | daily weather (`.met`) per location and year |
| `results/` | outputs of the runs reported in the paper |
| `paper/` | paper PDF |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `CUDA error: invalid device ordinal` | `GPU_IDS` lists GPUs that don't exist on this machine |
| `401` or `gated repo` when loading Gemma | accept the model licence on Hugging Face and log in |
| every simulation fails or yield is 0 | APSIM is not found: check `which Models` or set `APSIM_MODELS_EXE` |
| out of RAM or CPU | lower `OPTUNA_N_JOBS`, `SOW_SEARCH_N_JOBS` or the number of parallel GPUs |
| `scratch_sims/` grows large | safe to delete between runs |
