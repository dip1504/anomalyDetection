Process documentation for the ADM_ODM multi-model anomaly detection pipeline

Overview
--------
This document describes the end-to-end processing implemented in the `adm_odm` folder. The pipeline ingests event-level data, aggregates counts per week for each (insight_type, target_key), runs several anomaly detection models, and emits per-pair CSV flags plus diagnostic plots and a diagnostics summary CSV.

Files of interest
-----------------
- `odm_nb_anomaly.py` - main pipeline that fits multiple models and generates outputs.
- `run_odm.py` - small wrapper to run `odm_nb_anomaly.py` with default params.
- `preprocess_input.py` - trivial input passthrough (expandable for cleaning/deduping/timezones).
- `display_multi_model_results.py` - convenience plotting and per-pair display.
- `debug_groups.py` - simple group inspection utility.
- `requirements.txt` - project dependencies (pin versions for reproducibility).
- `synthetic_insights_weekly_1y.csv` - provided synthetic event-level source data.
- `synthetic_insights_weekly_1y_preprocessed.csv` - preprocessed source; used by `run_odm.py`.

High-level pipeline (odm_nb_anomaly.py)
--------------------------------------
1. Read input CSV (expects `insight_date_time` column). The preprocessed CSV includes `target_type` in addition to `insight_type`, `target_key`.
2. Convert timestamps to `week_start` (Monday-based week starts) and aggregate counts per (`insight_type`, `target_key`, `week_start`). Missing weeks are filled with zero counts to maintain continuous series.
3. For each group (insight_type, target_key):
   - Build a design matrix with intercept, Fourier seasonal harmonics (configurable `--fourier_K`), and a linear trend term.
   - Fit multiple models:
     - Negative Binomial GLM (NB2) - `statsmodels`.
     - Poisson GLM - `statsmodels`.
     - STL decomposition residual analysis (z-score).
     - Prophet forecasting (if `prophet` is installed).
     - IsolationForest (distributional outlier detection on counts).
     - Rolling z-score.
   - Detect change points using `ruptures.Pelt` and mark them on diagnostic plots.
   - For parametric models, compute prediction intervals and flag weeks outside PI as anomalies.
   - Compute per-row normalized distances and a consensus `consensus_count` across models.
   - Produce a per-pair CSV `flags_<insight_type>_<target_key>_multi.csv` and a diagnostic plot `diagnostics_<insight_type>_<target_key>_multi.png`.
4. A `diagnostics_multi.csv` is written summarizing AICs, change points, prophet availability, and consensus statistics for each pair.

New features added (summary)
---------------------------
- CLI flags: `--pen`, `--min_nonzero_weeks`, `--min_pi_width`, `--ensemble_thresh`.
- Defensive NB2 PI computation with fallback to approximate PI on numeric issues.
- Minimum PI width enforcement (`--min_pi_width`) to prevent 0-width intervals in small segments.
- Per-row ensemble scoring: `consensus_count`, `nb2_norm_dist`, `pois_norm_dist`, `sum_norm_dist`, and `ensemble_is_anom`.
- Structured logging for run arguments and progress.

How to run
----------
From the `adm_odm` directory run:

```powershell
python run_odm.py
```

Or call the main script directly with more options:

```powershell
python odm_nb_anomaly.py --input synthetic_insights_weekly_1y_preprocessed.csv --outdir out_odm --fourier_K 1 --pi_level 0.99 --pen 10 --min_nonzero_weeks 4 --min_pi_width 0.5 --ensemble_thresh 2
```

Outputs
-------
- `out_odm/flags_<insight_type>_<target_key>_multi.csv` — detailed per-week outputs and anomaly flags from each model, plus consensus columns.
- `out_odm/diagnostics_<insight_type>_<target_key>_multi.png` — diagnostic plot overlaying models and detected anomalies.
- `out_odm/diagnostics_multi.csv` — summary table with AICs, change points and consensus stats.

Evaluation recommendations
--------------------------
- If you have ground-truth labels for injected anomalies, compute precision/recall/F1, detection delay, and per-anomaly detection (did we catch the episode at least once?).
- Perform PI calibration: simulate NB2 series (from fitted params) and verify that empirical coverage matches nominal `pi_level` (e.g., ~1% outside for 0.99).
- Tune `ensemble_thresh` and `pi_level` to balance precision/recall based on your operational tolerance.

Further improvements (roadmap)
------------------------------
- Add parallel processing to speed up per-key fits.
- Implement hierarchical Bayesian pooling across keys (PyMC) to improve statistical strength on sparse keys.
- Add zero-inflated/hurdle models for very sparse series.
- Add an interactive Streamlit dashboard to browse anomalies and diagnostics.
- Pin dependency versions and add a reproducible environment (conda/pip lockfile).

Contact and maintenance
------------------------
- Keep `requirements.txt` pinned and add a few unit tests: `ensure_weekly_counts`, PI calibration, and a small end-to-end smoke test.  
- When updating `statsmodels` or `prophet`, re-run calibration tests as parameter indexing/dispersion outputs may change.

End of document
