# Run report

Run timestamp: 2025-10-08 03:50:59.995807

Actions performed:
- Cleared `out_odm` directory
- Re-ran `odm_nb_anomaly.py` sequentially (workers=1) on `synthetic_insights_weekly_1y_preprocessed.csv`
- Generated crisp plots for top 100 groups using `make_crisp_plots.py`

Summary statistics:
- flags files produced: 150
- crisp plots produced: 100
- diagnostics_multi rows: 150
- total ensemble anomalies (sum across flags): 419

Notes and warnings:
- Attempting parallel run with --workers 2 failed due to multiprocessing pickling of a nested function `process_group` (fall back to sequential run).
- Some statsmodels fits produced ConvergenceWarning or HessianInversionWarning; NB fallback logic applied when PI computation failed.
- make_crisp_plots.py uses plot style fallbacks; seaborn optional.
