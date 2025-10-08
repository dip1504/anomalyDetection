# Anomaly Visualization — Documentation

This document explains the crisp anomaly visualization produced by `make_crisp_plots.py` and the design decisions.

Files produced
- `diagnostics_crisp_<insight>_<target_key>.png` — high-resolution (dpi=200) plots for each `flags_*_multi.csv` series.
- `crisp_plots_summary.csv` — a CSV listing `flags_file`, `plot`, and `ensemble_anomalies` for each produced plot.

How to run
1. Ensure `adm_odm/out_odm/` contains `flags_*_multi.csv` files (produced by `odm_nb_anomaly.py`).
2. Run:

   py -3 make_crisp_plots.py --outdir y:\Study\repos\adm\anomalyDetection\adm_odm\out_odm --topn 100

Design rationale
- The visualization emphasizes ensemble anomalies (star marker, large, red) and annotates each with the consensus count. This makes it quick to find high-confidence anomalies.
- Background model predictions (NB2 and Poisson) are plotted with subdued fill for predictive intervals to show uncertainty bands.
- A secondary axis shows `consensus_count` as translucent purple bars to indicate how many detectors agreed per week.
- The plot uses a whitegrid style, large size, and 200 dpi to be clear in reports and slide decks.

Data fields used
- Observed: `count`
- NB2 Pred: `mu_nb2`, `pi_low_nb2`, `pi_high_nb2`
- Poisson Pred: `mu_pois`, `pi_low_pois`, `pi_high_pois`
- Consensus: `consensus_count`
- Ensemble anomaly boolean: `ensemble_is_anom`
- Ranking metric: `sum_norm_dist`

Notes & edge cases
- If `week_start` cannot be parsed, the script will coerce and may skip that series.
- If predicted fields are missing, the script still draws the observed series and consensus bars.
- The `--topn` flag allows processing only the top N series by `sum_norm_dist` to save time.

Next steps / extensions
- Add thumbnails and an HTML index for quick browsing of plots.
- Integrate plot generation into `odm_nb_anomaly.py` to save both standard and crisp plots in one run.
- Add interactive Plotly versions for zoom/pan.

Contact
- For questions about the underlying detection logic see `odm_nb_anomaly.py` and `PROCESS_DOCUMENTATION.md`.
