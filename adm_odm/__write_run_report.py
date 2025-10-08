import glob, pandas as pd, os
out='y:/Study/repos/adm/anomalyDetection/adm_odm/out_odm'
flags = glob.glob(os.path.join(out,'flags_*_multi.csv'))
crisp = glob.glob(os.path.join(out,'diagnostics_crisp_*.png'))
# sum ensemble anomalies
total_ens=0
for f in flags:
    try:
        df=pd.read_csv(f)
        if 'ensemble_is_anom' in df.columns:
            total_ens += int(df['ensemble_is_anom'].sum())
    except Exception:
        pass
# counts
flags_count = len(flags)
crisp_count = len(crisp)
# diagnostics_multi rows
diag_csv=os.path.join(out,'diagnostics_multi.csv')
diag_rows=0
if os.path.exists(diag_csv):
    try:
        diag_rows = len(pd.read_csv(diag_csv))
    except Exception:
        diag_rows=0
# write run report
report = os.path.join(out,'RUN_REPORT.md')
with open(report,'w',encoding='utf-8') as f:
    f.write('# Run report\n\n')
    f.write(f'Run timestamp: {pd.Timestamp.now()}\n\n')
    f.write('Actions performed:\n')
    f.write('- Cleared `out_odm` directory\n')
    f.write('- Re-ran `odm_nb_anomaly.py` sequentially (workers=1) on `synthetic_insights_weekly_1y_preprocessed.csv`\n')
    f.write('- Generated crisp plots for top 100 groups using `make_crisp_plots.py`\n\n')
    f.write('Summary statistics:\n')
    f.write(f'- flags files produced: {flags_count}\n')
    f.write(f'- crisp plots produced: {crisp_count}\n')
    f.write(f'- diagnostics_multi rows: {diag_rows}\n')
    f.write(f'- total ensemble anomalies (sum across flags): {total_ens}\n\n')
    f.write('Notes and warnings:\n')
    f.write('- Attempting parallel run with --workers 2 failed due to multiprocessing pickling of a nested function `process_group` (fall back to sequential run).\n')
    f.write('- Some statsmodels fits produced ConvergenceWarning or HessianInversionWarning; NB fallback logic applied when PI computation failed.\n')
    f.write('- make_crisp_plots.py uses plot style fallbacks; seaborn optional.\n')

print('Wrote', report)
