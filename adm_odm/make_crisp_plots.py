"""
Produce clear, high-resolution anomaly visualizations from per-group `flags_*_multi.csv` files
Saved images: `diagnostics_crisp_<insight>_<key>.png` in the same outdir.
Also writes `crisp_plots_summary.csv` listing produced plots and basic counts.

Usage:
  py -3 make_crisp_plots.py --outdir <path/to/out_odm> [--topn N]

Options:
  --outdir  required, directory containing `flags_*_multi.csv` files
  --topn    optional, process only top N groups sorted by max `sum_norm_dist` (default: all)

The script is defensive and skips files it cannot parse.
"""
import os, argparse, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def make_plot(df, outpath, insight_type, target_key):
    # ensure week_start is datetime
    if not np.issubdtype(df['week_start'].dtype, np.datetime64):
        try:
            df['week_start'] = pd.to_datetime(df['week_start'])
        except Exception:
            df['week_start'] = pd.to_datetime(df['week_start'], errors='coerce')
    df = df.sort_values('week_start')
    x = df['week_start']
    y = df['count'].values

    mu_nb2 = df.get('mu_nb2')
    pi_low_nb2 = df.get('pi_low_nb2')
    pi_high_nb2 = df.get('pi_high_nb2')

    mu_pois = df.get('mu_pois')
    pi_low_pois = df.get('pi_low_pois')
    pi_high_pois = df.get('pi_high_pois')

    consensus = df.get('consensus_count', pd.Series(np.zeros(len(df))))
    sum_norm = df.get('sum_norm_dist', pd.Series(np.zeros(len(df))))
    ensemble = df.get('ensemble_is_anom', pd.Series([False]*len(df)))

    # prefer seaborn theme if available, otherwise fall back to matplotlib built-in styles
    try:
        import seaborn as sns
        sns.set_theme(style='whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(14,6), dpi=200)

    # Observed line
    ax.plot(x, y, color='black', lw=1.5, label='Observed', zorder=2)
    ax.scatter(x, y, color='black', s=20, zorder=3)

    # NB2 prediction and PI
    if mu_nb2 is not None:
        ax.plot(x, mu_nb2, color='#1f77b4', linestyle='--', lw=1.2, label='NB2 Pred')
        ax.fill_between(x, pi_low_nb2, pi_high_nb2, color='#1f77b4', alpha=0.15, label='NB2 PI')

    # Poisson prediction and PI (subtle)
    if mu_pois is not None:
        ax.plot(x, mu_pois, color='#2ca02c', linestyle=':', lw=1.0, label='Poisson Pred')
        ax.fill_between(x, pi_low_pois, pi_high_pois, color='#2ca02c', alpha=0.10, label='Poisson PI')

    # Highlight ensemble anomalies strongly
    if ensemble is not None:
        anom_idx = np.where(ensemble.astype(bool))[0]
        if len(anom_idx) > 0:
            ax.scatter(x.iloc[anom_idx], y[anom_idx], s=150, marker='*', color='red', edgecolors='darkred', zorder=5, label='Ensemble Anomaly')
            # annotate with consensus_count
            for i in anom_idx:
                ax.annotate(str(int(consensus.iloc[i])),
                            (x.iloc[i], y[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='darkred')

    # secondary axis: consensus_count as bar chart below
    ax2 = ax.twinx()
    ax2.bar(x, consensus, width=5, alpha=0.12, color='purple', label='Consensus Count')
    ax2.set_ylabel('Consensus Count', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax.set_ylabel('Weekly Count')
    ax.set_title(f"{insight_type} - {target_key}", fontsize=12)

    # neat legend (combine handles)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels + labels2, handles + handles2))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

    # tight layout and save
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--topn', type=int, default=0, help='If >0, limit to top N groups by max(sum_norm_dist)')
    args = parser.parse_args()

    flags_glob = os.path.join(args.outdir, 'flags_*_multi.csv')
    files = glob.glob(flags_glob)
    if not files:
        print('No flags files found in', args.outdir)
        return

    summary_rows = []
    # optionally rank files by max sum_norm_dist
    rank = []
    for f in files:
        try:
            df = pd.read_csv(f)
            maxs = df['sum_norm_dist'].max() if 'sum_norm_dist' in df.columns else 0.0
            rank.append((f, maxs))
        except Exception:
            rank.append((f, 0.0))
    rank = sorted(rank, key=lambda x: x[1], reverse=True)
    if args.topn and args.topn > 0:
        rank = rank[:args.topn]

    for f, _ in rank:
        try:
            df = pd.read_csv(f)
            # try to infer insight and key from filename
            base = os.path.basename(f)
            name = base[len('flags_'):-len('_multi.csv')]
            if '_' in name:
                insight_type, target_key = name.split('_', 1)
            else:
                insight_type = name
                target_key = ''
            outpng = os.path.join(args.outdir, f'diagnostics_crisp_{insight_type}_{target_key}.png')
            make_plot(df, outpng, insight_type, target_key)
            n_ensemble = int(df['ensemble_is_anom'].sum()) if 'ensemble_is_anom' in df.columns else 0
            summary_rows.append({'flags_file': os.path.basename(f), 'plot': os.path.basename(outpng), 'ensemble_anomalies': n_ensemble})
            print('Wrote', outpng)
        except Exception as e:
            print('Skipping', f, 'due to', e)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, 'crisp_plots_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print('Wrote', summary_csv)

if __name__ == '__main__':
    main()
