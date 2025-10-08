import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy
import subprocess

def inject_anomalies(df, n_inject=5, mag=3.0, seed=42):
    np.random.seed(seed)
    df = df.copy()
    groups = list(df.groupby(['insight_type','target_key']))
    picks = groups[:min(10, len(groups))]
    gold = []
    for (it,tk), g in picks:
        idx = g.sample(n=min(n_inject, len(g)), random_state=seed).index
        for i in idx:
            df.loc[i, 'count'] = int(df.loc[i,'count'] * mag + 1)
            gold.append({'insight_type':it,'target_key':tk,'week_start':df.loc[i,'week_start']})
    return df, pd.DataFrame(gold)

def run_pipeline(input_csv, outdir, workers=1):
    cmd = ['py','-3','y:\\Study\\repos\\adm\\anomalyDetection\\adm_odm\\odm_nb_anomaly.py','--input', input_csv,'--outdir', outdir,'--workers',str(workers)]
    print('Running pipeline:', ' '.join(cmd))
    subprocess.check_call(cmd)

def score(outdir, gold):
    # read all flags and match to gold
    flags = []
    for f in os.listdir(outdir):
        if f.startswith('flags_') and f.endswith('_multi.csv'):
            d = pd.read_csv(os.path.join(outdir,f), parse_dates=['week_start'])
            flags.append(d)
    allf = pd.concat(flags, ignore_index=True)
    allf['is_ensemble'] = allf['ensemble_is_anom'].astype(bool)
    merged = allf.merge(gold.assign(week_start=lambda x: pd.to_datetime(x['week_start'])), on=['insight_type','target_key','week_start'], how='left', indicator=True)
    tp = ((merged['is_ensemble']) & (merged['_merge']=='both')).sum()
    fp = ((merged['is_ensemble']) & (merged['_merge']=='left_only')).sum()
    fn = ((~merged['is_ensemble']) & (merged['_merge']=='both')).sum()
    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
    print(f'Precision: {prec:.3f}, Recall: {rec:.3f}, TP={tp}, FP={fp}, FN={fn}')
    return {'precision':prec,'recall':rec,'tp':int(tp),'fp':int(fp),'fn':int(fn)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=['week_start'])
    df_inj, gold = inject_anomalies(df)
    tmp_input = os.path.join(args.outdir, 'injected_input.csv')
    df_inj.to_csv(tmp_input, index=False)
    run_pipeline(tmp_input, args.outdir, workers=args.workers)
    metrics = score(args.outdir, gold)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, 'eval_metrics.csv'), index=False)
    print('Wrote eval_metrics.csv')

if __name__ == '__main__':
    main()
