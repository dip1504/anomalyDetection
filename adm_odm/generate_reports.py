import os
import pandas as pd
import argparse

def top_n_anomalies(outdir, n=100, by='sum_norm_dist'):
    # read all flags_*_multi.csv
    files = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.startswith('flags_') and f.endswith('_multi.csv')]
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=['week_start'])
            d['source_file'] = os.path.basename(f)
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        print('No flags files found')
        return None
    allf = pd.concat(dfs, ignore_index=True)
    allf = allf.sort_values(by=by, ascending=False)
    out = allf.head(n)
    out_path = os.path.join(outdir, f'top_{n}_anomalies_by_{by}.csv')
    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(out)} rows)')
    return out_path

def problematic_groups(outdir):
    diag = pd.read_csv(os.path.join(outdir, 'diagnostics_multi.csv'))
    if 'error' in diag.columns:
        errs = diag[diag['error'].notnull()]
        print(f'Found {len(errs)} groups with errors')
        return errs
    print('No error column in diagnostics')
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--topn', type=int, default=100)
    args = parser.parse_args()
    top_n_anomalies(args.outdir, n=args.topn)
    errs = problematic_groups(args.outdir)
    if not errs.empty:
        errs.to_csv(os.path.join(args.outdir, 'problematic_groups.csv'), index=False)
        print('Wrote problematic_groups.csv')

if __name__ == '__main__':
    main()
