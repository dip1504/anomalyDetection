import pandas as pd
import os
import argparse
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def diagnose(outdir, groups_csv=None, limit=50):
    diag = pd.read_csv(os.path.join(outdir, 'diagnostics_multi.csv'))
    # select groups with errors or low mean_consensus (as a proxy)
    if 'error' in diag.columns:
        bad = diag[diag['error'].notnull()]
    else:
        bad = diag[diag['mean_consensus'].isnull()]
    if groups_csv:
        bad = bad.merge(pd.read_csv(groups_csv), on=['insight_type','target_key'], how='inner')
    bad = bad.head(limit)
    results = []
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for _, row in bad.iterrows():
        it = row['insight_type']
        tk = row['target_key']
        f = os.path.join(outdir, f'flags_{it}_{tk}_multi.csv')
        if not os.path.exists(f):
            results.append({'insight_type':it,'target_key':tk,'status':'missing_flags'})
            continue
        df = pd.read_csv(f)
        y = df['count'].values
        # re-build a simple design
        X = pd.DataFrame({'const':1, 't': range(len(y))}).values
        try:
            import statsmodels.api as sm
            mod = sm.NegativeBinomial(y, X, loglike_method='nb2')
            res = mod.fit(disp=False, maxiter=200)
            results.append({'insight_type':it,'target_key':tk,'aic':res.aic,'converged':res.mle_retvals.get('converged', True)})
        except Exception as e:
            results.append({'insight_type':it,'target_key':tk,'error':str(e)})
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(outdir, 'fit_diagnostics.csv'), index=False)
    print('Wrote fit_diagnostics.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--groups_csv', default=None)
    args = parser.parse_args()
    diagnose(args.outdir, groups_csv=args.groups_csv)

if __name__ == '__main__':
    main()
