#!/usr/bin/env python3
"""
Hierarchical Negative Binomial (NB2) with seasonality (Fourier) + trend (Gaussian Random Walk)
for weekly counts per (target_type, insight_type, target_key).

Outputs:
  1) per-key anomaly flags with 99% prediction intervals
  2) per-pair (target_type, insight_type) aggregate anomaly flags
  3) distribution-shift metrics & flags per pair (active keys, concentration, KL divergence)

Usage:
  python hierarchical_nb_pymc.py --input synthetic_insights_weekly_1y.csv --outdir out --draws 800 --tune 800 --cores 2
"""

import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import entropy

# ---------------------- Helpers ----------------------

def make_weekly_counts(df):
    """Return weekly counts per (target_type, insight_type, target_key, week_start)."""
    df["week_start"] = df["insight_date_time"].dt.to_period("W-MON").dt.start_time.dt.date
    g = (df.groupby(["target_type","insight_type","target_key","week_start"])
            .size().reset_index(name="count"))
    # ensure continuity per series
    def fill_weeks(grp):
        grp = grp.sort_values("week_start").copy()
        allw = pd.period_range(grp["week_start"].min(), grp["week_start"].max(), freq="W-MON").to_timestamp().date
        full = pd.DataFrame({"week_start": allw})
        out = full.merge(grp, on="week_start", how="left")
        out[["target_type","insight_type","target_key"]] = out[["target_type","insight_type","target_key"]].ffill().bfill()
        out["count"] = out["count"].fillna(0).astype(int)
        return out
    return (g.groupby(["target_type","insight_type","target_key"], group_keys=False)
             .apply(fill_weeks).reset_index(drop=True))

def fourier_features(T, K=2, period=52.0):
    """Return Tx(2K) Fourier design matrix with columns [sin1, cos1, sin2, cos2, ...]."""
    t = np.arange(T, dtype=float)
    cols = []
    for k in range(1, K+1):
        cols.append(np.sin(2*np.pi*k*t/period))
        cols.append(np.cos(2*np.pi*k*t/period))
    X = np.vstack(cols).T  # shape T x 2K
    return X

def distribution_shift_metrics(mat_counts):
    """
    Given TxK weekly counts matrix for a pair, compute pair-level distribution metrics each week:
      - active_keys
      - top_share
      - HHI (Herfindahl-Hirschman index)
      - KL divergence vs rolling 8-week reference (smoothed)
    Returns dataframe with these metrics by week index.
    """
    eps = 1e-9
    total = mat_counts.sum(axis=1, keepdims=True) + eps
    P = mat_counts / total  # TxK distributions
    active = (mat_counts > 0).sum(axis=1)
    top_share = P.max(axis=1)
    hhi = (P**2).sum(axis=1)

    # rolling 8-week reference (exclude current week)
    T = P.shape[0]
    kl = np.zeros(T)
    for t in range(T):
        s = max(0, t-8)
        ref = P[s:t, :].mean(axis=0) if t > s else P[:t+1, :].mean(axis=0)
        ref = ref / (ref.sum() + eps)
        pt = P[t, :] / (P[t, :].sum() + eps)
        ref = np.clip(ref, eps, 1.0)
        pt = np.clip(pt, eps, 1.0)
        kl[t] = entropy(pt, ref)  # KL(P||Q)

    return active, top_share, hhi, kl

def robust_z(x, window=12):
    """Rolling robust z-score using median & MAD; uses past window."""
    x = np.asarray(x, dtype=float)
    z = np.zeros_like(x)
    for i in range(len(x)):
        s = max(0, i-window)
        ref = x[s:i+1]
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) + 1e-9
        z[i] = 0.6745 * (x[i] - med) / mad
    return z

# ---------------------- Model & Inference ----------------------

def fit_pair_hier_nb(pair_df, fourier_K=2, draws=1000, tune=1000, chains=2, cores=2, target_accept=0.9, random_seed=42):
    """
    Fit a hierarchical NB2 model for a single (target_type, insight_type) pair across its keys.
    y_{t,k} ~ NB(mu_{t,k}, alpha)
    log mu_{t,k} = a_pair + b_key[k] + F_t dot beta_fourier + RW_t
    RW_t ~ GaussianRandomWalk(sigma_rw)
    """
    # Pivot to T x K matrix of counts
    keys_order = sorted(pair_df["target_key"].unique())
    weeks_order = sorted(pair_df["week_start"].unique())
    mat = (pair_df
           .pivot_table(index="week_start", columns="target_key", values="count", fill_value=0)
           .reindex(index=weeks_order, columns=keys_order, fill_value=0)
           .values.astype("int64"))
    T, K = mat.shape

    # Fourier features (shared across keys within pair)
    F = fourier_features(T, K=fourier_K, period=52.0)  # T x (2Kf)

    with pm.Model() as m:
        # Priors
        a = pm.Normal("a", 0.0, 1.0)                                # pair intercept
        sigma_key = pm.HalfNormal("sigma_key", 1.0)
        b_key = pm.Normal("b_key", 0.0, sigma_key, shape=K)         # key random intercepts

        beta = pm.Normal("beta", 0.0, 0.5, shape=F.shape[1])        # Fourier coefficients

        sigma_rw = pm.HalfNormal("sigma_rw", 0.3)
        rw = pm.GaussianRandomWalk("rw", sigma_rw, shape=T)         # trend over weeks

        alpha = pm.HalfNormal("alpha", 2.0)                         # NB overdispersion

        # Linear predictor eta[t,k] = a + b_key[k] + F[t]@beta + rw[t]
        eta = (a
               + b_key[None, :]                # 1 x K
               + pm.math.dot(F, beta)[:, None] # T x 1
               + rw[:, None])                  # T x 1
        mu = pm.math.exp(eta)                  # T x K

        # Likelihood
        y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=mat)

        # Sample
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=random_seed, progressbar=True)

        # Posterior predictive
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"], random_seed=random_seed, progressbar=True)

    return mat, weeks_order, keys_order, idata, ppc


def flags_from_ppc(mat, weeks, keys, ppc, level=0.99):
    ydraws = ppc["y"]  # draws x T x K
    mu_hat = ydraws.mean(axis=0)               # T x K
    low = np.quantile(ydraws, (1-level)/2.0, axis=0)
    high = np.quantile(ydraws, 1-(1-level)/2.0, axis=0)

    T, K = mat.shape
    records = []
    for ti in range(T):
        for ki in range(K):
            obs = int(mat[ti, ki])
            records.append({
                "week_start": str(weeks[ti]),
                "target_key": keys[ki],
                "count": obs,
                "mu_hat": float(mu_hat[ti, ki]),
                "pi_low": float(low[ti, ki]),
                "pi_high": float(high[ti, ki]),
                "is_anom": bool((obs < low[ti, ki]) or (obs > high[ti, ki])),
            })
    return pd.DataFrame.from_records(records)


def pair_aggregate_from_ppc(ppc, mat, weeks):
    ydraws = ppc["y"]  # draws x T x K
    draws_sum = ydraws.sum(axis=2)  # draws x T
    mu_hat = draws_sum.mean(axis=0)
    low = np.quantile(draws_sum, 0.005, axis=0)
    high = np.quantile(draws_sum, 0.995, axis=0)
    obs_pair = mat.sum(axis=1)  # T
    out = pd.DataFrame({
        "week_start": [str(w) for w in weeks],
        "count": obs_pair,
        "mu_hat": mu_hat,
        "pi_low": low,
        "pi_high": high
    })
    out["is_anom"] = (out["count"] < out["pi_low"]) | (out["count"] > out["pi_high"])
    return out

def compute_distribution_shift(pair_df):
    keys_order = sorted(pair_df["target_key"].unique())
    weeks_order = sorted(pair_df["week_start"].unique())
    mat = (pair_df
           .pivot_table(index="week_start", columns="target_key", values="count", fill_value=0)
           .reindex(index=weeks_order, columns=keys_order, fill_value=0)
           .values.astype("int64"))
    active, top_share, hhi, kl = distribution_shift_metrics(mat)
    dfm = pd.DataFrame({
        "week_start": [str(w) for w in weeks_order],
        "active_keys": active,
        "top_share": top_share,
        "hhi": hhi,
        "kl_div": kl
    })
    for col in ["active_keys", "top_share", "hhi", "kl_div"]:
        dfm[f"{col}_rz"] = robust_z(dfm[col].values, window=12)
    dfm["is_shift_anom"] = (
        (np.abs(dfm["top_share_rz"]) >= 4)
        | (np.abs(dfm["hhi_rz"]) >= 4)
        | (np.abs(dfm["kl_div_rz"]) >= 4)
    )
    return dfm

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--outdir", type=str, default="out_hier_nb")
    p.add_argument("--fourier_K", type=int, default=2)
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--tune", type=int, default=800)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--pi_level", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["insight_date_time"])
    wk = make_weekly_counts(df)

    per_key_out = []
    pair_out = []
    shift_out = []

    pairs = wk[["target_type","insight_type"]].drop_duplicates().to_records(index=False).tolist()
    for (t, ins) in pairs:
        pair_df = wk[(wk["target_type"] == t) & (wk["insight_type"] == ins)].copy()
        if pair_df["week_start"].nunique() < 20:
            continue

        try:
            mat, weeks, keys, idata, ppc = fit_pair_hier_nb(
                pair_df,
                fourier_K=args.fourier_K,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                random_seed=args.seed
            )
        except Exception as e:
            print(f"[WARN] Pair ({t}, {ins}) failed to sample: {e}")
            continue

        # Per-key flags
        fdf = flags_from_ppc(mat, weeks, keys, ppc, level=args.pi_level)
        fdf.insert(0, "insight_type", ins)
        fdf.insert(0, "target_type", t)
        per_key_out.append(fdf)

        # Pair aggregate flags
        p_agg = pair_aggregate_from_ppc(ppc, mat, weeks)
        p_agg.insert(0, "insight_type", ins)
        p_agg.insert(0, "target_type", t)
        pair_out.append(p_agg[["target_type","insight_type","week_start","count","mu_hat","pi_low","pi_high","is_anom"]])

        # Distribution shift
        sdf = compute_distribution_shift(pair_df)
        sdf.insert(0, "insight_type", ins)
        sdf.insert(0, "target_type", t)
        shift_out.append(sdf)

    if per_key_out:
        pd.concat(per_key_out, ignore_index=True)\
          .to_csv(os.path.join(args.outdir, "pymc_nb_flags_by_key_weekly.csv"), index=False)
    if pair_out:
        pd.concat(pair_out, ignore_index=True)\
          .to_csv(os.path.join(args.outdir, "pymc_nb_flags_by_pair_weekly.csv"), index=False)
    if shift_out:
        pd.concat(shift_out, ignore_index=True)\
          .to_csv(os.path.join(args.outdir, "pymc_distribution_shift_metrics.csv"), index=False)

    print("Done. Outputs saved to", args.outdir)

if __name__ == "__main__":
    main()
