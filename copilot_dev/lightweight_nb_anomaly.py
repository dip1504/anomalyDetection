#!/usr/bin/env python3
"""
Enhanced Lightweight Negative Binomial (NB2) anomaly detection for WEEKLY counts.
Model selection per (target_type, insight_type) pair, robust fallback, and change point placeholder.
Robust and debugged version, consistent with original code.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import nbinom
import statsmodels.api as sm
import ruptures as rpt
import matplotlib.pyplot as plt

def ensure_weekly_counts(df):
    df["week_start"] = df["insight_date_time"].dt.to_period("W-MON").dt.start_time.dt.date
    g = (df.groupby(["target_type","insight_type","target_key","week_start"])
            .size().reset_index(name="count"))
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

def build_design(g, fourier_K=1):
    g = g.sort_values("week_start").copy()
    g["t"] = np.arange(len(g), dtype=float)
    P = 52.0
    X = [np.ones(len(g))]
    for k in range(1, fourier_K+1):
        X.append(np.sin(2*np.pi*k*g["t"]/P))
        X.append(np.cos(2*np.pi*k*g["t"]/P))
    X.append(g["t"]/P)  # light trend
    X = np.column_stack(X)
    y = g["count"].astype(int).values
    return g, X, y

def fit_nb2(y, X):
    model = sm.NegativeBinomial(y, X, loglike_method="nb2")
    res = model.fit(disp=False, maxiter=200)
    return res

def predict_pi(res, X, level=0.99):
    params = res.params
    if hasattr(params, "index"):
        # pandas Series
        if "alpha" in params.index:
            beta = params.drop(labels=["alpha"]).values
            alpha = float(params["alpha"])
        else:
            beta = params.values[:-1]
            alpha = float(params.values[-1])
    else:
        beta = params[:-1]
        alpha = float(params[-1])
    mu = np.exp(X @ beta)
    alpha = max(alpha, 1e-12)
    r = 1.0 / alpha
    p = r / (r + mu)
    lower = nbinom.ppf((1-level)/2.0, r, p)
    upper = nbinom.ppf(1 - (1-level)/2.0, r, p)
    return mu, lower, upper

def robust_fallback(y, window=8):
    y = y.astype(float)
    mu = pd.Series(y).rolling(window, min_periods=1).mean().values
    std = pd.Series(y).rolling(window, min_periods=1).std().fillna(0.0).values
    lo = np.maximum(mu - 3*std, 0.0)
    hi = mu + 3*std
    return mu, lo, hi

def select_best_nb2_model(y, X_list, level=0.99):
    best_res = None
    best_X = None
    best_aic = np.inf
    for X in X_list:
        try:
            res = fit_nb2(y, X)
            if hasattr(res, 'aic') and res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_X = X
        except Exception:
            continue
    return best_res, best_X

def detect_change_points(y, pen=10):
    # Use ruptures Pelt search for a single change point (mean shift, l2 cost)
    y_trans = np.log1p(y)  # log1p for count data stability
    algo = rpt.Pelt(model="l2").fit(y_trans)
    result = algo.predict(pen=pen)
    cps = [cp for cp in result if cp < len(y)]
    return cps

def plot_series(g, y, mu, lo, hi, anomalies, change_points, title, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(g["week_start"], y, label="Actual", marker="o")
    plt.plot(g["week_start"], mu, label="Predicted", linestyle="--")
    plt.fill_between(g["week_start"], lo, hi, color="gray", alpha=0.2, label="PI")
    plt.scatter(g["week_start"].iloc[anomalies], y[anomalies], color="red", label="Anomaly", zorder=5)
    for cp in change_points:
        plt.axvline(g["week_start"].iloc[cp], color="purple", linestyle=":", label="Change Point" if cp == change_points[0] else None)
    plt.title(title)
    plt.xlabel("Week Start")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="out_lightweight")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--max_keys", type=int, default=None)
    ap.add_argument("--fourier_K", type=int, default=1)
    ap.add_argument("--pi_level", type=float, default=0.99)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["insight_date_time"])
    wk = ensure_weekly_counts(df)

    pairs = wk[["target_type","insight_type"]].drop_duplicates().sort_values(["target_type","insight_type"])
    if args.max_pairs:
        pairs = pairs.head(args.max_pairs)

    per_key_records = []
    pair_records = []
    diag_records = []
    plot_dir = os.path.join(args.outdir, "diagnostics_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for _, pr in pairs.iterrows():
        t, ins = pr["target_type"], pr["insight_type"]
        sub = wk[(wk["target_type"]==t) & (wk["insight_type"]==ins)].copy()

        # Pair aggregate
        pair_week = sub.groupby("week_start")["count"].sum().reset_index()
        yP = pair_week["count"].values

        # Change point detection
        cps = detect_change_points(yP)
        seg_starts = [0] + cps
        seg_ends = cps + [len(yP)]
        muP_full = np.zeros_like(yP, dtype=float)
        loP_full = np.zeros_like(yP, dtype=float)
        hiP_full = np.zeros_like(yP, dtype=float)
        fallback_used = False
        for seg_start, seg_end in zip(seg_starts, seg_ends):
            seg = pair_week.iloc[seg_start:seg_end].copy()
            candidate_X = []
            for K in range(0, args.fourier_K+1):
                gP, XP, yPseg = build_design(seg, fourier_K=K)
                candidate_X.append(XP)
            resP, best_XP = select_best_nb2_model(yPseg, candidate_X, args.pi_level)
            if resP is not None:
                muP, loP, hiP = predict_pi(resP, best_XP, args.pi_level)
                aic = resP.aic
            else:
                muP, loP, hiP = robust_fallback(yPseg)
                aic = None
                fallback_used = True
            muP_full[seg_start:seg_end] = muP
            loP_full[seg_start:seg_end] = loP
            hiP_full[seg_start:seg_end] = hiP
            diag_records.append({
                "target_type": t,
                "insight_type": ins,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "aic": aic,
                "fallback": fallback_used,
                "change_point": seg_start if seg_start > 0 else None
            })
        anomalies = np.where((yP < loP_full) | (yP > hiP_full))[0]
        for i in range(len(pair_week)):
            pair_records.append({
                "target_type": t,
                "insight_type": ins,
                "week_start": str(pair_week.iloc[i]["week_start"]),
                "count": int(yP[i]),
                "mu_hat": float(muP_full[i]),
                "pi_low": float(loP_full[i]),
                "pi_high": float(hiP_full[i]),
                "is_anom": bool(i in anomalies)
            })
        # Plot
        plot_path = os.path.join(plot_dir, f"pair_{t}_{ins}.png")
        plot_series(pair_week, yP, muP_full, loP_full, hiP_full, anomalies, cps, f"Pair: {t}, {ins}", plot_path)

        # Keys
        keys = sub["target_key"].drop_duplicates().sort_values()
        if args.max_keys:
            keys = keys.head(args.max_keys)
        for key in keys:
            gk = sub[sub["target_key"]==key][["week_start","count"]].copy().sort_values("week_start")
            y = gk["count"].values
            if (y > 0).sum() < 4:
                continue
            cps_k = detect_change_points(y)
            seg_starts_k = [0] + cps_k
            seg_ends_k = cps_k + [len(y)]
            mu_full = np.zeros_like(y, dtype=float)
            lo_full = np.zeros_like(y, dtype=float)
            hi_full = np.zeros_like(y, dtype=float)
            fallback_used_k = False
            for seg_start, seg_end in zip(seg_starts_k, seg_ends_k):
                seg = gk.iloc[seg_start:seg_end].copy()
                candidate_Xk = []
                for K in range(0, args.fourier_K+1):
                    gX, X, yseg = build_design(seg, fourier_K=K)
                    candidate_Xk.append(X)
                res, best_X = select_best_nb2_model(yseg, candidate_Xk, args.pi_level)
                if res is not None:
                    mu, lo, hi = predict_pi(res, best_X, args.pi_level)
                    aic = res.aic
                else:
                    mu, lo, hi = robust_fallback(yseg)
                    aic = None
                    fallback_used_k = True
                mu_full[seg_start:seg_end] = mu
                lo_full[seg_start:seg_end] = lo
                hi_full[seg_start:seg_end] = hi
                diag_records.append({
                    "target_type": t,
                    "insight_type": ins,
                    "target_key": key,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "aic": aic,
                    "fallback": fallback_used_k,
                    "change_point": seg_start if seg_start > 0 else None
                })
            anomalies_k = np.where((y < lo_full) | (y > hi_full))[0]
            for i in range(len(gk)):
                per_key_records.append({
                    "target_type": t,
                    "insight_type": ins,
                    "target_key": key,
                    "week_start": str(gk.iloc[i]["week_start"]),
                    "count": int(y[i]),
                    "mu_hat": float(mu_full[i]),
                    "pi_low": float(lo_full[i]),
                    "pi_high": float(hi_full[i]),
                    "is_anom": bool(i in anomalies_k)
                })
            # Plot
            plot_path_k = os.path.join(plot_dir, f"key_{t}_{ins}_{key}.png")
            plot_series(gk, y, mu_full, lo_full, hi_full, anomalies_k, cps_k, f"Key: {t}, {ins}, {key}", plot_path_k)
    diag_out = os.path.join(args.outdir, "diagnostics.csv")
    pd.DataFrame.from_records(diag_records).to_csv(diag_out, index=False)
    per_key_out = os.path.join(args.outdir, "light_nb_flags_by_key_weekly.csv")
    pair_out = os.path.join(args.outdir, "light_nb_flags_by_pair_weekly.csv")
    pd.DataFrame.from_records(per_key_records).to_csv(per_key_out, index=False)
    pd.DataFrame.from_records(pair_records).to_csv(pair_out, index=False)
    print("Saved:", diag_out, per_key_out, "and", pair_out)

if __name__ == "__main__":
    main()
