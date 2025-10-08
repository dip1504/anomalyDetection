#!/usr/bin/env python3
"""
Multi-model anomaly detection for weekly event counts:
- Negative Binomial GLM
- Poisson GLM
- STL decomposition + residual anomaly detection
"""
import argparse, os, logging
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson, zscore
import statsmodels.api as sm
import ruptures as rpt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

def ensure_weekly_counts(df):
    df["week_start"] = pd.to_datetime(df["insight_date_time"]).dt.to_period("W-MON").dt.start_time.dt.date
    g = (df.groupby(["insight_type","target_key","week_start"]).size().reset_index(name="count"))
    def fill_weeks(grp):
        grp = grp.sort_values("week_start").copy()
        allw = pd.period_range(grp["week_start"].min(), grp["week_start"].max(), freq="W-MON").to_timestamp().date
        full = pd.DataFrame({"week_start": allw})
        out = full.merge(grp, on="week_start", how="left")
        out[["insight_type","target_key"]] = out[["insight_type","target_key"]].ffill().bfill()
        out["count"] = out["count"].fillna(0).astype(int)
        return out
    return (g.groupby(["insight_type","target_key"], group_keys=False).apply(fill_weeks).reset_index(drop=True))

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

def fit_poisson(y, X):
    model = sm.GLM(y, X, family=sm.families.Poisson())
    res = model.fit()
    return res

def predict_nb2(res, X, level=0.99):
    params = res.params
    if hasattr(params, "index"):
        if "alpha" in params.index:
            beta = params.drop(labels=["alpha"]).values
            alpha = float(params["alpha"])
        else:
            beta = params.values[:-1]
            alpha = float(params.values[-1])
    else:
        beta = params[:-1]
        alpha = float(params[-1])
    mu_hat = np.exp(np.dot(X, beta))
    # defensively guard alpha and resulting PI values
    alpha = max(float(alpha), 1e-12)
    r = 1.0 / alpha
    p = r / (r + mu_hat)
    try:
        pi_low = nbinom.ppf((1-level)/2, r, p)
        pi_high = nbinom.ppf(1-(1-level)/2, r, p)
    except Exception:
        # fallback to approximate PI using mean +/- 3*sqrt(var)
        var = mu_hat + alpha * (mu_hat ** 2)
        sd = np.sqrt(np.maximum(var, 0.0))
        pi_low = np.maximum(mu_hat - 3.0 * sd, 0.0)
        pi_high = mu_hat + 3.0 * sd
    # ensure integer-like bounds
    pi_low = np.floor(np.nan_to_num(pi_low, nan=0.0)).astype(float)
    pi_high = np.ceil(np.nan_to_num(pi_high, nan=pi_low + 1.0)).astype(float)
    return mu_hat, pi_low, pi_high

def predict_poisson(res, X, level=0.99):
    mu_hat = np.exp(np.dot(X, res.params))
    pi_low = poisson.ppf((1-level)/2, mu_hat)
    pi_high = poisson.ppf(1-(1-level)/2, mu_hat)
    return mu_hat, pi_low, pi_high

def stl_anomaly(y, period=52, z_thresh=3):
    stl = STL(y, period=period, robust=True)
    res = stl.fit()
    resid = res.resid
    z = zscore(resid, nan_policy='omit')
    is_anom = np.abs(z) > z_thresh
    return res, is_anom, resid

def detect_change_points(y, pen=10):
    algo = rpt.Pelt(model="rbf").fit(y)
    result = algo.predict(pen=pen)
    return result

def rolling_zscore(y, window=8, z_thresh=3):
    s = pd.Series(y)
    roll_mean = s.rolling(window, min_periods=1, center=True).mean()
    roll_std = s.rolling(window, min_periods=1, center=True).std(ddof=0)
    z = (s - roll_mean) / roll_std
    is_anom = z.abs() > z_thresh
    return z.values, is_anom.values

def prophet_anomaly(g, z_thresh=3):
    if Prophet is None:
        return np.full(len(g), np.nan), np.full(len(g), False), np.full(len(g), np.nan), np.full(len(g), np.nan)
    dfp = pd.DataFrame({'ds': pd.to_datetime(g['week_start']), 'y': g['count']})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=0, freq='W-MON')
    forecast = m.predict(future)
    yhat = forecast['yhat'].values
    yhat_lower = forecast['yhat_lower'].values
    yhat_upper = forecast['yhat_upper'].values
    is_anom = (g['count'].values < yhat_lower) | (g['count'].values > yhat_upper)
    return yhat, yhat_lower, yhat_upper, is_anom

def isolation_forest_anomaly(y):
    if len(y) < 8:
        return np.full(len(y), False)
    X = np.array(y).reshape(-1, 1)
    clf = IsolationForest(contamination=0.1, random_state=42)
    preds = clf.fit_predict(X)
    is_anom = preds == -1
    return is_anom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fourier_K", type=int, default=1)
    parser.add_argument("--pi_level", type=float, default=0.99)
    parser.add_argument("--pen", type=float, default=10.0, help="ruptures penalty for change point detection")
    parser.add_argument("--min_nonzero_weeks", type=int, default=4, help="minimum number of non-zero weeks to attempt modeling")
    parser.add_argument("--min_pi_width", type=float, default=0.5, help="minimum PI width to avoid zero-width intervals")
    parser.add_argument("--ensemble_thresh", type=float, default=2.0, help="minimum consensus count to mark ensemble anomaly")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel workers (ProcessPoolExecutor)")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info(f"Args: {args}")
    df = pd.read_csv(args.input, parse_dates=["insight_date_time"])
    dfw = ensure_weekly_counts(df)
    # Logging: show how many time series we will process
    groups = list(dfw.groupby(["insight_type","target_key"]))
    print(f"Found {len(groups)} (insight_type, target_key) groups to process")
    if len(groups) > 0:
        print("First 5 groups:", [g[0] for g in groups[:5]])
    diagnostics = []
    def process_group(it, tk, g):
        # local args capture
        try:
            g2, X, y = build_design(g, args.fourier_K)
            # NB2
            res_nb2 = fit_nb2(y, X)
            mu_nb2, pi_low_nb2, pi_high_nb2 = predict_nb2(res_nb2, X, args.pi_level)
            # enforce minimum PI width
            width_nb2 = np.maximum(pi_high_nb2 - pi_low_nb2, args.min_pi_width)
            pi_low_nb2 = mu_nb2 - width_nb2 / 2.0
            pi_high_nb2 = mu_nb2 + width_nb2 / 2.0
            is_anom_nb2 = (y < pi_low_nb2) | (y > pi_high_nb2)
            # Poisson
            res_pois = fit_poisson(y, X)
            mu_pois, pi_low_pois, pi_high_pois = predict_poisson(res_pois, X, args.pi_level)
            width_pois = np.maximum(pi_high_pois - pi_low_pois, args.min_pi_width)
            pi_low_pois = mu_pois - width_pois / 2.0
            pi_high_pois = mu_pois + width_pois / 2.0
            is_anom_pois = (y < pi_low_pois) | (y > pi_high_pois)
            # STL
            stl_res, is_anom_stl, resid_stl = stl_anomaly(y)
            # Prophet
            yhat_prop, yhat_lower_prop, yhat_upper_prop, is_anom_prop = prophet_anomaly(g2)
            # Isolation Forest
            is_anom_iforest = isolation_forest_anomaly(y)
            # Rolling Z-score
            z_roll, is_anom_roll = rolling_zscore(y)
            # Change points (on raw y)
            cps = detect_change_points(y, pen=args.pen)
            # compute consensus and normalized distances
            nb2_dist = np.abs(y - mu_nb2) / np.maximum(width_nb2, args.min_pi_width)
            pois_dist = np.abs(y - mu_pois) / np.maximum(width_pois, args.min_pi_width)
            z_roll_abs = np.abs(z_roll)
            consensus_count = (is_anom_nb2.astype(int) + is_anom_pois.astype(int) + is_anom_stl.astype(int) +
                               (is_anom_prop if np.any(~np.isnan(is_anom_prop)) else np.zeros_like(is_anom_stl)).astype(int) +
                               is_anom_iforest.astype(int) + is_anom_roll.astype(int))
            sum_norm_dist = np.nansum(np.vstack([nb2_dist, pois_dist, z_roll_abs]), axis=0)
            ensemble_is_anom = consensus_count >= args.ensemble_thresh

            diag = {
                "insight_type": it,
                "target_key": tk,
                "aic_nb2": res_nb2.aic,
                "aic_pois": res_pois.aic,
                "change_points": cps,
                "prophet_available": Prophet is not None,
                "mean_consensus": float(np.nanmean(consensus_count)),
                "max_consensus": int(np.nanmax(consensus_count))
            }

            # Save plot
            try:
                plt.figure(figsize=(14,8))
                plt.plot(g2["week_start"], y, label="Observed", marker="o", color='black')
                plt.plot(g2["week_start"], mu_nb2, label="NB2 Predicted", linestyle="--", color='blue')
                plt.fill_between(g2["week_start"], pi_low_nb2, pi_high_nb2, color="blue", alpha=0.1, label="NB2 PI")
                plt.plot(g2["week_start"], mu_pois, label="Poisson Predicted", linestyle=":", color='green')
                plt.fill_between(g2["week_start"], pi_low_pois, pi_high_pois, color="green", alpha=0.1, label="Poisson PI")
                if Prophet is not None:
                    plt.plot(g2["week_start"], yhat_prop, label="Prophet Predicted", linestyle="-.", color='brown')
                    plt.fill_between(g2["week_start"], yhat_lower_prop, yhat_upper_prop, color="brown", alpha=0.1, label="Prophet PI")
                plt.scatter(g2["week_start"], y, c=["red" if a else "black" for a in is_anom_nb2], label="NB2 Anomaly", marker='x')
                plt.scatter(g2["week_start"], y, c=["orange" if a else "none" for a in is_anom_pois], label="Poisson Anomaly", marker='o', edgecolors='orange')
                plt.scatter(g2["week_start"], y, c=["purple" if a else "none" for a in is_anom_stl], label="STL Anomaly", marker='s', edgecolors='purple')
                if Prophet is not None:
                    plt.scatter(g2["week_start"], y, c=["brown" if a else "none" for a in is_anom_prop], label="Prophet Anomaly", marker='D', edgecolors='brown')
                plt.scatter(g2["week_start"], y, c=["cyan" if a else "none" for a in is_anom_iforest], label="IForest Anomaly", marker='P', edgecolors='cyan')
                plt.scatter(g2["week_start"], y, c=["magenta" if a else "none" for a in is_anom_roll], label="Rolling Z Anomaly", marker='*', edgecolors='magenta')
                for i_cp, cp in enumerate(cps):
                    if cp < len(g2):
                        plt.axvline(g2["week_start"].iloc[cp], color="green", linestyle=":", label="Change Point" if i_cp==0 else None)
                plt.title(f"{it}-{tk} (Multi-Model)")
                plt.legend()
                plot_path = os.path.join(args.outdir, f"diagnostics_{it}_{tk}_multi.png")
                plt.savefig(plot_path)
                plt.close()
            except Exception:
                # non-fatal for plotting
                plot_path = None

            # Save flags
            out = pd.DataFrame({
                "insight_type":it,
                "target_key":tk,
                "week_start":g2["week_start"],
                "count":y,
                "mu_nb2":mu_nb2,
                "pi_low_nb2":pi_low_nb2,
                "pi_high_nb2":pi_high_nb2,
                "is_anom_nb2":is_anom_nb2,
                "mu_pois":mu_pois,
                "pi_low_pois":pi_low_pois,
                "pi_high_pois":pi_high_pois,
                "is_anom_pois":is_anom_pois,
                "stl_resid":resid_stl,
                "is_anom_stl":is_anom_stl,
                "prophet_yhat": yhat_prop,
                "prophet_yhat_lower": yhat_lower_prop,
                "prophet_yhat_upper": yhat_upper_prop,
                "is_anom_prophet": is_anom_prop,
                "is_anom_iforest": is_anom_iforest,
                "z_roll": z_roll,
                "is_anom_roll": is_anom_roll,
                # ensemble scoring and diagnostics per-row
                "consensus_count": consensus_count,
                "nb2_norm_dist": nb2_dist,
                "pois_norm_dist": pois_dist,
                "sum_norm_dist": sum_norm_dist,
                "ensemble_is_anom": ensemble_is_anom
            })
            out_path = os.path.join(args.outdir, f"flags_{it}_{tk}_multi.csv")
            out.to_csv(out_path, index=False)

            return {**diag, "plot_path": plot_path, "flags_path": out_path}
        except Exception as e:
            tb = traceback.format_exc()
            return {"insight_type": it, "target_key": tk, "error": str(e), "traceback": tb}

    # parallelize if requested
    if hasattr(args, 'workers') and args.workers and args.workers > 1:
        logging.info(f"Running with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_group, it, tk, g): (it, tk) for (it, tk), g in groups}
            for fut in as_completed(futures):
                res = fut.result()
                diagnostics.append(res)
    else:
        for idx, ((it, tk), g) in enumerate(groups):
            logging.info(f"Processing {idx+1}/{len(groups)}: {it} - {tk}")
            res = process_group(it, tk, g)
            diagnostics.append(res)

    pd.DataFrame(diagnostics).to_csv(os.path.join(args.outdir, "diagnostics_multi.csv"), index=False)

if __name__ == "__main__":
    main()

# End of script
