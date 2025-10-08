import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Helper to find available pairs
def get_available_pairs(outdir):
    files = [f for f in os.listdir(outdir) if f.endswith('_multi.csv')]
    pairs = [f.replace('flags_','').replace('_multi.csv','').split('_',1) for f in files]
    return pairs

def display_results(outdir, insight_type, target_key):
    fname = f"flags_{insight_type}_{target_key}_multi.csv"
    fpath = os.path.join(outdir, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return
    df = pd.read_csv(fpath, parse_dates=["week_start"])
    plt.figure(figsize=(14,8))
    plt.plot(df["week_start"], df["count"], label="Observed", marker="o", color="black")
    plt.plot(df["week_start"], df["mu_nb2"], label="NB2 Predicted", linestyle="--", color="blue")
    plt.fill_between(df["week_start"], df["pi_low_nb2"], df["pi_high_nb2"], color="blue", alpha=0.1, label="NB2 PI")
    plt.plot(df["week_start"], df["mu_pois"], label="Poisson Predicted", linestyle=":", color="green")
    plt.fill_between(df["week_start"], df["pi_low_pois"], df["pi_high_pois"], color="green", alpha=0.1, label="Poisson PI")
    plt.scatter(df["week_start"], df["count"], c=["red" if a else "black" for a in df["is_anom_nb2"]], label="NB2 Anomaly", marker='x')
    plt.scatter(df["week_start"], df["count"], c=["orange" if a else "none" for a in df["is_anom_pois"]], label="Poisson Anomaly", marker='o', edgecolors='orange')
    plt.scatter(df["week_start"], df["count"], c=["purple" if a else "none" for a in df["is_anom_stl"]], label="STL Anomaly", marker='s', edgecolors='purple')
    plt.scatter(df["week_start"], df["count"], c=["brown" if a else "none" for a in df.get("is_anom_prophet", [False]*len(df))], label="Prophet Anomaly", marker='D', edgecolors='brown')
    plt.scatter(df["week_start"], df["count"], c=["cyan" if a else "none" for a in df.get("is_anom_iforest", [False]*len(df))], label="IForest Anomaly", marker='P', edgecolors='cyan')
    plt.scatter(df["week_start"], df["count"], c=["magenta" if a else "none" for a in df.get("is_anom_roll", [False]*len(df))], label="Rolling Z Anomaly", marker='*', edgecolors='magenta')
    plt.title(f"Multi-Model Anomaly Detection: {insight_type} - {target_key}")
    plt.xlabel("Week Start")
    plt.ylabel("Event Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Print summary table
    summary = {
        'NB2 Anomalies': int(df['is_anom_nb2'].sum()),
        'Poisson Anomalies': int(df['is_anom_pois'].sum()),
        'STL Anomalies': int(df['is_anom_stl'].sum()),
        'Prophet Anomalies': int(df.get('is_anom_prophet', pd.Series([0]*len(df))).sum()),
        'IForest Anomalies': int(df.get('is_anom_iforest', pd.Series([0]*len(df))).sum()),
        'Rolling Z Anomalies': int(df.get('is_anom_roll', pd.Series([0]*len(df))).sum())
    }
    print("\nAnomaly Summary:")
    for k,v in summary.items():
        print(f"{k}: {v}")
    # Optionally, print first few rows
    print("\nSample of results:")
    print(df.head())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='out_odm', help='Directory with multi-model results')
    parser.add_argument('--insight_type', required=False, help='Insight type to display')
    parser.add_argument('--target_key', required=False, help='Target key to display')
    args = parser.parse_args()
    pairs = get_available_pairs(args.outdir)
    if not pairs:
        print("No multi-model results found in output directory.")
        return
    if not args.insight_type or not args.target_key:
        print("Available pairs:")
        for it, tk in pairs:
            print(f"  {it}, {tk}")
        print("\nSpecify --insight_type and --target_key to display a specific pair.")
        return
    display_results(args.outdir, args.insight_type, args.target_key)

if __name__ == "__main__":
    main()
