#!/usr/bin/env python3
import os, sys, subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_DIR)
SCRIPT = os.path.join(THIS_DIR, "odm_nb_anomaly.py")
INPUT_PATH = os.path.join(DATA_PATH, "synthetic_insights_weekly_1y_preprocessed.csv")
OUTDIR = os.path.join(THIS_DIR, "out_odm")

if not os.path.exists(SCRIPT):
    print("ERROR: odm_nb_anomaly.py not found.")
    sys.exit(1)
if not os.path.exists(INPUT_PATH):
    print("ERROR: synthetic_insights_weekly_1y_preprocessed.csv not found.")
    sys.exit(1)

os.makedirs(OUTDIR, exist_ok=True)

cmd = [
    sys.executable, SCRIPT,
    "--input", INPUT_PATH,
    "--outdir", OUTDIR,
    "--fourier_K", "1",
    "--pi_level", "0.99"
]
print("Running:", " ".join(cmd))
sys.exit(subprocess.run(cmd).returncode)
