#!/usr/bin/env python3
"""
Simple runner for PyCharm.
- Put `synthetic_insights_weekly_1y.csv` in the same folder as this script (or change INPUT_PATH below).
- Then run this file in PyCharm (Run ▶). Outputs will go to ./out_hier_nb by default.
"""
import os, sys, subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH=THIS_DIR+"/synthetic_data"
SCRIPT = os.path.join(THIS_DIR, "hierarchical_nb_pymc.py")
INPUT_PATH = os.path.join(DATA_PATH, "synthetic_insights_weekly_1y.csv")
OUTDIR = os.path.join(THIS_DIR, "out_hier_nb")

if not os.path.exists(SCRIPT):
    print("ERROR: hierarchical_nb_pymc.py not found next to this runner.")
    sys.exit(1)
if not os.path.exists(INPUT_PATH):
    print("ERROR: synthetic_insights_weekly_1y.csv not found. Copy your weekly CSV here or update INPUT_PATH.")
    sys.exit(1)

os.makedirs(OUTDIR, exist_ok=True)

cmd = [
    sys.executable, SCRIPT,
    "--input", INPUT_PATH,
    "--outdir", OUTDIR,
    "--draws", "800",
    "--tune", "800",
    "--chains", "2",
    "--cores", "2",
    "--fourier_K", "2",
    "--pi_level", "0.99"
]

print("Running:", " ".join(cmd))
ret = subprocess.run(cmd)
sys.exit(ret.returncode)
