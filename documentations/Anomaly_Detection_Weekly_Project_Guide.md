# Weekly Anomaly Detection Project — Design, Models, and Runbook
_Last updated: 2025-09-30 04:28_

## 1) Executive Summary
This project detects **unusual weekly behavior** in the number of **target_key hits** grouped by **(target_type, insight_type)**.
We create **1 year of weekly data** with realistic seasonality and **injected anomalies** (spikes, drops, and sustained shifts), then apply **Negative Binomial (NB2) regression with seasonality** to flag anomalies at both the **per-key** and **pair-level** (pair ≡ `(target_type, insight_type)`).

Two pipelines are provided:
- **Lightweight (Statsmodels)** — fast, no compiler required. Uses NB2 with Fourier seasonality and a light trend, plus a robust rolling fallback.
- **Hierarchical (PyMC)** — richer pooling across keys within a pair with a **Gaussian Random Walk** trend and **posterior predictive** anomaly flags. Requires a compiler (or conda toolchain).

Both pipelines generate **CSV outputs** and **charts** (matplotlib) that highlight where and when anomalies occur.

---

## 2) Data Design (Weekly Refresh)
**Goal:** “Observe if number of target_key hits has gone abnormal against each target_type and corresponding insight_type.”

**Columns** (single raw CSV):
- `insight_id` — unique row id (one row per **hit**)
- `insight_type` — e.g., performance, risk, behavior, opportunity, quality
- `target_type` — top layer (client, account, advisor, product, segment)
- `target_key` — specific entity within a `(target_type, insight_type)`
- `metric name` — `"insight_hit"`
- `metric_type` — `"event"`
- `metric_value` — `1` (each row is one hit)
- `run_id` — weekly identifier `RUN-YYYYWww-###`
- `insight_date_time` — timestamp of the weekly refresh (anchored around Monday 06:00 with small jitter)
- `insights_category` — auxiliary label (e.g., SoW, CIO, BGap, Compliance, Engagement…)

**Weekly generation choices**
- **Refresh cadence:** exactly once per week per key (some weeks may have 0 hits).
- **Seasonality:** yearly pattern (via Fourier), light random drift.
- **Anomalies:** 
  - **Spikes** (1–2 weeks, ×3–×7)
  - **Drops** (2–4 weeks, ~2–25% of normal)
  - **Sustained shifts** (6–10 weeks, ×1.7–×2.8)

These yield **variable, overdispersed counts**, which motivates **NB2** over Poisson.

---

## 3) Modeling Approach — Why Negative Binomial (NB2)?
**Poisson** assumes `Var(Y|X) = μ`. Weekly hits are often **overdispersed** (variance > mean).  
**NB2** allows `Var(Y|X) = μ + α μ²` with dispersion **α > 0**, yielding **wider, calibrated prediction intervals**, fewer false alarms, and robustness to spikes.

**Seasonality & Trend**
- Weekly **Fourier terms** (period ≈ 52) capture sinusoidal seasonality.
- **Trend**: a light linear term (lightweight pipeline) or a **Gaussian Random Walk** (PyMC) to adapt to slow drifts.

**Anomaly Criterion**
- Compute **predictive intervals** for each week.  
- Flag `is_anom = True` when observed `count` falls **outside** a high-confidence band (default **99%** PI).  
- Store **severity metrics** (e.g., deviation ratio, Pearson residuals) for ranking.

---

## 4) Pipelines

### 4.1 Lightweight Pipeline (Statsmodels) — Recommended for laptops
**File:** `lightweight_nb_anomaly.py`  
**Dependencies:** `requirements_light.txt` (pure Python/NumPy; adds matplotlib for charts).

**Per-key model** (for each `(target_type, insight_type, target_key)`):  
\[ \log\mu_t = \beta_0 + \sum_k [a_k\sin(2\pi k t/52) + b_k\cos(2\pi k t/52)] + c\cdot t/52 \]

**Pair-level model** on the aggregate counts per `(target_type, insight_type)` uses the same structure.

**Fallback:** If NB2 fails or a series is tiny, use **rolling mean ± 3σ** (8-week window).

**Outputs:**
- `light_nb_flags_by_key_weekly.csv` — per-key: `count`, `mu_hat`, `pi_low`, `pi_high`, `is_anom`
- `light_nb_flags_by_pair_weekly.csv` — pair aggregate: same columns (no key)
- **Charts** (`out_lightweight/figs/`): top N per-key and pair-level series with bands

**Run (PyCharm / CLI):**
```bash
# Install deps
pip install -r requirements_light.txt

# Run with visuals
python lightweight_nb_anomaly.py   --input synthetic_insights_weekly_1y.csv   --outdir out_lightweight   --fourier_K 1   --pi_level 0.99   --viz_top 10
```
> Tip: Increase `--fourier_K` to 2 if you need stronger seasonality. Use `--max_pairs` and `--max_keys` to limit scope for speed.

---

### 4.2 Hierarchical Pipeline (PyMC) — Partial pooling & posterior predictive
**File:** `hierarchical_nb_pymc.py`  
**Env:** `environment.yml` (includes PyMC, ArviZ, matplotlib). Requires a compiler toolchain.

**Structure (within each pair `g`):**
- **Random intercepts per key**: \( b_k \sim \mathcal{N}(0, \sigma_{key}) \)
- **Pair intercept** \( a_{pair} \), **Fourier seasonality** \( F_t \beta \), and **trend** \( RW_t \sim \text{GRW}(\sigma_{rw}) \)
- **Likelihood**: \( y_{t,k} \sim \text{NB}(\mu_{t,k}, \alpha) \), with \( \log \mu_{t,k} = a_{pair} + b_k + F_t\beta + RW_t \)

**Why hierarchical?**  
Stabilizes estimates for **sparse keys**, shares a **pair-specific curve** across keys, and yields calibrated **posterior predictive intervals**.

**Outputs:**
- `pymc_nb_flags_by_key_weekly.csv`
- `pymc_nb_flags_by_pair_weekly.csv`
- `pymc_distribution_shift_metrics.csv` — pair-level **active_keys**, **top_share**, **HHI**, **KL divergence** + robust z-scores
- **Charts** (`<outdir>/figs/`): top N per-key and pair-level series

**Run (conda recommended):**
```bash
conda env create -f environment.yml
conda activate hier-nb

python hierarchical_nb_pymc.py   --input synthetic_insights_weekly_1y.csv   --outdir out_hier_nb   --draws 800 --tune 800 --chains 2 --cores 2   --fourier_K 2 --pi_level 0.99 --viz_top 10
```
> If you see a compiler warning (e.g., “g++ not available”), install a conda toolchain (`gxx_linux-64`, `m2w64-toolchain`, or `clangxx_*` depending on OS).

---

## 5) Visualization
Both pipelines save **per-series charts** with:
- Line for **observed counts**
- Line for **model mean** (`mu_hat`)
- Lines for **prediction interval** (`pi_low`, `pi_high`)

Ranking logic selects **top-N** series by **# anomalous weeks** and **max deviation ratio** to quickly surface the most interesting cases.

---

## 6) Evaluation & Thresholds
- **PI Coverage:** On stable periods, ~99% of points should fall within the **99% PI**. If too many alerts, reduce to **95%** or increase seasonality/trend flexibility.
- **Precision/Recall (if ground truth exists):** Compare `is_anom` to labeled events (e.g., injected anomalies) and tune `--pi_level` and `fourier_K` accordingly.
- **Alert Budget:** Adjust PI level to keep weekly alert count manageable (e.g., target 0.5–2% of weeks flagged per series).

---

## 7) Dealing with Heterogeneity (“different change curves”)
- **Per-pair baselines:** Each `(target_type, insight_type)` gets its **own** seasonal/trend structure.
- **Hierarchical pooling:** Keys borrow the pair’s curve via random effects; good for **sparse keys**.
- **Distribution shift checks:** Even if totals look normal, track **reallocation**:
  - `active_keys`: # keys with >0
  - `top_share`: max key share
  - `HHI`: sum of squared shares
  - `KL divergence`: vs a rolling 8-week reference  
  Large robust-z on these signals a **mix change** anomaly.

---

## 8) Runbook (PyCharm)
1. **Place files** in a project folder along with `synthetic_insights_weekly_1y.csv` (your weekly raw data).
2. **Lightweight path (fast):**
   - Create a venv and install `requirements_light.txt`.
   - Run: `run_lightweight.py` (configured to produce charts).
3. **Hierarchical path (richer):**
   - `conda env create -f environment.yml` → `conda activate hier-nb`
   - Run: `run_in_pycharm.py` (or call the script directly).  
4. **Review outputs:**
   - CSVs: flagged anomalies per key and per pair.
   - PNGs in the `figs/` folder for quick triage.

---

## 9) Limitations & Next Steps
- **Short history**: With < 26–39 weeks, seasonal estimates can be unstable. Use lower `fourier_K`, wider PIs, or robust fallback.
- **Abrupt regime changes**: Consider adding a **change-point detector** (e.g., Bayesian Online Change Point Detection) to reset baselines.
- **Covariates / exposure**: If available (e.g., # active accounts), include `log(exposure)` as an offset to normalize volumes.
- **Productionization**: Schedule weekly runs, persist results to a table, and integrate with a dashboard (Power BI/Looker) and alerting (email/Slack).

---

## 10) File Map (What you have)
- **Data:** `synthetic_insights_weekly_1y.csv` (raw weekly events)
- **Lightweight:** `requirements_light.txt`, `lightweight_nb_anomaly.py`, `run_lightweight.py`
- **Hierarchical:** `environment.yml`, `hierarchical_nb_pymc.py`, `run_in_pycharm.py`
- **Outputs (created on run):**
  - `out_lightweight/` or `out_hier_nb/` → CSV flags and `figs/*.png`

---

## 11) FAQ
**Q:** Why NB2 instead of Poisson?  
**A:** Weekly counts are overdispersed; NB2 models variance as `μ + α μ²`, reducing false positives.

**Q:** Why Fourier vs month dummies?  
**A:** Fourier uses few parameters, fits smooth seasonality, and generalizes well on weekly data.

**Q:** Can I throttle computation?  
**A:** Yes. Lower `fourier_K`, raise PI (e.g., 99.5%), or restrict scope via `--max_pairs` / `--max_keys` (lightweight), or reduce `--draws/--tune` (PyMC).

---

*Prepared for: Weekly anomaly detection on target_key hits by (target_type, insight_type).*
