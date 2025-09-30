# Anomaly Detection Pipeline Documentation

## Overview
This project implements a robust, production-ready anomaly detection pipeline for weekly count data, supporting multiple (target_type, insight_type) pairs and keys. The pipeline is designed to handle different seasonality, trends, and abrupt changes in the data, and provides:
- Model selection (Negative Binomial with Fourier terms)
- Change point detection (ruptures)
- Robust fallback
- Diagnostics and visualization

---

## Data Structure
- **Input CSV**: Must contain at least the following columns:
  - `insight_date_time` (datetime)
  - `target_type` (categorical)
  - `insight_type` (categorical)
  - `target_key` (categorical)

---

## Pipeline Steps

### 1. Data Aggregation
- Aggregates raw data to weekly counts for each (target_type, insight_type, target_key).
- Fills missing weeks with zero counts.

### 2. Model Fitting & Selection
- For each (target_type, insight_type) pair and key:
  - Fits Negative Binomial (NB2) regression models with different numbers of Fourier terms (seasonality).
  - Selects the best model using AIC.
  - If model fitting fails, uses a robust rolling window fallback.

### 3. Change Point Detection
- Uses the `ruptures` library (Pelt algorithm, model='l2', log1p transform) to detect change points in each time series.
- If a change point is found, splits the series and fits separate models to each segment.

### 4. Anomaly Detection
- Calculates prediction intervals using the NB2 model.
- Flags anomalies where observed counts fall outside the prediction interval.

### 5. Diagnostics & Visualization
- Outputs diagnostics (AIC, fallback usage, change points) for each segment to a CSV.
- Generates plots for each series, showing:
  - Actual counts
  - Predicted mean
  - Prediction intervals
  - Anomalies (red dots)
  - Change points (purple vertical lines)
- Plots are saved in `diagnostics_plots/`.

---

## Output Files
- `light_nb_flags_by_key_weekly.csv`: Anomaly flags for each key.
- `light_nb_flags_by_pair_weekly.csv`: Anomaly flags for each pair.
- `diagnostics.csv`: Model diagnostics for each segment.
- `diagnostics_plots/`: Plots for each pair and key.

---

## Example Visualization

Below is an example of the type of plot generated for each series:

![Example Anomaly Plot](diagnostics_plots/example_plot.png)

- **Blue line**: Actual weekly counts
- **Dashed line**: Model-predicted mean
- **Gray band**: Prediction interval
- **Red dots**: Detected anomalies
- **Purple lines**: Detected change points

---

## Code Structure

### lightweight_nb_anomaly.py
- Main script implementing the pipeline.
- Key functions:
  - `ensure_weekly_counts`: Aggregates and fills missing weeks.
  - `build_design`: Builds the regression design matrix.
  - `fit_nb2`: Fits NB2 model.
  - `predict_pi`: Calculates prediction intervals.
  - `robust_fallback`: Fallback for model failures.
  - `detect_change_points`: Finds change points using ruptures.
  - `plot_series`: Generates and saves plots.
  - `main`: Orchestrates the pipeline.

### run_lightweight.py
- Runner script to execute the pipeline with the correct input/output paths.

---

## Requirements
- Python 3.7+
- pandas
- numpy
- statsmodels
- ruptures
- matplotlib

Install with:
```
pip install pandas numpy statsmodels ruptures matplotlib
```

---

## Usage
1. Place your input CSV in the correct location.
2. Update `run_lightweight.py` if needed.
3. Run:
```
python run_lightweight.py
```
4. Review outputs in the `out_lightweight` directory.

---

## Notes
- The pipeline is robust to missing data, overdispersion, and abrupt changes.
- Diagnostics and plots help validate model quality and anomaly detection.
- You can tune the change point penalty (`pen` in `detect_change_points`) and model complexity (`fourier_K`) as needed.

---

## Contact
For further customization or support, contact your data science team.

