# Multi-Model Anomaly Detection Pipeline Documentation

## Overview
This pipeline detects anomalies in weekly event counts for each (insight_type, target_key) pair using multiple statistical models. It is designed for robust, interpretable anomaly detection in business or operational time series data.

## Data Preparation
- **Input:** Raw event-level data with at least these columns: `insight_type`, `target_key`, `insight_date_time`.
- **Aggregation:** Events are aggregated by week (Monday start) for each (insight_type, target_key) pair.
- **Missing Weeks:** The pipeline fills in missing weeks with zero counts to ensure a continuous time series.

## Models Implemented
### 1. Negative Binomial GLM
- Suitable for overdispersed count data (variance > mean).
- Fits a regression model with trend and seasonality (Fourier terms).
- Anomalies are flagged when observed counts fall outside the 99% prediction interval.

### 2. Poisson GLM
- Suitable for count data with mean ≈ variance.
- Similar regression structure as NB2.
- Anomalies flagged using Poisson prediction intervals.

### 3. STL Decomposition + Residual Anomaly Detection
- Decomposes the time series into trend, seasonality, and residuals.
- Anomalies are flagged when the residual z-score exceeds a threshold (default |z| > 3).

### 4. Prophet
- Handles trend and seasonality, robust to missing data and outliers.
- Good for business time series with complex seasonality.
- Anomalies flagged when observed counts fall outside Prophet's prediction interval.

### 5. Isolation Forest
- Unsupervised anomaly detection using tree-based ensemble.
- Detects outliers based on data distribution, not time order.
- Useful for non-parametric, distributional anomaly detection.

### 6. Rolling Z-score
- Simple statistical method using rolling mean and standard deviation.
- Flags anomalies when the z-score exceeds a threshold (default |z| > 3).
- Fast, interpretable, but less robust to non-stationarity.

## Output Files
- `flags_<insight_type>_<target_key>_multi.csv`: Weekly results for each pair, including model predictions, intervals, and anomaly flags.
- `diagnostics_<insight_type>_<target_key>_multi.png`: Unified plot showing observed counts, model fits, intervals, and anomalies.
- `diagnostics_multi.csv`: Summary of model diagnostics (AIC, change points, errors) for all pairs.

## Unified Display Script
- `display_multi_model_results.py`:
  - Lists available pairs.
  - For a selected pair, shows a single plot comparing all models and prints anomaly counts.
  - Usage:
    - List pairs: `python display_multi_model_results.py --outdir out_odm`
    - Display results: `python display_multi_model_results.py --outdir out_odm --insight_type performance --target_key CL-69294`

## How to Interpret Results
- **Unified Plot:**
  - Black: Observed weekly counts
  - Blue: NB2 model fit and interval; red x: NB2 anomaly
  - Green: Poisson fit and interval; orange o: Poisson anomaly
  - Purple s: STL anomaly
  - Green vertical lines: Detected change points
- **CSV:**
  - Each row is a week; columns show predictions, intervals, and anomaly flags for each model.
- **Diagnostics:**
  - AIC values help compare model fit; lower is better.
  - Change points indicate structural shifts in the time series.

## Model Comparison and Recommendations
| Model              | Strengths                                      | Weaknesses                        | Best Use Case                                 |
|--------------------|------------------------------------------------|------------------------------------|-----------------------------------------------|
| NB2 GLM            | Handles overdispersion, interpretable           | Needs enough data, parametric      | Count data with variance > mean               |
| Poisson GLM        | Simple, interpretable                          | Assumes mean ≈ variance            | Count data, low overdispersion                |
| STL                | Captures trend/seasonality, interpretable      | Sensitive to outliers, no PI       | Seasonal/trending time series                 |
| Prophet            | Robust, handles missing/outliers, flexible     | Slower, more complex               | Business/operational time series              |
| Isolation Forest   | Non-parametric, robust to distribution         | Ignores time order, less interpretable | Outlier detection in any distribution    |
| Rolling Z-score    | Fast, simple, interpretable                    | Sensitive to window/threshold      | Quick checks, stationary series               |

### Recommended Model(s)
- **For most business/operational time series:**
  - **Prophet** is recommended for its robustness, flexibility, and ability to handle missing data and outliers.
  - **NB2 GLM** is a strong choice for count data with overdispersion and when interpretability is important.
- **For quick, interpretable checks:**
  - **Rolling Z-score** is useful for fast anomaly detection in stationary data.
- **For non-parametric or distributional anomalies:**
  - **Isolation Forest** can catch outliers missed by parametric models.
- **For seasonal/trending data:**
  - **STL** is useful for decomposing and understanding structure, but pair with another model for anomaly flagging.

**Best practice:**
- Use the unified display script to compare model results for your data.
- Start with Prophet and NB2 for most cases, and use others for confirmation or special scenarios.

## Extending the Pipeline
- Additional models (e.g., Prophet) can be added as needed.
- Thresholds and model parameters can be tuned for your specific use case.

## Troubleshooting
- If a pair has too few weeks, models may not fit well; check for errors in diagnostics_multi.csv.
- Ensure all dependencies in requirements.txt are installed.

## Contact
For further customization or support, contact the project maintainer.
