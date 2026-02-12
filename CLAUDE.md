# WeirdML Time Horizons

## What This Project Does

Estimates the "time horizon" of LLMs — the human-equivalent task duration at
which a model has 50% success probability on WeirdML tasks. Fits a logistic
curve per model, then tracks how time horizons grow over model generations.

## Key Result

Time horizons double roughly every 6 months (calibrated). From ~9 min (gpt-4,
June 2023) to ~6 hours (gpt-5.2, December 2025).

## Data

- `data/all_estimates.json` — LLM-generated human time estimates. 4 estimator
  models (gpt-5.2:medium, gemini-3-pro-preview, claude-opus-4.5, grok-4-07-09)
  × 17 tasks × 5 thresholds (25%, 50%, 70%, 90%, 95%). 
- `data/all_estimates.csv` — Same data in CSV format.
- `data/weirdml_results.json` — WeirdML benchmark scores. 84 models × 17 tasks
  × multiple runs (typically 5). Each run has a score 0–1.
- `data/human_estimates.json` — Human baseline for 3 tasks. Used to generate 
  the calibration
- `analysis/calibration.json` — Threshold-dependent calibration factors, 
  derived from the human estimates. `avg_log_diff` is in **ln** space 
  (divide by ln(10) for log10).

## Scripts

### `fit_bootstrap.py` — Main analysis (per-model logistic fits)

For each of 9 models, fits: `p(success) = sigmoid(β * (log10(hours) - x50))`

- Each (task, threshold, estimator, run) = one binary data point
- Using all 4 estimator models' time estimates as separate x-values naturally
  captures x-uncertainty without integrals
- Task-level block bootstrap (resample 17 tasks with replacement) for
  uncertainty in x50 and β — accounts for within-task correlations
- β reparameterized as β = -exp(v) to keep it negative
- L-BFGS-B optimizer with bounds: x50 ∈ [-3, 7], v ∈ [log(1e-6), log(50)]
- 5000 bootstrap samples per model
- Outputs per model: `fit.png`, `corner.png`, `bootstrap_samples.npz`
- Outputs: `summary.json` with medians and 95% CIs
- Toggle `USE_CALIBRATION` for calibrated vs uncalibrated runs
- Results go to `results_cal/` or `results_no_cal/`

### `plot_timeline.py` — Money plot (time horizon vs release date)

- Loads bootstrap samples from per-model .npz files
- Fits exponential trend: `log10(hours) = a + b * t_years`
- 10000 bootstrap iterations (draw one x50 sample per model, OLS fit)
- Reports doubling time with 95% CI
- Toggle `USE_CALIBRATION`

## Models (9 total)
These are the models that, at some time, had the highest WeirdML score. 
gpt-4, claude-3-opus, claude-3.5-sonnet-20240620, o1-preview,
o4-mini (high), o3-pro (high), gpt-5 (high), gemini-3-pro-preview (high),
gpt-5.2 (xhigh)

Note: `claude-3.5-sonnet-20240620` is the original June 2024 model, not the
later one posthumously called claude-3.6-sonnet.

## Gotchas

- One entry in all_estimates.json has hours=0 ("impossible") — must skip
  (log(0) = -inf)
- Calibration offsets are in **ln** space — divide by ln(10) when working in
  log10
- Weak models (gpt-4) with few successes can cause pathological bootstrap fits
  — bounded optimizer handles this
- Use `scipy.special.expit` instead of manual sigmoid to avoid overflow
- Folder names are derived from display names with spaces/slashes replaced by
  underscores
