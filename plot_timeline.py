"""
Money plot: x50 (estimated human time at 50% success) vs model release date,
with exponential trend fit using bootstrap resampling for uncertainty.

Model: log10(hours) = a + b * t  (t in years since first model)
  => hours = 10^a * 10^(b*t)
  => doubling time = log10(2) / b years
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────────────────

USE_CALIBRATION = True

N_TREND_BOOTSTRAP = 10000
SEED = 123

results_dir = "results_cal" if USE_CALIBRATION else "results_no_cal"

with open(f"{results_dir}/summary.json") as f:
    summary = json.load(f)

rng = np.random.default_rng(SEED)

# ── Parse data + load bootstrap samples ────────────────────────────────

# We need: dates, display names, and the full bootstrap x50 arrays
models = []
for model_key, info in summary.items():
    rd = info.get("release_date")
    if rd is None:
        continue
    dt = datetime.strptime(rd, "%Y-%m-%d")

    # Load bootstrap samples (x50 is already in log10-hours)
    fn = info["display_name"].replace(" ", "_").replace("/", "_").replace(":", "_")
    npz_path = os.path.join(results_dir, fn, "bootstrap_samples.npz")

    if not os.path.exists(npz_path):
        print(f"  Warning: no bootstrap samples for {info['display_name']}, skipping")
        continue

    boot = np.load(npz_path)
    models.append({
        "key": model_key,
        "name": info["display_name"],
        "date": dt,
        "x50_samples": boot["x50"],  # log10(hours), shape (5000,)
        "x50_q025": info["x50_q025"],
        "x50_q50": info["x50_q50"],
        "x50_q975": info["x50_q975"],
    })

# Sort by date
models.sort(key=lambda m: m["date"])
n_models = len(models)

# Reference date: first model's release
t0 = models[0]["date"]

# Convert dates to years since t0
def date_to_years(dt):
    return (dt - t0).days / 365.25

t_years = np.array([date_to_years(m["date"]) for m in models])
dates = [m["date"] for m in models]
names = [m["name"] for m in models]

# Median and quantiles in hours
hours_q50 = np.array([10**m["x50_q50"] for m in models])
hours_q025 = np.array([10**m["x50_q025"] for m in models])
hours_q975 = np.array([10**m["x50_q975"] for m in models])

print(f"Models: {n_models}")
print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"Time span: {t_years[-1]:.1f} years")

# ── Exponential trend fit via bootstrap resampling ─────────────────────

# For each iteration: draw one x50 sample per model, fit line in
# (t_years, log10_hours) space using ordinary least squares.

boot_a = np.zeros(N_TREND_BOOTSTRAP)  # intercept (log10-hours at t0)
boot_b = np.zeros(N_TREND_BOOTSTRAP)  # slope (log10-hours per year)

n_samples = len(models[0]["x50_samples"])

for i in range(N_TREND_BOOTSTRAP):
    # Draw one random bootstrap sample per model
    y = np.array([m["x50_samples"][rng.integers(n_samples)] for m in models])
    # OLS: y = a + b*t
    coeffs = np.polyfit(t_years, y, 1)
    boot_b[i] = coeffs[0]
    boot_a[i] = coeffs[1]

a_q50 = np.median(boot_a)
b_q50 = np.median(boot_b)
a_q025, a_q975 = np.percentile(boot_a, [2.5, 97.5])
b_q025, b_q975 = np.percentile(boot_b, [2.5, 97.5])

# Doubling time: time for hours to double = log10(2) / b years
doubling_time = np.log10(2) / boot_b  # array
dt_q50 = np.median(doubling_time)
dt_q025, dt_q975 = np.percentile(doubling_time, [2.5, 97.5])

# 10x time: time for hours to increase 10x = 1/b years
tenfold_time = 1.0 / boot_b
tf_q50 = np.median(tenfold_time)
tf_q025, tf_q975 = np.percentile(tenfold_time, [2.5, 97.5])

print(f"\nExponential trend: log10(hours) = {a_q50:.3f} + {b_q50:.3f} * t_years")
print(f"  Slope: {b_q50:.3f} [{b_q025:.3f}, {b_q975:.3f}] log10(hours)/year")
print(f"  Doubling time: {dt_q50*12:.1f} months [{dt_q025*12:.1f}, {dt_q975*12:.1f}]")
print(f"  10x time: {tf_q50:.2f} years [{tf_q025:.2f}, {tf_q975:.2f}]")

# ── Plot ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

# Data points with error bars
err_lo = hours_q50 - hours_q025
err_hi = hours_q975 - hours_q50
ax.errorbar(dates, hours_q50, yerr=[err_lo, err_hi],
            fmt="o", markersize=8, capsize=5, capthick=1.5,
            color="C0", ecolor="C0", elinewidth=1.5, zorder=3,
            label="Models (median + 95% CI)")

# Trend line + uncertainty band
# Extend a bit beyond the data range
t_plot_years = np.linspace(-0.3, t_years[-1] + 0.3, 200)
dates_plot = [t0 + timedelta(days=t * 365.25) for t in t_plot_years]

# Draw curves from bootstrap
n_draw = min(500, N_TREND_BOOTSTRAP)
idx_draw = rng.choice(N_TREND_BOOTSTRAP, n_draw, replace=False)
curves = np.zeros((n_draw, len(t_plot_years)))
for i, idx in enumerate(idx_draw):
    curves[i] = 10**(boot_a[idx] + boot_b[idx] * t_plot_years)

median_curve = np.median(curves, axis=0)
lo_curve = np.percentile(curves, 2.5, axis=0)
hi_curve = np.percentile(curves, 97.5, axis=0)

ax.plot(dates_plot, median_curve, color="C1", lw=2, zorder=2,
        label=f"Exponential fit (2x every {dt_q50*12:.1f} mo [{dt_q025*12:.1f}, {dt_q975*12:.1f}])")
ax.fill_between(dates_plot, lo_curve, hi_curve, alpha=0.2, color="C1", zorder=1)

# Label each point
for i, name in enumerate(names):
    offset = 14 if i % 2 == 0 else -18
    va = "bottom" if i % 2 == 0 else "top"
    ax.annotate(name, (dates[i], hours_q50[i]),
                textcoords="offset points", xytext=(0, offset),
                ha="center", va=va, fontsize=8)

ax.set_yscale("log")
ax.set_ylabel("Estimated human work-hours at 50% success", fontsize=12)
ax.set_xlabel("Model release date", fontsize=12)

cal_tag = " (calibrated)" if USE_CALIBRATION else ""
ax.set_title(f"WeirdML time horizon by release date{cal_tag}", fontsize=14)

# Y-axis: human-readable time ticks
# Work-time convention: 1 day = 8h, 1 week = 40h, 1 month = 160h
time_ticks_hours = [
    1/60, 2/60, 5/60, 10/60, 20/60, 30/60,
    1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 1920, 3840,
]
time_tick_labels = [
    "1 min", "2 min", "5 min", "10 min", "20 min", "30 min",
    "1 h", "2 h", "5 h", "10 h", "20 h", "1 wk (40h)", "2 wk", "1 mo", "2 mo", "4 mo", "8 mo", "1 yr", "2 yr",
]

ymin = min(hours_q025.min(), lo_curve.min()) / 2
ymax = max(hours_q975.max(), hi_curve.max()) * 2
visible = [(h, l) for h, l in zip(time_ticks_hours, time_tick_labels)
           if ymin <= h <= ymax]
if visible:
    ax.set_yticks([h for h, _ in visible])
    ax.set_yticklabels([l for _, l in visible])
ax.set_ylim(ymin, ymax)
ax.yaxis.set_minor_locator(plt.NullLocator())

# X-axis
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=30)

xpad = timedelta(days=60)
ax.set_xlim(dates[0] - xpad, dates[-1] + xpad)

ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
out_path = f"{results_dir}/timeline.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved {out_path}")
