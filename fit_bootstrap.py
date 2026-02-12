"""
Bootstrap analysis of logistic time-horizon fit.

1. Fit full data with scipy.minimize (L-BFGS-B, bounded) to get initial (x50, beta)
2. Bootstrap: resample 17 tasks with replacement, optimize (x50, v)
   where beta = -exp(v), using L-BFGS-B with bounds
3. Report median + 95% CI from bootstrap distribution

x = log10(hours) throughout.
"""

import json
import os
import numpy as np
import corner
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

LN10 = np.log(10)

# ── Load data ──────────────────────────────────────────────────────────

with open("data/all_estimates.json") as f:
    estimates = json.load(f)

with open("data/weirdml_results.json") as f:
    results = json.load(f)

with open("analysis/calibration.json") as f:
    calibration = json.load(f)

# ── Config ─────────────────────────────────────────────────────────────

MODELS = [
    "gpt-4",                     # gpt-4-0613
    "claude-3-opus",             # claude-3-opus
    "claude-3.5-sonnet-20240620", # claude-3.5-sonnet (original)
    "o1-preview",                # o1-preview
    "o4-mini-high-2025-04-16",   # o4-mini (high)
    "o3-pro-2025-06-10",         # o3-pro (high)
    "gpt-5-2025-08-07",          # gpt-5 (high)
    "gemini-3-pro-preview",      # gemini-3-pro-preview (high)
    "gpt-5.2",                   # gpt-5.2 (xhigh)
]

USE_CALIBRATION = True

THRESHOLDS = [0.25, 0.50, 0.70, 0.90, 0.95]
THRESHOLD_LABELS = ["25%", "50%", "70%", "90%", "95%"]
ESTIMATOR_MODELS = ["gpt-5.2:medium", "gemini-3-pro-preview", "claude-opus-4.5", "grok-4-07-09"]

# Calibration offsets converted from ln to log10
CAL_OFFSETS = {}
for t in THRESHOLD_LABELS:
    if USE_CALIBRATION:
        CAL_OFFSETS[t] = calibration["by_threshold"][t]["avg_log_diff"] / LN10
    else:
        CAL_OFFSETS[t] = 0.0

N_BOOTSTRAP = 5000
SEED = 42

# Bounds: x50 in [-3, 7] (log10-hours), v = log(-beta) in [log(1e-6), log(50)]
BOUNDS = [(-3, 7), (np.log(1e-6), np.log(50))]

# Output directory
OUTPUT_BASE = "results_cal" if USE_CALIBRATION else "results_no_cal"

# Time ticks for the top x-axis (work-hours, label)
# Convention: 1 day = 8h, 1 week = 40h, 1 month = 160h (matching data conversion)
TIME_TICKS = [
    (1/60,   "1 min"),
    (5/60,   "5 min"),
    (10/60,  "10 min"),
    (30/60,  "30 min"),
    (1,      "1 h"),
    (2,      "2 h"),
    (5,      "5 h"),
    (10,     "10 h"),
    (20,     "20 h"),
    (40,     "1 wk (40h)"),
    (80,     "2 wk"),
    (160,    "1 mo"),
    (320,    "2 mo"),
    (640,    "4 mo"),
    (1280,   "8 mo"),
    (1920,   "1 yr"),
    (3840,   "2 yr"),
]

# ── Helpers ────────────────────────────────────────────────────────────

def neg_log_likelihood(params, xs, ys):
    """Negative log-likelihood. params = (x50, v) where beta = -exp(v)."""
    x50, v = params
    beta = -np.exp(v)
    eta = beta * (xs - x50)
    ll = ys * eta - np.logaddexp(0, eta)
    return -np.sum(ll)

def build_task_data(model_key):
    model_tasks = results["models"][model_key]["tasks"]
    task_data_dict = {}

    for task_name, task_data in model_tasks.items():
        if task_name not in estimates:
            print(f"  WARNING: task '{task_name}' not in estimates, skipping")
            continue

        task_estimates = estimates[task_name]["estimates"]
        runs = task_data["runs"]

        task_xs = []
        task_ys = []

        for thresh, thresh_label in zip(THRESHOLDS, THRESHOLD_LABELS):
            cal = CAL_OFFSETS[thresh_label]

            for estimator in ESTIMATOR_MODELS:
                if estimator not in task_estimates:
                    raise KeyError(f"Estimator '{estimator}' missing from estimates for task '{task_name}'")
                if thresh_label not in task_estimates[estimator]:
                    raise KeyError(f"Threshold '{thresh_label}' missing from estimates for task '{task_name}', estimator '{estimator}'")

                hours = task_estimates[estimator][thresh_label]["hours"]
                if hours <= 0:
                    print(f"  WARNING: hours={hours} for {task_name}/{estimator}/{thresh_label}, skipping")
                    continue
                x = np.log10(hours) + cal

                for run in runs:
                    y = 1 if run["score"] >= thresh else 0
                    task_xs.append(x)
                    task_ys.append(y)

        if len(task_xs) > 0:
            task_data_dict[task_name] = (np.array(task_xs), np.array(task_ys))

    return task_data_dict

def get_display_name(model_key):
    return results["models"][model_key].get("display_name", model_key)

def safe_folder_name(display_name):
    return display_name.replace(" ", "_").replace("/", "_").replace(":", "_")

def get_time_ticks(xlim):
    ticks = []
    labels = []
    for hours, label in TIME_TICKS:
        lx = np.log10(hours)
        if xlim[0] <= lx <= xlim[1]:
            ticks.append(lx)
            labels.append(label)
    return ticks, labels

def x_to_hours(x):
    """Convert log10-hours to hours."""
    return 10**x

def format_hours(h):
    """Human-readable time string from work-hours (8h/day, 40h/week, 160h/month)."""
    if h < 1/60:
        return f"{h*3600:.0f} s"
    if h < 1:
        return f"{h*60:.0f} min"
    if h < 40:
        return f"{h:.1f} h"
    if h < 160:
        return f"{h/40:.1f} wk ({h:.0f} h)"
    return f"{h/160:.1f} mo ({h:.0f} h)"

# ══════════════════════════════════════════════════════════════════════
#  Main loop over models
# ══════════════════════════════════════════════════════════════════════

all_results = {}

for model_key in MODELS:
    display_name = get_display_name(model_key)
    folder_name = safe_folder_name(display_name)
    out_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {display_name} ({model_key})")
    print(f"Output: {out_dir}/")
    print(f"{'='*60}")

    # Reset RNG per model for reproducibility
    rng = np.random.default_rng(SEED)

    # Build data
    task_data_dict = build_task_data(model_key)
    task_names = list(task_data_dict.keys())
    n_tasks = len(task_names)

    all_xs = np.concatenate([task_data_dict[t][0] for t in task_names])
    all_ys = np.concatenate([task_data_dict[t][1] for t in task_names])

    print(f"Tasks: {n_tasks}, Data points: {len(all_xs)}, "
          f"Successes: {all_ys.sum():.0f}/{len(all_ys)}")

    # ── Step 1: Fit full data ──────────────────────────────────────────

    res_full = minimize(neg_log_likelihood, [0.0, 0.0], args=(all_xs, all_ys),
                        method="L-BFGS-B", bounds=BOUNDS,
                        options={"maxiter": 2000})

    if not res_full.success:
        print(f"  Warning: full-data fit did not converge: {res_full.message}")

    x50_full = res_full.x[0]
    beta_full = -np.exp(res_full.x[1])
    h_full = x_to_hours(x50_full)
    print(f"Full-data fit: x50 = {x50_full:.3f} ({format_hours(h_full)}), "
          f"beta = {beta_full:.3f}")

    opt_start = res_full.x.copy()

    # ── Step 2: Bootstrap ──────────────────────────────────────────────

    print(f"Bootstrap ({N_BOOTSTRAP} samples, seed={SEED})...")

    boot_x50 = []
    boot_beta = []
    n_failed = 0

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n_tasks, size=n_tasks, replace=True)
        sampled_tasks = [task_names[j] for j in idx]

        boot_xs = np.concatenate([task_data_dict[t][0] for t in sampled_tasks])
        boot_ys = np.concatenate([task_data_dict[t][1] for t in sampled_tasks])

        res = minimize(neg_log_likelihood, opt_start, args=(boot_xs, boot_ys),
                       method="L-BFGS-B", bounds=BOUNDS,
                       options={"maxiter": 2000})

        if res.success:
            boot_x50.append(res.x[0])
            boot_beta.append(-np.exp(res.x[1]))
        else:
            n_failed += 1

    boot_x50 = np.array(boot_x50)
    boot_beta = np.array(boot_beta)
    n_good = len(boot_x50)

    print(f"  Successful: {n_good}/{N_BOOTSTRAP}" +
          (f" (skipped {n_failed})" if n_failed else ""))

    # Results
    x50_q50 = np.median(boot_x50)
    x50_q025 = np.percentile(boot_x50, 2.5)
    x50_q975 = np.percentile(boot_x50, 97.5)
    beta_q50 = np.median(boot_beta)
    beta_q025 = np.percentile(boot_beta, 2.5)
    beta_q975 = np.percentile(boot_beta, 97.5)

    h_q50 = x_to_hours(x50_q50)
    h_q025 = x_to_hours(x50_q025)
    h_q975 = x_to_hours(x50_q975)

    print(f"  x50  = {x50_q50:.3f} [{x50_q025:.3f}, {x50_q975:.3f}]")
    print(f"  time = {format_hours(h_q50)} [{format_hours(h_q025)}, {format_hours(h_q975)}]"
          f"  ({h_q50:.2f} h [{h_q025:.2f}, {h_q975:.2f}])")
    print(f"  beta = {beta_q50:.3f} [{beta_q025:.3f}, {beta_q975:.3f}]")

    # Save bootstrap samples
    np.savez(os.path.join(out_dir, "bootstrap_samples.npz"),
             x50=boot_x50, beta=boot_beta)

    model_meta = results["models"][model_key]
    all_results[model_key] = {
        "display_name": display_name,
        "release_date": model_meta.get("release_date"),
        "cost_per_run_usd": model_meta.get("cost_per_run_usd"),
        "mean_total_output_tokens": model_meta.get("mean_total_output_tokens"),
        "x50_q025": x50_q025, "x50_q50": x50_q50, "x50_q975": x50_q975,
        "x50_hours_q025": h_q025, "x50_hours_q50": h_q50, "x50_hours_q975": h_q975,
        "x50_time_q025": format_hours(h_q025),
        "x50_time_q50": format_hours(h_q50),
        "x50_time_q975": format_hours(h_q975),
        "beta_q025": beta_q025, "beta_q50": beta_q50, "beta_q975": beta_q975,
        "n_data": len(all_xs), "n_success": int(all_ys.sum()),
        "n_bootstrap_good": n_good,
    }

    # ── Curve data (shared by plots) ───────────────────────────────────

    x_plot = np.linspace(all_xs.min() - 0.2, all_xs.max() + 0.2, 300)
    n_draw = min(300, n_good)
    idx_draw = rng.choice(n_good, n_draw, replace=False)
    curves = np.zeros((n_draw, len(x_plot)))
    for i, idx in enumerate(idx_draw):
        eta = boot_beta[idx] * (x_plot - boot_x50[idx])
        curves[i] = expit(eta)

    median_curve = np.median(curves, axis=0)
    lo_curve = np.percentile(curves, 2.5, axis=0)
    hi_curve = np.percentile(curves, 97.5, axis=0)

    xlim = (x_plot[0], x_plot[-1])

    # ── Plot 1: Corner plot ────────────────────────────────────────────

    boot_samples = np.column_stack([boot_x50, boot_beta])

    # Set axis range based on 2.5%–97.5% percentiles, expanded by 2×
    corner_range = []
    for dim in range(2):
        q_lo = np.percentile(boot_samples[:, dim], 2.5)
        q_hi = np.percentile(boot_samples[:, dim], 97.5)
        span = q_hi - q_lo
        corner_range.append((q_lo - span, q_hi + span))

    fig = corner.corner(
        boot_samples,
        labels=[r"$x_{50}$ ($\log_{10}$ work-hours)", r"$\beta$"],
        quantiles=[0.025, 0.5, 0.975],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 11},
        range=corner_range,
    )
    fig.suptitle(f"Bootstrap posterior — {display_name}", y=1.02)
    plt.savefig(os.path.join(out_dir, "corner.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Histogram + fitted logistic ────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = 20
    bin_edges = np.linspace(all_xs.min() - 0.1, all_xs.max() + 0.1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    bin_fracs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (all_xs >= bin_edges[b]) & (all_xs < bin_edges[b + 1])
        n = mask.sum()
        bin_counts[b] = n
        if n > 0:
            bin_fracs[b] = all_ys[mask].mean()

    has_data = bin_counts > 0
    ax.bar(bin_centers[has_data], bin_fracs[has_data], width=bin_width * 0.85,
           color="C0", alpha=0.5, edgecolor="k", linewidth=0.5, zorder=2,
           label="Binned success rate")

    for b in np.where(has_data)[0]:
        n_success = int(bin_fracs[b] * bin_counts[b] + 0.5)
        ax.text(bin_centers[b], bin_fracs[b] + 0.02,
                f"{n_success}/{bin_counts[b]}", ha="center", va="bottom", fontsize=7, color="0.3")

    ax.plot(x_plot, median_curve, color="C1", lw=2, label="Median fit", zorder=4)
    ax.fill_between(x_plot, lo_curve, hi_curve, alpha=0.3, color="C1",
                    label="95% CI (bootstrap)", zorder=1)

    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(x50_q50, color="gray", ls=":", lw=1,
               label=f"$x_{{50}}$ = {x50_q50:.2f} ({format_hours(h_q50)})"
                     f" [{format_hours(h_q025)}, {format_hours(h_q975)}]")

    ax.set_ylabel("Success rate")
    cal_tag = " (calibrated)" if USE_CALIBRATION else ""
    ax.set_title(f"Bootstrap logistic fit — {display_name}{cal_tag}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, which="both")

    # Replace log10 ticks with human-readable time labels on the bottom axis
    ticks, tick_labels = get_time_ticks(xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Estimated human work-hours")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fit.png"), dpi=150)
    plt.close()

    print(f"  Saved {out_dir}/corner.png, {out_dir}/fit.png")

# ── Save summary ───────────────────────────────────────────────────────

summary_path = os.path.join(OUTPUT_BASE, "summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSummary saved to {summary_path}")
