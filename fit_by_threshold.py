"""
Per-threshold-group logistic fits for a single model.

Three groups: easy (25%+50%), medium (70%), hard (90%+95%).
Same method as fit_bootstrap.py but split by threshold group.
Combined plot with step-function histograms and median logistic curves.
"""

import json
import numpy as np
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

MODEL_KEY = "gpt-5-2025-08-07"

USE_CALIBRATION = False

THRESHOLDS = [0.25, 0.50, 0.70, 0.90, 0.95]
THRESHOLD_LABELS = ["25%", "50%", "70%", "90%", "95%"]
ESTIMATOR_MODELS = ["gpt-5.2:medium", "gemini-3-pro-preview", "claude-opus-4.5", "grok-4-07-09"]

# Threshold groups: (label, list of (thresh, thresh_label) pairs)
GROUPS = [
    ("25%+50%", [(0.25, "25%"), (0.50, "50%")]),
    ("70%",     [(0.70, "70%")]),
    ("90%+95%", [(0.90, "90%"), (0.95, "95%")]),
]
GROUP_COLORS = ["C0", "C2", "C3"]

N_BOOTSTRAP = 5000
N_BINS = 20
SEED = 42
BOUNDS = [(-3, 7), (np.log(1e-6), np.log(50))]

# Calibration offsets converted from ln to log10
CAL_OFFSETS = {}
for t in THRESHOLD_LABELS:
    if USE_CALIBRATION:
        CAL_OFFSETS[t] = calibration["by_threshold"][t]["avg_log_diff"] / LN10
    else:
        CAL_OFFSETS[t] = 0.0

# Time ticks (work-hours convention)
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
    x50, v = params
    beta = -np.exp(v)
    eta = beta * (xs - x50)
    ll = ys * eta - np.logaddexp(0, eta)
    return -np.sum(ll)


def get_time_ticks(xlim):
    ticks, labels = [], []
    for hours, label in TIME_TICKS:
        lx = np.log10(hours)
        if xlim[0] <= lx <= xlim[1]:
            ticks.append(lx)
            labels.append(label)
    return ticks, labels


def x_to_hours(x):
    return 10**x


def format_hours(h):
    if h < 1/60:
        return f"{h*3600:.0f} s"
    if h < 1:
        return f"{h*60:.0f} min"
    if h < 40:
        return f"{h:.1f} h"
    if h < 160:
        return f"{h/40:.1f} wk ({h:.0f} h)"
    return f"{h/160:.1f} mo ({h:.0f} h)"


def build_group_data(model_key, thresh_pairs):
    """Build (task_name -> (xs, ys)) for a group of thresholds."""
    model_tasks = results["models"][model_key]["tasks"]
    task_data = {}

    for task_name, tdata in model_tasks.items():
        if task_name not in estimates:
            print(f"  WARNING: task '{task_name}' not in estimates, skipping")
            continue

        task_estimates = estimates[task_name]["estimates"]
        runs = tdata["runs"]
        xs, ys = [], []

        for thresh, thresh_label in thresh_pairs:
            cal = CAL_OFFSETS[thresh_label]

            for estimator in ESTIMATOR_MODELS:
                if estimator not in task_estimates:
                    raise KeyError(f"Estimator '{estimator}' missing for task '{task_name}'")
                if thresh_label not in task_estimates[estimator]:
                    raise KeyError(f"Threshold '{thresh_label}' missing for task '{task_name}', estimator '{estimator}'")

                hours = task_estimates[estimator][thresh_label]["hours"]
                if hours <= 0:
                    print(f"  WARNING: hours={hours} for {task_name}/{estimator}/{thresh_label}, skipping")
                    continue
                x = np.log10(hours) + cal

                for run in runs:
                    y = 1 if run["score"] >= thresh else 0
                    xs.append(x)
                    ys.append(y)

        if xs:
            task_data[task_name] = (np.array(xs), np.array(ys))

    return task_data


# ══════════════════════════════════════════════════════════════════════
#  Fit each group
# ══════════════════════════════════════════════════════════════════════

display_name = results["models"][MODEL_KEY].get("display_name", MODEL_KEY)
print(f"Model: {display_name} ({MODEL_KEY})")
cal_tag = " (calibrated)" if USE_CALIBRATION else ""

all_xs_global = []
group_results = {}

# Get joint fit as initialization: fit all thresholds combined (like fit_bootstrap.py)
joint_task_data = build_group_data(MODEL_KEY, list(zip(THRESHOLDS, THRESHOLD_LABELS)))
joint_xs = np.concatenate([joint_task_data[t][0] for t in joint_task_data])
joint_ys = np.concatenate([joint_task_data[t][1] for t in joint_task_data])
res_joint = minimize(neg_log_likelihood, [0.0, 0.0], args=(joint_xs, joint_ys),
                     method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 2000})
joint_init = res_joint.x.copy()
print(f"Joint init: x50={joint_init[0]:.3f}, beta={-np.exp(joint_init[1]):.3f}")

for (group_label, thresh_pairs), color in zip(GROUPS, GROUP_COLORS):
    print(f"\n  Group: {group_label}")

    task_data = build_group_data(MODEL_KEY, thresh_pairs)
    task_names = list(task_data.keys())
    n_tasks = len(task_names)

    all_xs = np.concatenate([task_data[t][0] for t in task_names])
    all_ys = np.concatenate([task_data[t][1] for t in task_names])
    all_xs_global.append(all_xs)

    print(f"    Data points: {len(all_xs)}, Successes: {all_ys.sum():.0f}/{len(all_ys)}")

    # Full-data fit, initialized from joint fit
    res_full = minimize(neg_log_likelihood, joint_init, args=(all_xs, all_ys),
                        method="L-BFGS-B", bounds=BOUNDS,
                        options={"maxiter": 2000})
    if not res_full.success:
        print(f"    Warning: fit did not converge: {res_full.message}")

    opt_start = res_full.x.copy()

    # Bootstrap
    thresh_rng = np.random.default_rng(SEED)
    boot_x50, boot_beta = [], []

    for _ in range(N_BOOTSTRAP):
        idx = thresh_rng.choice(n_tasks, size=n_tasks, replace=True)
        sampled = [task_names[j] for j in idx]
        bx = np.concatenate([task_data[t][0] for t in sampled])
        by = np.concatenate([task_data[t][1] for t in sampled])

        res = minimize(neg_log_likelihood, joint_init, args=(bx, by),
                       method="L-BFGS-B", bounds=BOUNDS,
                       options={"maxiter": 2000})
        if res.success:
            boot_x50.append(res.x[0])
            boot_beta.append(-np.exp(res.x[1]))

    boot_x50 = np.array(boot_x50)
    boot_beta = np.array(boot_beta)

    x50_q50 = np.median(boot_x50)
    x50_q025 = np.percentile(boot_x50, 2.5)
    x50_q975 = np.percentile(boot_x50, 97.5)

    h_q50 = x_to_hours(x50_q50)
    print(f"    x50 = {x50_q50:.3f} ({format_hours(h_q50)}) "
          f"[{format_hours(x_to_hours(x50_q025))}, {format_hours(x_to_hours(x50_q975))}]")

    group_results[group_label] = {
        "all_xs": all_xs,
        "all_ys": all_ys,
        "boot_x50": boot_x50,
        "boot_beta": boot_beta,
        "x50_q50": x50_q50,
        "color": color,
    }

# ══════════════════════════════════════════════════════════════════════
#  Combined plot
# ══════════════════════════════════════════════════════════════════════

all_xs_concat = np.concatenate(all_xs_global)
x_lo = all_xs_concat.min() - 0.3
x_hi = all_xs_concat.max() + 0.3
x_plot = np.linspace(x_lo, x_hi, 400)
xlim = (x_lo, x_hi)

# Shared bin edges across all groups
bin_edges = np.linspace(x_lo + 0.2, x_hi - 0.2, N_BINS + 1)

fig, ax = plt.subplots(figsize=(12, 7))

for (group_label, _), color in zip(GROUPS, GROUP_COLORS):
    gr = group_results[group_label]
    xs, ys = gr["all_xs"], gr["all_ys"]
    boot_x50 = gr["boot_x50"]
    boot_beta = gr["boot_beta"]

    # Step-plot histogram using shared bins
    bin_fracs = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = (xs >= bin_edges[b]) & (xs < bin_edges[b + 1])
        n = mask.sum()
        if n > 0:
            bin_fracs[b] = ys[mask].mean()

    # Build step coordinates, skipping empty bins
    for b in range(N_BINS):
        if not np.isnan(bin_fracs[b]):
            ax.plot([bin_edges[b], bin_edges[b + 1]],
                    [bin_fracs[b], bin_fracs[b]],
                    color=color, alpha=0.6, lw=3.0, zorder=2)

    # Median logistic curve (no CI bands)
    n_draw = min(300, len(boot_x50))
    draw_idx = np.random.default_rng(SEED).choice(len(boot_x50), n_draw, replace=False)
    curves = np.zeros((n_draw, len(x_plot)))
    for i, idx in enumerate(draw_idx):
        eta = boot_beta[idx] * (x_plot - boot_x50[idx])
        curves[i] = expit(eta)

    median_curve = np.median(curves, axis=0)

    x50_q50 = gr["x50_q50"]
    h_str = format_hours(x_to_hours(x50_q50))
    ax.plot(x_plot, median_curve, color=color, lw=2.5, zorder=4,
            label=f"{group_label} — x₅₀ = {h_str}")

ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_ylabel("Success rate", fontsize=12)
ax.set_title(f"Logistic fit by threshold group — {display_name}{cal_tag}", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(xlim)
ax.grid(True, alpha=0.3, which="both")

ticks, tick_labels = get_time_ticks(xlim)
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_xlabel("Estimated human work-hours", fontsize=12)

plt.tight_layout()

output_base = "results_cal" if USE_CALIBRATION else "results_no_cal"
out_path = f"{output_base}/by_threshold_{display_name.replace(' ', '_')}.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved {out_path}")
