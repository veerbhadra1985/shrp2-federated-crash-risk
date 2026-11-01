"""
Figures for Steps 14-15 (leave-one-location-out calibration).

Fig 1  predicted vs actual conflict rate (3 s) for the 12 held-out locations
Fig 2  reliability diagrams (3 s, equal-mass bins) per held-out location
Fig 3  conformal false-alarm rate vs target at each held-out location
Usage : python src/plots/lolo_figures.py            (optionally --method isotonic)
Input : results/lolo_summary.csv, results/lolo_reliability.csv, results/lolo_conformal.csv
Output: results/figures/lolo_fig1_rates.(png|pdf), lolo_fig2_reliability.(png|pdf),
        lolo_fig3_false_alarms.(png|pdf)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

RES = Path("results"); FIG = RES / "figures"
# hand-set label offsets (points) for the crowded upper-right cluster; others default to (5, 3)
LABEL_OFFSETS = {"USA-EP0": (-36, 3), "USA-EP1": (-10, 7), "USA-FT": (-8, -12), "USA-EP": (5, 4),
                 "DEU-OF": (5, -9), "USA-GL": (-34, -3), "BGR-VA": (5, -7), "CHN-LN": (-38, -10),
                 "USA-MA": (5, 4), "CHN-ZS": (5, -9), "DEU-MT": (-8, 7)}
TYPE_STYLE = {"Roundabout": ("#1f77b4", "o"), "Intersection": ("#d62728", "s"), "Merging": ("#2ca02c", "^")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def short(loc):
    parts = loc.split("_")                 # e.g. DR_USA_Roundabout_FT -> USA-FT
    return f"{parts[1]}-{parts[3]}" if len(parts) >= 4 else loc


def kind(loc):
    return next((k for k in TYPE_STYLE if k in loc), "Intersection")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"saved {FIG / name}.png/.pdf")


def fig_rates(s, method):
    d = s[(s.method == method) & (s.horizon_s == 3)].copy()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo, hi = 0.05, 6
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--", label="perfect calibration")
    src = d.prev_source.mean() * 100
    ax.axhline(src, color="0.7", lw=1, ls=":", label=f"source average ({src:.1f}%)")
    for k, (c, m) in TYPE_STYLE.items():
        dk = d[d.held_out.map(kind) == k]
        ax.scatter(dk.prev_target * 100, dk.mean_pred * 100, c=c, marker=m, s=38, label=k, zorder=3)
    for _, r in d.iterrows():
        lab = short(r.held_out)
        ax.annotate(lab, (r.prev_target * 100, r.mean_pred * 100), fontsize=7,
                    xytext=LABEL_OFFSETS.get(lab, (5, 3)), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [0.1, 0.3, 1, 3]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Actual conflict rate at held-out location (%)")
    ax.set_ylabel(f"Mean predicted risk, {method} (%)")
    ax.set_title("Predicted vs actual risk at unseen locations (3 s)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "lolo_fig1_rates")


def fig_reliability(rel, s, method):
    r = rel[rel.method == method]; locs = sorted(r.held_out.unique())
    n = len(locs); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.2), squeeze=False)
    for ax, L in zip(axes.flat, locs):
        q = r[r.held_out == L].sort_values("bin")
        top = max(q.mean_pred.max(), q.observed.max()) * 1.1 + 1e-4
        ax.plot([0, top], [0, top], color="0.6", lw=0.8, ls="--")
        ax.plot(q.mean_pred, q.observed, marker="o", ms=3, lw=1, color=TYPE_STYLE[kind(L)][0])
        meta = s[(s.held_out == L) & (s.method == method) & (s.horizon_s == 3)].iloc[0]
        ax.set_title(f"{short(L)}\nECE {meta.ECE:.4f} | {int(meta.events)} events", fontsize=7.5)
        ax.set_xlim(0, top); ax.set_ylim(0, top); ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.supxlabel("Mean predicted risk (equal-mass bins)", fontsize=8.5)
    fig.supylabel("Observed conflict rate", fontsize=8.5)
    fig.suptitle(f"Reliability at unseen locations, 3 s ({method})", fontsize=10)
    fig.tight_layout()
    save(fig, "lolo_fig2_reliability")


def fig_false_alarms(c):
    d = c[c.horizon_s == 3]; locs = sorted(d.held_out.unique()); x = np.arange(len(locs))
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); w = 0.38
    for j, (al, col) in enumerate([(0.05, "#1f77b4"), (0.01, "#ff7f0e")]):
        v = d[d.alpha == al].set_index("held_out").reindex(locs).false_alarm_rate * 100
        ax.bar(x + (j - 0.5) * w, v, w, color=col, label=f"target {al:.0%}")
        ax.hlines(np.full(len(x), al * 100), x + (j - 1) * w, x + j * w, colors="k", lw=1.2)
        for xi, vi in zip(x, v):
            if vi > al * 100 * 1.001:
                ax.text(xi + (j - 0.5) * w, vi + 0.15, "✗", ha="center", fontsize=8, color="crimson")
    ax.set_xticks(x); ax.set_xticklabels([short(l) for l in locs], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("False-alarm rate at held-out location (%)")
    ax.set_title("Split-conformal false-alarm guarantee at unseen locations (3 s); black line = target, ✗ = violated")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "lolo_fig3_false_alarms")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="raw", choices=["raw", "temperature", "isotonic"])
    a = ap.parse_args()
    s = pd.read_csv(RES / "lolo_summary.csv"); rel = pd.read_csv(RES / "lolo_reliability.csv")
    c = pd.read_csv(RES / "lolo_conformal.csv")
    fig_rates(s, a.method); fig_reliability(rel, s, a.method); fig_false_alarms(c)


if __name__ == "__main__":
    main()