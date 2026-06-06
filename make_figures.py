"""
make_figures.py - generate all paper figures.

Produces:
  figure1_tiers.png       example benign/phishing codes across the three tiers
  figure2_results.png     XGBoost vs ResNet AUC per tier, with chance/density baseline
  figure3_confound.png    AUC before vs after length-matching (the confound effect)
  figure4_robustness.png  tamper-robustness curve (from robustness_curve.csv, if present)

(The Grad-CAM attention figure is produced separately by grad_cam.py.)

Edit the RESULTS / CONFOUND dicts below if you re-run any experiment.

Install: python -m pip install matplotlib pandas pillow numpy
Usage:   python make_figures.py --meta data/metadata.csv --curve robustness_curve.csv
"""
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- corrected, length-matched, within-generator AUC (edit if you re-run) ----
RESULTS = {
    "synthetic_clean": {"XGBoost": 0.755, "ResNet-18": 0.688},
    "rendered":        {"XGBoost": 0.683, "ResNet-18": 0.577},
    "captured":        {"XGBoost": 0.536, "ResNet-18": 0.523},
}
# XGBoost AUC before vs after removing the URL-length confound
CONFOUND = {
    "synthetic_clean": {"before": 0.878, "after": 0.755},
    "rendered":        {"before": 0.846, "after": 0.683},
    "captured":        {"before": 0.771, "after": 0.536},
}
CHANCE = 0.50  # density-only baseline after length-matching (~0.498)


def figure1_tiers(meta_path, out, generator):
    meta = pd.read_csv(meta_path)
    tiers = [t for t in ["synthetic_clean", "rendered", "captured"] if t in meta.tier.unique()]
    fig, axes = plt.subplots(len(tiers), 2, figsize=(4.2, 2.1 * len(tiers)))
    if len(tiers) == 1:
        axes = axes[None, :]
    for i, tier in enumerate(tiers):
        sub = meta[meta.tier == tier]
        if "generator" in meta.columns:
            sub = sub[sub.generator == generator]
        for j, (lab, name) in enumerate([(0, "Benign"), (1, "Phishing")]):
            ax = axes[i, j]
            rows = sub[sub.label == lab]
            if len(rows):
                ax.imshow(Image.open(rows.iloc[0].path).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                ax.set_ylabel(tier, fontsize=10)
    fig.suptitle("Figure 1. QR codes across realism tiers", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", out)


def grouped_bars(out, title, groups, series, ylabel="ROC-AUC", chance=None, colors=None):
    labels = list(groups)
    names = list(series)
    x = np.arange(len(labels)); w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, nm in enumerate(names):
        vals = [series[nm][g] for g in labels]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=nm,
                      color=(colors[i] if colors else None))
        try:
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        except Exception:
            pass
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", lw=1, label=f"chance ({chance:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel(ylabel); ax.set_ylim(0.45, 0.92)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", out)


def figure4_curve(curve_path, out):
    df = pd.read_csv(curve_path)
    dec = pd.to_numeric(df["decodable"], errors="coerce")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["corruption"], df["xgb_auc"], "o-", color="#1f77b4", label="XGBoost AUC")
    ax1.plot(df["corruption"], df["resnet_auc"], "s-", color="#d62728", label="ResNet-18 AUC")
    ax1.set_xlabel("Corruption fraction f"); ax1.set_ylabel("ROC-AUC")
    ax1.set_ylim(0.5, 0.9); ax1.grid(alpha=0.3); ax1.legend(loc="lower left")
    ax2 = ax1.twinx()
    ax2.plot(df["corruption"], dec, "^--", color="#2ca02c", alpha=0.75, label="Fraction decodable")
    ax2.set_ylabel("Fraction still decodable"); ax2.set_ylim(0, 1.02); ax2.legend(loc="upper right")
    fig.suptitle("Figure 4. Tamper robustness vs. decodability", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--curve", default="robustness_curve.csv")
    ap.add_argument("--generator", default="segno")
    args = ap.parse_args()

    try:
        figure1_tiers(args.meta, "figure1_tiers.png", args.generator)
    except Exception as e:
        print("Figure 1 skipped:", e)

    grouped_bars("figure2_results.png",
                 "Figure 2. Detection AUC by tier (confound-controlled)",
                 RESULTS.keys(), {"XGBoost": {t: RESULTS[t]["XGBoost"] for t in RESULTS},
                                  "ResNet-18": {t: RESULTS[t]["ResNet-18"] for t in RESULTS}},
                 chance=CHANCE, colors=["#1f77b4", "#d62728"])

    grouped_bars("figure3_confound.png",
                 "Figure 3. Effect of removing the URL-length confound (XGBoost)",
                 CONFOUND.keys(),
                 {"Before (confounded)": {t: CONFOUND[t]["before"] for t in CONFOUND},
                  "After (length-matched)": {t: CONFOUND[t]["after"] for t in CONFOUND}},
                 chance=CHANCE, colors=["#bbbbbb", "#1f77b4"])

    try:
        figure4_curve(args.curve, "figure4_robustness.png")
    except FileNotFoundError:
        print(f"Figure 4 skipped: {args.curve} not found (re-run adversarial_probe.py on the new data)")
    except Exception as e:
        print("Figure 4 skipped:", e)


if __name__ == "__main__":
    main()
