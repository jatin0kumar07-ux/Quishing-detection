"""
multiseed_eval.py - multi-seed evaluation with bootstrap confidence intervals.

Runs the XGBoost-vs-backbone comparison over several random seeds and reports,
per tier:
  - mean +/- std AUC ACROSS seeds   (training/seed variance)
  - mean bootstrap 95% CI on the test set  (test-sampling variance)

Reuses functions from train_eval.py -- keep both files in the same folder.

Usage:
  python multiseed_eval.py --train-meta data/metadata.csv --seeds 0 1 2 --epochs 15 --max-per-tier 8000
  # cross-generator or real test set: add  --test-meta data_xgen/metadata.csv  (or real/metadata.csv)
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import train_eval as TE


def boot_ci(y, p, B, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y); a = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", default="data/metadata.csv")
    ap.add_argument("--test-meta", default=None)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-per-tier", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_meta_path = args.test_meta or args.train_meta
    print(f"device: {device} | model: {args.model} | seeds: {args.seeds}")
    print(f"train: {args.train_meta} | test: {test_meta_path}")

    train_meta = pd.read_csv(args.train_meta)
    test_meta = pd.read_csv(test_meta_path)
    tiers = [t for t in train_meta.tier.unique() if t in set(test_meta.tier.unique())]

    rows = []
    for tier in tiers:
        trf = train_meta[(train_meta.tier == tier) & (train_meta.split == "train")]
        te = test_meta[(test_meta.tier == tier) & (test_meta.split == "test")]
        if len(trf) < 50 or len(te) < 20 or te.label.nunique() < 2:
            print(f"[{tier}] skipped (insufficient data)"); continue
        yte = te.label.values
        Ate = TE.load_gray(te.path.values, args.img_size)
        Xte = Ate.reshape(len(Ate), -1).astype(np.float32) / 255.0

        xgb_aucs, cnn_aucs, xgb_ci, cnn_ci = [], [], [], []
        for s in args.seeds:
            np.random.seed(s); torch.manual_seed(s)
            tr = trf.sample(min(args.max_per_tier, len(trf)), random_state=s) if args.max_per_tier else trf
            ytr = tr.label.values
            Atr = TE.load_gray(tr.path.values, args.img_size)
            Xtr = Atr.reshape(len(Atr), -1).astype(np.float32) / 255.0
            xgb_p = TE.run_xgb(Xtr, ytr, Xte)
            cnn_p = TE.run_backbone(Atr, ytr, Ate, args.model, args.epochs, args.batch, args.lr, device)
            xa, ca = roc_auc_score(yte, xgb_p), roc_auc_score(yte, cnn_p)
            xgb_aucs.append(xa); cnn_aucs.append(ca)
            xgb_ci.append(boot_ci(yte, xgb_p, args.bootstrap, s))
            cnn_ci.append(boot_ci(yte, cnn_p, args.bootstrap, s))
            print(f"  [{tier}] seed {s}: XGB={xa:.4f}  {args.model}={ca:.4f}")

        def agg(aucs, cis):
            return (np.mean(aucs), np.std(aucs),
                    np.mean([c[0] for c in cis]), np.mean([c[1] for c in cis]))
        xm, xsd, xlo, xhi = agg(xgb_aucs, xgb_ci)
        cm, csd, clo, chi = agg(cnn_aucs, cnn_ci)
        rows.append(dict(tier=tier, n_seeds=len(args.seeds), n_test=len(te),
                         xgb=f"{xm:.3f}+/-{xsd:.3f}", xgb_95CI=f"[{xlo:.3f},{xhi:.3f}]",
                         cnn=f"{cm:.3f}+/-{csd:.3f}", cnn_95CI=f"[{clo:.3f},{chi:.3f}]"))
        print(f"[{tier}] XGB {xm:.3f}+/-{xsd:.3f} CI[{xlo:.3f},{xhi:.3f}]  |  "
              f"{args.model} {cm:.3f}+/-{csd:.3f} CI[{clo:.3f},{chi:.3f}]")

    print("\n==================== MULTI-SEED SUMMARY ====================")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nxgb/cnn = mean +/- std over seeds; 95CI = mean bootstrap 95% CI on the test set.")


if __name__ == "__main__":
    main()
