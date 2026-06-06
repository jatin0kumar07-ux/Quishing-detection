
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score


def load_gray(paths, size):
    A = np.full((len(paths), size, size), 255, np.uint8)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("L").resize((size, size))
            A[i] = np.asarray(im, np.uint8)
        except Exception:
            pass
    return A


def mcnemar(y, pa, pb):
    a, b = (pa == y), (pb == y)
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b)); n = n01 + n10
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return binomtest(min(n01, n10), n, 0.5).pvalue
    except Exception:
        from scipy.stats import binom
        return min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5))


def run_xgb(Xtr, ytr, Xte):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, eval_metric="logloss", n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def run_backbone(Atr, ytr, Ate, model_name, epochs, batch, lr, device):
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self, in_ch=3):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            return self.head(self.f(x).flatten(1))

    if model_name == "smallcnn":
        net = SmallCNN(3).to(device)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
    else:
        import timm
        net = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=3).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr)

    lossf = nn.BCEWithLogitsLoss()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32)

    def to_batch(A, idx, train):
        xb = torch.from_numpy(A[idx]).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        if train:
            b = xb.shape[0]
            c = 0.8 + 0.4 * torch.rand(b, 1, 1, 1)
            br = 0.2 * (torch.rand(b, 1, 1, 1) - 0.5)
            xb = ((xb - 0.5) * c + 0.5 + br).clamp(0, 1)
            H, W = xb.shape[2], xb.shape[3]
            eh, ew = int(H * 0.15), int(W * 0.15)
            for j in range(b):
                if torch.rand(1).item() < 0.3:
                    y0 = int(torch.randint(0, H - eh + 1, (1,))); x0 = int(torch.randint(0, W - ew + 1, (1,)))
                    xb[j, :, y0:y0 + eh, x0:x0 + ew] = 1.0
        xb = xb.to(device)
        return (xb - mean) / std

    n = len(Atr)
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            yb = yt[idx].to(device)
            opt.zero_grad()
            loss = lossf(net(to_batch(Atr, idx, True)).squeeze(1), yb)
            loss.backward()
            opt.step()
        print(f"    epoch {ep + 1}/{epochs} done")

    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(Ate), batch):
            idx = np.arange(k, min(k + batch, len(Ate)))
            out.append(torch.sigmoid(net(to_batch(Ate, idx, False)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", default="data/metadata.csv")
    ap.add_argument("--test-meta", default=None, help="if set, cross-generator test source")
    ap.add_argument("--model", default="resnet18", help="resnet18 / any timm name / smallcnn")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-per-tier", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_meta_path = args.test_meta or args.train_meta
    print(f"device: {device} | model: {args.model}")
    print(f"train-meta: {args.train_meta}")
    print(f"test-meta : {test_meta_path}" + ("   (CROSS-GENERATOR)" if args.test_meta else ""))

    train_meta = pd.read_csv(args.train_meta)
    test_meta = pd.read_csv(test_meta_path)
    tiers = [t for t in train_meta.tier.unique() if t in set(test_meta.tier.unique())]

    results = []
    for tier in tiers:
        tr = train_meta[(train_meta.tier == tier) & (train_meta.split == "train")]
        te = test_meta[(test_meta.tier == tier) & (test_meta.split == "test")]
        if len(tr) < 50 or len(te) < 20 or te.label.nunique() < 2:
            print(f"[{tier}] not enough data - skipping"); continue
        if args.max_per_tier and len(tr) > args.max_per_tier:
            tr = tr.sample(args.max_per_tier, random_state=args.seed)
        ytr, yte = tr.label.values, te.label.values
        print(f"\n[{tier}] train={len(tr)} test={len(te)}")

        print("  loading images...")
        Atr = load_gray(tr.path.values, args.img_size)
        Ate = load_gray(te.path.values, args.img_size)

        print("  XGBoost (pixel vectors)...")
        xgb_p = run_xgb(Atr.reshape(len(Atr), -1).astype(np.float32) / 255.0,
                        ytr, Ate.reshape(len(Ate), -1).astype(np.float32) / 255.0)

        print(f"  {args.model}...")
        cnn_p = run_backbone(Atr, ytr, Ate, args.model, args.epochs, args.batch, args.lr, device)

        xgb_auc, cnn_auc = roc_auc_score(yte, xgb_p), roc_auc_score(yte, cnn_p)
        xp, cp = (xgb_p > .5).astype(int), (cnn_p > .5).astype(int)
        p = mcnemar(yte, xp, cp)
        winner = "CNN" if cnn_auc > xgb_auc else "XGBoost"
        results.append(dict(tier=tier, n_train=len(tr), n_test=len(te),
                            xgb_auc=round(xgb_auc, 4), cnn_auc=round(cnn_auc, 4),
                            xgb_f1=round(f1_score(yte, xp), 3), cnn_f1=round(f1_score(yte, cp), 3),
                            mcnemar_p=round(p, 4), higher_auc=winner))
        print(f"  -> XGB AUC={xgb_auc:.4f}  {args.model} AUC={cnn_auc:.4f}  McNemar p={p:.4f}  (higher: {winner})")

    print("\n==================== SUMMARY ====================")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
