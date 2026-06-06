"""
adversarial_probe.py - tamper-robustness of XGBoost vs ResNet-18 on QR codes.

Trains both models on one tier, then corrupts the TEST codes by inverting a random
fraction f of module-sized blocks (a realizable, decodability-limited tamper) and
reports each model's AUC as f rises. With pyzbar installed it also reports the
fraction of tampered codes that still DECODE (i.e. that an attacker could use).
This is the first step toward optimized adversarial attacks (your PhD direction).

Install: python -m pip install timm torch xgboost scikit-learn pillow numpy pandas
         optional decodability check: python -m pip install pyzbar
Usage:   python adversarial_probe.py --tier rendered --epochs 8 --grid 21
"""
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score


def load_gray(paths, size):
    A = np.full((len(paths), size, size), 255, np.uint8)
    for i, p in enumerate(paths):
        try:
            A[i] = np.asarray(Image.open(p).convert("L").resize((size, size)), np.uint8)
        except Exception:
            pass
    return A


def corrupt(A, f, grid, rng):
    """Invert a random fraction f of grid x grid blocks per image (approx module flips)."""
    if f <= 0:
        return A.copy()
    N, H, W = A.shape
    out = A.copy()
    bh, bw = H // grid, W // grid
    nblocks = grid * grid
    k = int(round(f * nblocks))
    for n in range(N):
        for b in rng.choice(nblocks, size=k, replace=False):
            r, c = divmod(int(b), grid)
            y0, x0 = r * bh, c * bw
            out[n, y0:y0 + bh, x0:x0 + bw] = 255 - out[n, y0:y0 + bh, x0:x0 + bw]
    return out


def decodable_fraction(A_uint8, sample=200):
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return None
    idx = np.linspace(0, len(A_uint8) - 1, min(sample, len(A_uint8))).astype(int)
    ok = sum(1 for i in idx if decode(Image.fromarray(A_uint8[i])))
    return ok / len(idx)


def train_models(Atr, ytr, epochs, batch, device):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, eval_metric="logloss", n_jobs=-1)
    xgb.fit(Atr.reshape(len(Atr), -1).astype(np.float32) / 255.0, ytr)

    import torch
    import torch.nn as nn
    import timm
    net = timm.create_model("resnet18", pretrained=True, num_classes=1, in_chans=3).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    lossf = nn.BCEWithLogitsLoss()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Atr)
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            x = torch.from_numpy(Atr[idx]).float().div(255.).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            opt.zero_grad()
            loss = lossf(net(((x - mean) / std)).squeeze(1), yt[idx].to(device))
            loss.backward(); opt.step()
        print(f"  resnet epoch {ep + 1}/{epochs}")
    return xgb, (net, mean, std)


def resnet_predict(model, A_uint8, batch, device):
    import torch
    net, mean, std = model
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(A_uint8), batch):
            x = torch.from_numpy(A_uint8[k:k + batch]).float().div(255.).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            out.append(torch.sigmoid(net(((x - mean) / std)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--tier", default="rendered")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--grid", type=int, default=21, help="blocks per side (~QR modules)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--levels", default="0,0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="robustness_curve.csv")
    args = ap.parse_args()

    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.img_size
    rng = np.random.default_rng(args.seed)
    print("device:", device, "| tier:", args.tier)

    meta = pd.read_csv(args.meta)
    sub = meta[meta.tier == args.tier]
    tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
    if len(tr) > args.max_train:
        tr = tr.sample(args.max_train, random_state=args.seed)
    Atr, ytr = load_gray(tr.path.values, size), tr.label.values
    Ate, yte = load_gray(te.path.values, size), te.label.values

    print("training models...")
    xgb, resnet = train_models(Atr, ytr, args.epochs, args.batch, device)

    rows = []
    for f in [float(x) for x in args.levels.split(",")]:
        Ac = corrupt(Ate, f, args.grid, rng)
        xa = roc_auc_score(yte, xgb.predict_proba(Ac.reshape(len(Ac), -1).astype(np.float32) / 255.0)[:, 1])
        ra = roc_auc_score(yte, resnet_predict(resnet, Ac, args.batch, device))
        dec = decodable_fraction(Ac)
        rows.append(dict(corruption=f, decodable=(round(dec, 3) if dec is not None else "n/a"),
                         xgb_auc=round(xa, 4), resnet_auc=round(ra, 4)))
        print(f"  f={f:.2f}  decodable={rows[-1]['decodable']}  XGB={xa:.4f}  ResNet={ra:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n==================== ROBUSTNESS CURVE ====================")
    print(df.to_string(index=False))
    print("saved", args.out)


if __name__ == "__main__":
    main()
