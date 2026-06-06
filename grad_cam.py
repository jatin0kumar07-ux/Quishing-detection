"""
grad_cam.py - visualize WHERE a ResNet-18 looks when classifying QR codes.

Trains a resnet18 briefly on one tier, then saves a grid of
(input | Grad-CAM overlay) for a few phishing and benign test codes.
This is the figure that explains *why* the CNN doesn't beat the pixel model.

Install: python -m pip install timm torch pillow opencv-python pandas numpy
Usage:   python grad_cam.py --tier rendered --epochs 8 --n-show 8 --out grad_cam.png
"""
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def load_gray(paths, size):
    A = np.full((len(paths), size, size), 255, np.uint8)
    for i, p in enumerate(paths):
        try:
            A[i] = np.asarray(Image.open(p).convert("L").resize((size, size)), np.uint8)
        except Exception:
            pass
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--tier", default="rendered")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--n-show", type=int, default=8)
    ap.add_argument("--out", default="grad_cam.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.img_size
    print("device:", device)

    meta = pd.read_csv(args.meta)
    sub = meta[meta.tier == args.tier]
    tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
    if len(tr) > args.max_train:
        tr = tr.sample(args.max_train, random_state=args.seed)
    Atr, ytr = load_gray(tr.path.values, size), tr.label.values

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def to_norm3(A_uint8):
        x = torch.from_numpy(A_uint8).float().div(255.).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        return (x - mean) / std

    net = timm.create_model("resnet18", pretrained=True, num_classes=1, in_chans=3).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Atr)
    print("training briefly...")
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(n)
        for k in range(0, n, args.batch):
            idx = perm[k:k + args.batch]
            opt.zero_grad()
            loss = lossf(net(to_norm3(Atr[idx])).squeeze(1), yt[idx].to(device))
            loss.backward(); opt.step()
        print(f"  epoch {ep + 1}/{args.epochs}")

    # Grad-CAM hooks on the last conv block
    store = {}
    net.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("act", o.detach()))
    net.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0].detach()))

    half = args.n_show // 2
    pick = pd.concat([te[te.label == 1].head(half), te[te.label == 0].head(args.n_show - half)])
    rows = []
    net.eval()
    for _, r in pick.iterrows():
        g = np.asarray(Image.open(r.path).convert("L").resize((size, size)), np.uint8)
        x = to_norm3(g[None])
        net.zero_grad()
        logit = net(x).squeeze()
        prob = torch.sigmoid(logit).item()
        logit.backward()
        A, G = store["act"], store["grad"]
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(1, keepdim=True).clamp(min=0)
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
        pair = np.hstack([base, overlay])
        pair = cv2.copyMakeBorder(pair, 24, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        lab = "PHISH" if r.label == 1 else "BENIGN"
        cv2.putText(pair, f"{lab}  p(phish)={prob:.2f}  [input | Grad-CAM]",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(pair)

    cv2.imwrite(args.out, np.vstack(rows))
    print("saved", args.out)


if __name__ == "__main__":
    main()
