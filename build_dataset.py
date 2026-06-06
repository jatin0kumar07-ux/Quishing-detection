"""
build_dataset.py  (v2) — render QR images from labeled URLs, in three tiers.

Fix vs v1: colours are applied by recolouring a black/white QR with PIL (robust),
instead of passing colour params into the generator (which caused a '%x format'
error). Adds a simple progress counter.

Tiers:
  synthetic_clean : fixed ECC=L, fixed module size, black/white (your paper's setting)
  rendered        : random ECC, random module size, random colours, optional logo
  captured        : 'rendered' + perspective/blur/lighting/JPEG/occlusion

Cross-generator test: --generator qrcode  (render a split with a different library)

Output:
  <out_dir>/images/<split>/<tier>/<label>/<id>.png
  <out_dir>/metadata.csv   columns: id,path,url,label,split,tier,generator,ecc,module,version

Usage:
  python build_dataset.py --urls-dir data/urls --out-dir data --tiers synthetic_clean rendered captured --limit 50
  python build_dataset.py --urls-dir data/urls --out-dir data --tiers synthetic_clean rendered captured
"""
import argparse
import io
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2

ECC = ["l", "m", "q", "h"]


# ----------------------------- QR generators (black/white) ----------------------
def render_segno(url, ecc, scale):
    import segno
    qr = segno.make(url, error=ecc)              # version auto from URL length + ECC
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=scale, border=4)   # default: black on white
    buf.seek(0)
    return Image.open(buf).convert("RGB"), qr.version


def render_qrcode(url, ecc, scale):
    import qrcode
    import qrcode.constants as C
    m = {"l": C.ERROR_CORRECT_L, "m": C.ERROR_CORRECT_M,
         "q": C.ERROR_CORRECT_Q, "h": C.ERROR_CORRECT_H}
    qr = qrcode.QRCode(error_correction=m[ecc], box_size=scale, border=4)
    qr.add_data(url); qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB"), qr.version


# ----------------------------- styling (PIL, robust) ----------------------------
def recolour(img, dark, light):
    """Map the black modules -> dark colour, white background -> light colour."""
    g = np.array(img.convert("L"))
    out = np.zeros((*g.shape, 3), np.uint8)
    mask = g < 128
    out[mask] = dark
    out[~mask] = light
    return Image.fromarray(out)


def maybe_logo(img, rng, p):
    if rng.random() > p:
        return img
    w, h = img.size; s = w // 6
    logo = Image.new("RGB", (s, s), tuple(int(x) for x in rng.integers(0, 256, 3)))
    img = img.copy(); img.paste(logo, ((w - s) // 2, (h - s) // 2))
    return img


# ----------------------------- capture noise (OpenCV) ---------------------------
def capture_noise(pil, rng):
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); h, w = img.shape[:2]
    if rng.random() < .6:
        m = .08; src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, src + (rng.uniform(-m, m, (4, 2)) * [w, h]).astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
    if rng.random() < .4:
        k = int(rng.choice([3, 5, 7])); img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < .5:
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(.7, 1.3)), beta=float(rng.uniform(-30, 30)))
    if rng.random() < .5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 90))])
        if ok: img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < .2:
        ow, oh = int(w * .08), int(h * .08)
        x, y = int(rng.integers(0, w - ow)), int(rng.integers(0, h - oh))
        img[y:y + oh, x:x + ow] = 255
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ----------------------------- features (for the XGBoost baseline) --------------
def pixel_vector(pil, size=64):
    g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return (g.reshape(-1) / 255.0).astype("float32")


# ----------------------------- one image ----------------------------------------
def render_one(url, tier, generator, rng):
    if tier == "synthetic_clean":
        ecc, scale = "l", 6
        img, version = (render_qrcode if generator == "qrcode" else render_segno)(url, ecc, scale)
        return img, ecc, scale, version
    ecc = str(rng.choice(ECC)); scale = int(rng.integers(4, 9))
    img, version = (render_qrcode if generator == "qrcode" else render_segno)(url, ecc, scale)
    dark = tuple(int(x) for x in rng.integers(0, 90, 3))
    light = tuple(int(x) for x in rng.integers(170, 256, 3))
    img = recolour(img, dark, light)
    img = maybe_logo(img, rng, p=0.3 if ecc in ("q", "h") else 0.0)
    if tier == "captured":
        img = capture_noise(img, rng)
    return img, ecc, scale, version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-dir", default="data/urls")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tiers", nargs="+", default=["synthetic_clean", "rendered", "captured"])
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--generator", default="segno", choices=["segno", "qrcode"])
    ap.add_argument("--limit", type=int, default=None, help="cap URLs per split (for testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed); random.seed(args.seed)
    rows, done, failed = [], 0, 0
    for split in args.split:
        df = pd.read_csv(os.path.join(args.urls_dir, f"{split}.csv"))
        if args.limit:
            df = df.head(args.limit)
        for tier in args.tiers:
            for i, r in df.iterrows():
                try:
                    img, ecc, scale, ver = render_one(r["url"], tier, args.generator, rng)
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print("render failed:", e)
                    continue
                d = os.path.join(args.out_dir, "images", split, tier, str(r["label"]))
                os.makedirs(d, exist_ok=True)
                uid = f"{split}_{tier}_{r['label']}_{i}"
                p = os.path.join(d, uid + ".png"); img.save(p)
                rows.append(dict(id=uid, path=p, url=r["url"], label=r["label"], split=split,
                                 tier=tier, generator=args.generator, ecc=ecc, module=scale, version=ver))
                done += 1
                if done % 500 == 0:
                    print(f"  ...{done} images rendered")
        print(f"done {split}")

    meta = pd.DataFrame(rows)
    mp = os.path.join(args.out_dir, "metadata.csv")
    if os.path.exists(mp):
        meta.to_csv(mp, mode="a", header=False, index=False)
    else:
        meta.to_csv(mp, index=False)
    print(f"wrote {len(meta)} images -> {mp}  ({failed} failed)")


if __name__ == "__main__":
    main()
