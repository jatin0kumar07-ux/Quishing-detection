"""
collect_urls_matched.py - remove the URL-length confound.

(1) Use PhishStorm's own legitimate full URLs as the benign class (not bare domains).
(2) Length-match benign vs phishing so URL length no longer separates the classes.
Then make a leakage-safe, label-stratified, domain-disjoint train/val/test split.

Install: python -m pip install pandas numpy scikit-learn scipy tldextract
Usage:
  python collect_urls_matched.py --phishstorm-csv urlset.csv --target-per-class 8000
"""
import argparse
import os
import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold


def detect_cols(df, url_col, label_col):
    cols = {c.lower(): c for c in df.columns}
    if url_col is None:
        for cand in ["url", "domain", "urls"]:
            if cand in cols:
                url_col = cols[cand]; break
    if label_col is None:
        for cand in ["label", "class", "result", "target"]:
            if cand in cols:
                label_col = cols[cand]; break
    if url_col is None or label_col is None:
        raise SystemExit(f"Could not detect columns from {list(df.columns)} - "
                         f"pass --url-col and --label-col explicitly.")
    return url_col, label_col


def reg_domain(u):
    e = tldextract.extract(str(u))
    return (e.domain + "." + e.suffix).lower() if e.suffix else e.domain.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phishstorm-csv", required=True)
    ap.add_argument("--url-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--target-per-class", type=int, default=8000)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out-dir", default="data/urls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # latin-1 reads any byte; on_bad_lines="skip" drops the few malformed rows
    df = pd.read_csv(args.phishstorm_csv, encoding="latin-1", on_bad_lines="skip")
    uc, lc = detect_cols(df, args.url_col, args.label_col)
    print(f"using url column='{uc}', label column='{lc}'")
    df = df[[uc, lc]].rename(columns={uc: "url", lc: "label"}).dropna()
    df["url"] = df["url"].astype(str).str.strip()
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(
            lambda v: 1 if any(k in str(v) for k in ["phish", "mal", "bad", "1"]) else 0)
    df["label"] = df["label"].astype(int)
    df = df[df["url"].str.len() > 0].drop_duplicates("url")
    df["len"] = df["url"].str.len()

    print("\nBefore matching:")
    for lab in [0, 1]:
        s = df[df.label == lab]["len"]
        print(f"  label {lab}: n={len(s):6d}  mean_len={s.mean():5.1f}  median={s.median():.0f}")

    # length-match using shared quantile bins
    edges = np.quantile(df["len"], np.linspace(0, 1, args.bins + 1))
    edges[0] -= 1; edges[-1] += 1
    df["bin"] = np.digitize(df["len"], edges[1:-1])
    keep = []
    for _, g in df.groupby("bin"):
        g0, g1 = g[g.label == 0], g[g.label == 1]
        k = min(len(g0), len(g1))
        if k:
            keep.append(g0.sample(k, random_state=args.seed))
            keep.append(g1.sample(k, random_state=args.seed))
    m = pd.concat(keep).reset_index(drop=True)

    # cap to target per class, preserving the (now matched) length distribution
    out = []
    for lab in [0, 1]:
        sub = m[m.label == lab]
        if len(sub) > args.target_per_class:
            frac = args.target_per_class / len(sub)
            sub = sub.groupby("bin", group_keys=False).sample(frac=frac, random_state=args.seed)
        out.append(sub)
    m = pd.concat(out).reset_index(drop=True)

    from scipy.stats import pointbiserialr
    r, _ = pointbiserialr(m["label"], m["len"])
    print("\nAfter matching:")
    for lab in [0, 1]:
        s = m[m.label == lab]["len"]
        print(f"  label {lab}: n={len(s):6d}  mean_len={s.mean():5.1f}  median={s.median():.0f}")
    print(f"  length-label correlation r = {r:+.3f}   (target: near 0)")

    # leakage-safe, label-stratified split by registered domain
    m["domain"] = m["url"].map(reg_domain)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    folds = list(sgkf.split(m, m["label"], groups=m["domain"]))
    test_idx, val_idx = folds[0][1], folds[1][1]
    used = set(test_idx.tolist()) | set(val_idx.tolist())
    train_idx = np.array([i for i in range(len(m)) if i not in used])

    os.makedirs(args.out_dir, exist_ok=True)
    print()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        part = m.iloc[idx][["url", "label", "domain"]].copy()
        part["split"] = name
        part.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"  {name}: {len(part)} urls, {part.label.mean():.0%} phishing, {part.domain.nunique()} domains")

    d_tr, d_va, d_te = (set(m.iloc[train_idx].domain), set(m.iloc[val_idx].domain), set(m.iloc[test_idx].domain))
    leak = (d_tr & d_va) | (d_tr & d_te) | (d_va & d_te)
    print("domain leakage:", "NONE" if not leak else f"{len(leak)} leaked")
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
