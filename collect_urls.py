"""
collect_urls.py  (v2) — large, balanced, leakage-safe URL collection.

Primary bulk source: PhishStorm — 96,018 URLs (48,009 legit + 48,009 phishing).
  Download urlset.csv from Aalto University (free):
    https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset/
  It downloads as a zip; unzip it to get 'urlset.csv'. In that file the 'domain'
  column is actually the URL, and 'label' is 0 = legitimate / 1 = phishing.

Optionally pools fresh phishing from the OpenPhish public feed, and extra benign
from Majestic Million.

Output: <out_dir>/{train,val,test}.csv  with columns: url,label,domain,split
Split : leakage-safe (no registered domain in more than one split) AND
        label-stratified (each split keeps ~50/50 phishing/benign).

RESPONSIBLE USE: phishing URLs are live and hostile; stored only as text to train a
detector. Never open, click, or scan them; work in an isolated environment.

Usage:
  # download + unzip urlset.csv into this folder first, then:
  python collect_urls.py --phishstorm-csv urlset.csv --target-per-class 8000 --use-openphish
"""
import argparse
import os
import numpy as np
import pandas as pd
import requests

MAJESTIC = "http://downloads.majestic.com/majestic_million.csv"
OPENPHISH = "https://openphish.com/feed.txt"


def add_scheme(u):
    u = str(u).strip()
    return u if u.startswith(("http://", "https://")) else "http://" + u


def reg_domain(url):
    """Registered domain; uses tldextract if available, else last-two-labels."""
    try:
        import tldextract
        e = tldextract.extract(url)
        return ".".join(p for p in [e.domain, e.suffix] if p)
    except Exception:
        from urllib.parse import urlparse
        h = urlparse(url).netloc.split(":")[0]
        p = h.split(".")
        return ".".join(p[-2:]) if len(p) >= 2 else h


def load_phishstorm(path):
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"could not read PhishStorm CSV at {path}")
    cols = {c.lower().strip(): c for c in df.columns}
    url_col = cols.get("url") or cols.get("domain") or list(df.columns)[0]
    label_col = cols.get("label") or cols.get("class")
    urls = df[url_col].astype(str).map(add_scheme)
    labels = pd.to_numeric(df[label_col], errors="coerce").round()
    out = pd.DataFrame({"url": urls, "label": labels}).dropna()
    out["label"] = out["label"].astype(int)
    return out[out["label"].isin([0, 1])]


def fetch_openphish(n=2000):
    try:
        r = requests.get(OPENPHISH, timeout=30); r.raise_for_status()
        u = [x.strip() for x in r.text.splitlines() if x.strip().startswith("http")]
        return pd.DataFrame({"url": list(dict.fromkeys(u))[:n], "label": 1})
    except Exception as e:
        print("OpenPhish skipped:", e)
        return pd.DataFrame(columns=["url", "label"])


def fetch_majestic(n):
    if n <= 0:
        return pd.DataFrame(columns=["url", "label"])
    mm = pd.read_csv(MAJESTIC)
    return pd.DataFrame({"url": ["http://" + d for d in mm["Domain"].dropna().head(n)],
                         "label": 0})


def split_stratified_grouped(df, seed):
    """Leakage-safe (group=domain) AND label-stratified ~72/14/14 split."""
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgk = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        fold = np.empty(len(df), dtype=int)
        for k, (_, idx) in enumerate(sgk.split(df, df["label"], groups=df["domain"])):
            fold[idx] = k
        d = df.assign(_f=fold)
        return (d[d._f >= 2].drop(columns="_f"),
                d[d._f == 1].drop(columns="_f"),
                d[d._f == 0].drop(columns="_f"))
    except Exception as e:
        print("StratifiedGroupKFold unavailable, using GroupShuffleSplit:", e)
        from sklearn.model_selection import GroupShuffleSplit
        g = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
        tr, tmp = next(g.split(df, groups=df["domain"]))
        train, temp = df.iloc[tr], df.iloc[tmp]
        g2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
        va, te = next(g2.split(temp, groups=temp["domain"]))
        return train, temp.iloc[va], temp.iloc[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phishstorm-csv", default=None, help="path to PhishStorm urlset.csv")
    ap.add_argument("--target-per-class", type=int, default=8000,
                    help="balanced count per class (capped by what's available)")
    ap.add_argument("--use-openphish", action="store_true",
                    help="add fresh phishing from the OpenPhish feed")
    ap.add_argument("--majestic-extra", type=int, default=0,
                    help="extra benign from Majestic Million")
    ap.add_argument("--out-dir", default="data/urls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pools = []
    if args.phishstorm_csv:
        ps = load_phishstorm(args.phishstorm_csv)
        print(f"PhishStorm: {len(ps)} rows "
              f"({(ps.label == 0).sum()} legit / {(ps.label == 1).sum()} phishing)")
        pools.append(ps)
    if args.use_openphish:
        op = fetch_openphish(); print(f"OpenPhish: {len(op)} phishing"); pools.append(op)
    if args.majestic_extra > 0:
        mj = fetch_majestic(args.majestic_extra); print(f"Majestic: {len(mj)} benign"); pools.append(mj)
    if not pools:
        raise SystemExit("No sources given. Pass --phishstorm-csv and/or --use-openphish.")

    df = pd.concat(pools, ignore_index=True)
    df["url"] = df["url"].astype(str).str.strip()
    df = df.dropna(subset=["url", "label"]).drop_duplicates("url")
    # drop any URL that appears under both labels
    nun = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(nun[nun > 1].index)]
    df["domain"] = df["url"].map(reg_domain)
    df = df[df["domain"].astype(bool)].reset_index(drop=True)

    # balance the two classes
    benign, phish = df[df.label == 0], df[df.label == 1]
    n = min(args.target_per_class, len(benign), len(phish))
    df = pd.concat([benign.sample(n, random_state=args.seed),
                    phish.sample(n, random_state=args.seed)]).reset_index(drop=True)
    print(f"balanced dataset: {len(df)} urls ({n} per class)")

    train, val, test = split_stratified_grouped(df, args.seed)

    tr, va, te = set(train.domain), set(val.domain), set(test.domain)
    assert not (tr & te) and not (tr & va) and not (va & te), "DOMAIN LEAKAGE"

    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.assign(split=name)
        part.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(part)} urls, {part.domain.nunique()} domains, "
              f"{part.label.mean():.0%} phishing")
    print("No domain leakage across splits ✔  (and label-stratified)")


if __name__ == "__main__":
    main()
