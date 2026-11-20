"""Download and unzip both datasets into data/raw/.

    python download_data.py            # both datasets (~940 MB total)
    python download_data.py --force    # re-download even if already present

No Kaggle account is needed: both are public Kaggle downloads.
If the download fails, download the two zips in a browser from the URLs below,
put them in data/downloads/ with the names shown, and run the script again (it will just unzip).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from common import DATA_DIR, RAW_DIR

DATASETS = {
    "garbage_classification": {
        # Kaggle: mostafaabla/garbage-classification (12 classes, 15,515 jpg)
        "url": "https://www.kaggle.com/api/v1/datasets/download/mostafaabla/garbage-classification",
        "page": "https://www.kaggle.com/datasets/mostafaabla/garbage-classification",
        "zip": "garbage-classification.zip",
        "approx_mb": 251,
    },
    "realwaste": {
        # RealWaste (UCI id 908, CC BY 4.0) - Kaggle mirror joebeachcapital/realwaste is much faster than UCI
        "url": "https://www.kaggle.com/api/v1/datasets/download/joebeachcapital/realwaste",
        "page": "https://archive.ics.uci.edu/dataset/908/realwaste",
        "zip": "realwaste.zip",
        "approx_mb": 689,
    },
}
DOWNLOAD_DIR = DATA_DIR / "downloads"


def download(url: str, dest: Path) -> None:
    from tqdm import tqdm

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "comvis-final-downloader"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        with tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    ap.add_argument("--only", choices=list(DATASETS), help="download just one dataset")
    args = ap.parse_args(argv)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in DATASETS.items():
        if args.only and name != args.only:
            continue
        out_dir = RAW_DIR / name
        zip_path = DOWNLOAD_DIR / d["zip"]
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            print(f"[skip] {name}: already extracted in {out_dir}")
            continue
        if args.force or not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            print(f"[download] {name} (~{d['approx_mb']} MB) from {d['url']}")
            try:
                download(d["url"], zip_path)
            except Exception as e:  # network problems, proxies, ...
                print(f"\nDownload failed: {e}\nGet it manually from {d['page']} and save it as {zip_path}")
                sys.exit(1)
        print(f"[unzip] {zip_path.name} -> {out_dir}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    print("Done. Next: python prepare_data.py")


if __name__ == "__main__":
    main()
