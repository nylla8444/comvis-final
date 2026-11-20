"""Build the 3-bin dataset and the fixed train/val/test splits used by ALL three methods.

    python prepare_data.py                 # default: up to 2,500 images per bin, 70/15/15 split
    python prepare_data.py --per-bin 3000  # use more images

Steps
1. Find the source class folders of both datasets under data/raw/ (any nesting depth).
2. Map every source class to one of the 3 bins (table SOURCE_TO_BIN below). Batteries are dropped:
   under RA 9003 they are special/hazardous waste, not one of the three household bins.
3. Remove duplicate images (256-bit difference hash). A duplicate that appears with two different
   bins is dropped completely; otherwise one copy is kept. This stops the same picture from landing
   in both train and test.
4. Balance: each bin is capped at --per-bin images, sampled evenly across its source classes
   ("water filling"), so e.g. the 5,325 clothes photos cannot dominate the residual bin.
5. Split 70/15/15, stratified by source class, with a fixed seed.
6. Save every selected image as a 256x256 centre-cropped JPEG in data/processed/ and write
   splits/train.csv, splits/val.csv, splits/test.csv and splits/summary.json.

Optional local photos (your own phone pictures, used ONLY as an extra test set):
    data/raw/local/biodegradable/*.jpg
    data/raw/local/recyclable/*.jpg
    data/raw/local/residual/*.jpg        -> splits/local_test.csv
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import CLASS_TO_IDX, CLASSES, PROCESSED_DIR, RAW_DIR, ROOT, SEED, SPLITS_DIR, preprocess_pil, save_json

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# source dataset folder under data/raw  ->  {source class folder name: bin}
SOURCE_TO_BIN = {
    "garbage_classification": {  # Kaggle mostafaabla/garbage-classification
        "biological": "biodegradable",
        "paper": "recyclable",
        "cardboard": "recyclable",
        "metal": "recyclable",
        "plastic": "recyclable",
        "brown-glass": "recyclable",
        "green-glass": "recyclable",
        "white-glass": "recyclable",
        "trash": "residual",
        "shoes": "residual",
        "clothes": "residual",
        "battery": None,  # special / hazardous waste -> dropped
    },
    "realwaste": {  # UCI RealWaste (landfill photos, Wollongong, Australia)
        "Food Organics": "biodegradable",
        "Vegetation": "biodegradable",
        "Cardboard": "recyclable",
        "Glass": "recyclable",
        "Metal": "recyclable",
        "Paper": "recyclable",
        "Plastic": "recyclable",
        "Miscellaneous Trash": "residual",
        "Textile Trash": "residual",
    },
}
LOCAL_ALIASES = {
    "biodegradable": "biodegradable", "nabubulok": "biodegradable", "bio": "biodegradable",
    "recyclable": "recyclable", "recyclables": "recyclable",
    "residual": "residual", "nonbiodegradable": "residual", "dinabubulok": "residual",
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_class_dirs(root: Path, wanted: list[str]) -> dict[str, Path]:
    """Walk `root` and return {class name: folder} for every wanted class folder that contains images."""
    want = {norm(w): w for w in wanted}
    found: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        key = norm(Path(dirpath).name)
        if key in want and any(Path(f).suffix.lower() in IMG_EXT for f in filenames):
            found.setdefault(want[key], Path(dirpath))
    return found


def dhash(path: Path, size: int = 16) -> str | None:
    """256-bit difference hash (hex): grey 17x16 thumbnail, compare horizontal neighbours.
    Identical or re-saved/resized copies get the same hash; different photos almost never do."""
    try:
        with Image.open(path) as im:
            im.draft("L", (128, 128))  # fast JPEG decode at reduced size
            a = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def collect(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for source, mapping in SOURCE_TO_BIN.items():
        src_root = raw_dir / source
        if not src_root.exists():
            raise SystemExit(f"Missing {src_root}. Run `python download_data.py` first.")
        dirs = find_class_dirs(src_root, list(mapping))
        missing = [c for c in mapping if c not in dirs]
        if missing:
            raise SystemExit(f"{source}: could not find class folders {missing} under {src_root}")
        for cls, folder in dirs.items():
            bin_ = mapping[cls]
            if bin_ is None:
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"orig_path": str(f), "source": source, "source_class": cls, "label": bin_})
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    hashes = [dhash(Path(p)) for p in tqdm(df["orig_path"], desc="hashing (duplicate check)")]
    df = df.assign(hash=hashes)
    unreadable = int(df["hash"].isna().sum())
    df = df[df["hash"].notna()].copy()
    labels_per_hash = df.groupby("hash")["label"].nunique()
    conflicting = set(labels_per_hash[labels_per_hash > 1].index)
    n_conflict = int(df["hash"].isin(conflicting).sum())
    df = df[~df["hash"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates("hash", keep="first")
    stats = {"unreadable": unreadable, "dropped_conflicting_label_copies": n_conflict, "dropped_duplicates": before - len(df)}
    return df.reset_index(drop=True), stats


def water_fill(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Split `target` as evenly as possible across groups, never taking more than a group has."""
    alloc = {k: 0 for k in sizes}
    remaining, open_ = target, {k for k, v in sizes.items() if v > 0}
    while remaining > 0 and open_:
        share = max(1, remaining // len(open_))
        for k in sorted(open_):
            take = min(share, sizes[k] - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] == sizes[k]:
                open_.discard(k)
            if remaining == 0:
                break
    return alloc


def balance(df: pd.DataFrame, per_bin: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for bin_ in CLASSES:
        sub = df[df["label"] == bin_]
        groups = {k: g for k, g in sub.groupby(["source", "source_class"])}
        alloc = water_fill({k: len(g) for k, g in groups.items()}, per_bin)
        for k, g in groups.items():
            idx = rng.permutation(len(g))[: alloc[k]]
            parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def split(df: pd.DataFrame, val: float, test: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "train"
    for _, g in df.groupby(["source", "source_class"]):
        idx = rng.permutation(g.index.to_numpy())
        n_test, n_val = int(round(len(idx) * test)), int(round(len(idx) * val))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test : n_test + n_val], "split"] = "val"
    return df


def save_processed(df: pd.DataFrame, out_dir: Path, size: int) -> pd.DataFrame:
    paths = []
    for row in tqdm(df.itertuples(), total=len(df), desc="resizing to 256x256"):
        stem = f"{norm(row.source)[:8]}_{norm(row.source_class)}_{norm(Path(row.orig_path).stem)}_{row.hash[-6:]}"
        dest = out_dir / row.label / f"{stem}.jpg"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(row.orig_path) as im:
                preprocess_pil(im, size).save(dest, quality=95)
        paths.append(dest.relative_to(ROOT).as_posix())
    df = df.copy()
    df["path"] = paths
    return df


def write_csv(df: pd.DataFrame, name: str) -> None:
    out = df.assign(label_idx=df["label"].map(CLASS_TO_IDX))
    cols = ["path", "label", "label_idx", "source", "source_class", "hash", "orig_path"]
    out["orig_path"] = [Path(p).relative_to(ROOT).as_posix() if Path(p).is_relative_to(ROOT) else p for p in out["orig_path"]]
    out[cols].to_csv(SPLITS_DIR / f"{name}.csv", index=False)


def prepare_local(raw_dir: Path, size: int) -> int:
    local = raw_dir / "local"
    if not local.exists():
        return 0
    rows = []
    for d in local.iterdir():
        bin_ = LOCAL_ALIASES.get(norm(d.name))
        if d.is_dir() and bin_:
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"orig_path": str(f), "source": "local", "source_class": d.name, "label": bin_, "hash": str(dhash(f))})
    if not rows:
        return 0
    df = save_processed(pd.DataFrame(rows), PROCESSED_DIR / "local", size)
    write_csv(df, "local_test")
    return len(df)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--per-bin", type=int, default=2500, help="max images per bin (default 2500)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = collect(args.raw)
    print(f"found {len(df)} usable images (batteries excluded)")
    print(df.groupby(["label", "source", "source_class"]).size().to_string(), "\n")

    df, dup_stats = deduplicate(df)
    print("duplicate check:", dup_stats)

    df = balance(df, args.per_bin, rng)
    df = split(df, args.val, args.test, rng)
    df = save_processed(df, PROCESSED_DIR, args.size)
    for name in ["train", "val", "test"]:
        write_csv(df[df["split"] == name], name)

    n_local = prepare_local(args.raw, args.size)

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    summary = {
        "total_images": int(len(df)),
        "per_bin_cap": args.per_bin,
        "seed": args.seed,
        "split_fractions": {"train": 1 - args.val - args.test, "val": args.val, "test": args.test},
        "duplicates": dup_stats,
        "counts": {s: {c: int(counts.loc[s, c]) if c in counts.columns else 0 for c in CLASSES} for s in counts.index},
        "source_classes": {f"{s}/{c}": int(n) for (s, c), n in df.groupby(["source", "source_class"]).size().items()},
        "local_test_images": n_local,
    }
    save_json(summary, SPLITS_DIR / "summary.json")
    print("\nimages per split and bin:\n", counts.to_string())
    print(f"\nTOTAL = {len(df)} images" + ("" if len(df) >= 5000 else "  (WARNING: below the 5,000 minimum!)"))
    if n_local:
        print(f"local test photos: {n_local} -> splits/local_test.csv")
    print("splits written to", SPLITS_DIR)


if __name__ == "__main__":
    main()
