"""Resize every image listed in the CheXpert CSVs once and write it to a new
root with the same folder layout. Safe to re-run: finished files are skipped.

Example:
  python preprocess_resize.py \
      --src_root /path/to/CheXpert-v1.0 \
      --dst_root /path/to/CheXpert-v1.0-384 \
      --size 384 --workers 8
"""
import argparse
import os
import shutil
from multiprocessing import Pool

import pandas as pd
from PIL import Image

from common import strip_prefix


def _resize_one(job):
    src, dst, size = job
    if os.path.exists(dst):
        return None
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
        tmp = dst + ".tmp"
        img.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, dst)  # atomic, so an interrupted job leaves no half-written file
        return None
    except Exception as exc:  # report and continue
        return f"{src}\t{exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--csvs", nargs="+", default=["train.csv", "valid.csv"])
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    jobs = []
    for name in args.csvs:
        csv_path = os.path.join(args.src_root, name)
        df = pd.read_csv(csv_path)
        df = df[df["Path"].notna()]
        for p in df["Path"]:
            rel = strip_prefix(p)
            jobs.append((os.path.join(args.src_root, rel), os.path.join(args.dst_root, rel), args.size))
        shutil.copy2(csv_path, os.path.join(args.dst_root, name))
    print(f"{len(jobs)} images to check", flush=True)

    failures = []
    with Pool(args.workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=256), 1):
            if err:
                failures.append(err)
            if i % 10000 == 0:
                print(f"  {i}/{len(jobs)} done, {len(failures)} failed", flush=True)

    print(f"Finished. {len(jobs) - len(failures)} ok, {len(failures)} failed.")
    if failures:
        fail_path = os.path.join(args.dst_root, "resize_failures.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failures))
        print(f"Failure list written to {fail_path}")


if __name__ == "__main__":
    main()
