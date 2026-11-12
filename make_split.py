"""Hold out a set of PATIENTS from train.csv for checkpoint selection, so the
official validation set is never used to choose a model.

Writes train_split.csv and select_split.csv. No patient appears in both.

Example:
  python make_split.py --train_csv /path/to/CheXpert-v1.0-384/train.csv \
      --out_dir /path/to/CheXpert-v1.0-384 --holdout_patients 2000 --seed 0
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import patient_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdout_patients", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    df = df[df["Path"].notna()].reset_index(drop=True)
    patients = df["Path"].map(patient_id)
    unique = np.array(sorted(patients.unique()))
    if args.holdout_patients >= len(unique):
        raise ValueError("holdout_patients must be smaller than the number of patients")

    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(unique, size=args.holdout_patients, replace=False).tolist())
    is_held = patients.isin(held).to_numpy()

    train_df, select_df = df[~is_held], df[is_held]
    assert not set(train_df["Path"].map(patient_id)) & set(select_df["Path"].map(patient_id))

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.out_dir, "train_split.csv"), index=False)
    select_df.to_csv(os.path.join(args.out_dir, "select_split.csv"), index=False)
    print(f"patients: {len(unique)} total, {len(held)} held out")
    print(f"images:   {len(train_df)} train_split, {len(select_df)} select_split")


if __name__ == "__main__":
    main()
