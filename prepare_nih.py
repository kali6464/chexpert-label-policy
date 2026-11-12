"""Prepare the official NIH ChestX-ray14 TEST split for scoring with the CheXpert models.

  python prepare_nih.py --nih_root <external>/nih --out_root <external> --workers 8

Inputs  : <nih_root>/Data_Entry_2017_v2020.csv (or Data_Entry_2017.csv), <nih_root>/test_list.txt,
          <nih_root>/images/*.png
Outputs : <out_root>/nih_test/<image>.jpg   384 x 384, 8-bit grayscale, as for CheXpert
          <out_root>/nih_test.csv            CheXpert-style CSV (Path + 14 label columns)

Rules (decided before any model is scored on these images):
  * Only images in the official test_list.txt are used (about 25,600 images, no patient
    overlap with the NIH training split; the data come from a different hospital than CheXpert).
  * The NIH labels were text-mined from reports, so this is an external test with
    report-derived labels, not radiologist image labels.
  * Mapping: Atelectasis, Cardiomegaly, Consolidation, Edema -> same name; Effusion -> Pleural
    Effusion. A label is 1 if it appears in "Finding Labels" and 0 otherwise. Every other
    CheXpert column is -1 (excluded).
"""
import argparse
import os
from multiprocessing import Pool

import pandas as pd
from PIL import Image

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
MAP = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
       "Edema": "Edema", "Effusion": "Pleural Effusion"}


def convert(job):
    src, dst, size = job
    if os.path.exists(dst):
        return None
    try:
        Image.open(src).convert("L").resize((size, size), Image.BILINEAR).save(dst + ".tmp", format="JPEG", quality=95)
        os.replace(dst + ".tmp", dst)
        return None
    except Exception as exc:
        return f"{src}\t{exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    csv = next((os.path.join(a.nih_root, n) for n in ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv")
                if os.path.exists(os.path.join(a.nih_root, n))), None)
    if csv is None:
        raise SystemExit("Data_Entry_2017_v2020.csv not found in --nih_root")
    df = pd.read_csv(csv)
    with open(os.path.join(a.nih_root, "test_list.txt")) as f:
        test = {line.strip() for line in f if line.strip()}
    df = df[df["Image Index"].isin(test)].reset_index(drop=True)
    print(f"{len(test)} names in test_list.txt, {len(df)} found in {os.path.basename(csv)}, "
          f"{df['Patient ID'].nunique()} patients")

    img_dir = os.path.join(a.out_root, "nih_test")
    os.makedirs(img_dir, exist_ok=True)
    jobs, absent = [], 0
    for name in df["Image Index"]:
        src = os.path.join(a.nih_root, "images", name)
        if os.path.exists(src):
            jobs.append((src, os.path.join(img_dir, name.replace(".png", ".jpg")), a.size))
        else:
            absent += 1
    print(f"{len(jobs)} images found, {absent} missing from <nih_root>/images")
    with Pool(a.workers) as pool:
        fails = [r for r in pool.imap_unordered(convert, jobs, chunksize=64) if r]
    print(f"converted {len(jobs) - len(fails)}, failed {len(fails)}")

    rows = []
    for _, r in df.iterrows():
        jpg = r["Image Index"].replace(".png", ".jpg")
        if not os.path.exists(os.path.join(img_dir, jpg)):
            continue
        found = set(str(r["Finding Labels"]).split("|"))
        row = {"Path": f"nih_test/{jpg}", "Frontal/Lateral": "Frontal", "View": r.get("View Position", "")}
        for name in ALL_LABELS:
            row[name] = -1.0
        for nih, chex in MAP.items():
            row[chex] = 1.0 if nih in found else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_root, "nih_test.csv"), index=False)
    print(f"wrote {len(out)} rows to {os.path.join(a.out_root, 'nih_test.csv')}")
    print("positives per label:")
    for chex in MAP.values():
        print(f"  {chex:18s} {int((out[chex] == 1).sum()):6d}")
    if "View" in out:
        print("views:", out["View"].value_counts().to_dict())


if __name__ == "__main__":
    main()
