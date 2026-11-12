import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import ALL_LABELS, resolve_labels, strip_prefix, study_id

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CheXpertDataset(Dataset):
    """Returns (image, target[14], uncertain_mask[14]).

    uncertain_mask marks entries that were -1 in the CSV. Training ignores it
    for now; it is what a learned uncertain-label method will consume.
    With keep_uncertain=True the targets keep -1 so evaluation can drop them.
    """

    def __init__(self, csv_file, img_dir, transform, policy="zeros",
                 keep_uncertain=False, views="all"):
        df = pd.read_csv(csv_file)
        df = df[df["Path"].notna()]
        if views == "frontal":
            df = df[df["Frontal/Lateral"] == "Frontal"]
        elif views != "all":
            raise ValueError(f"views must be 'all' or 'frontal', got {views!r}")
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.paths = [strip_prefix(p) for p in self.df["Path"]]
        self.study_ids = np.array([study_id(p) for p in self.paths])
        raw = self.df[ALL_LABELS].to_numpy(dtype=np.float32)
        self.targets, self.uncertain = resolve_labels(raw, policy, keep_uncertain)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.paths[idx])).convert("RGB")
        image = self.transform(image)
        return (image,
                torch.from_numpy(self.targets[idx]),
                torch.from_numpy(self.uncertain[idx]))


def get_transforms(img_size, train, hflip=False):
    ops = [transforms.Resize((img_size, img_size))]  # no-op on pre-resized images
    if train:
        ops.append(transforms.RandomRotation(10))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    ops += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


def get_dataloaders(cfg):
    """train: policy applied.  select: held-out training patients, uncertain
    entries kept as -1 and excluded from AUC.  official_val: radiologist labels."""
    size, workers = int(cfg["img_size"]), int(cfg["num_workers"])
    bs, views = int(cfg["batch_size"]), cfg.get("views", "all")
    hflip = bool(cfg.get("hflip", False))

    train_ds = CheXpertDataset(cfg["train_csv"], cfg["img_dir"], get_transforms(size, True, hflip),
                               policy=cfg["uncertainty_policy"], views=views)
    select_ds = CheXpertDataset(cfg["select_csv"], cfg["img_dir"], get_transforms(size, False),
                                keep_uncertain=True, views=views)
    val_ds = CheXpertDataset(cfg["official_val_csv"], cfg["img_dir"], get_transforms(size, False),
                             keep_uncertain=True, views=views)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    select_loader = DataLoader(select_ds, batch_size=bs * 2, shuffle=False, **common)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, **common)
    return train_loader, select_loader, val_loader
