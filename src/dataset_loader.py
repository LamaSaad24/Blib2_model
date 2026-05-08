# @title
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, images_dir, csv_path):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)

    def _load_image(self, filename):
        path = os.path.join(self.images_dir, filename)
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Fashion200KComposedDataset  error {path}: {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ref_img = self._load_image(row["ref_path"])
        trg_img = self._load_image(row["trg_path"])

        mod_text = str(row["mod_text"])
        trg_caption = str(row["trg_caption"])

        return {
            "ref_image": ref_img,
            "mod_text": mod_text,
            "trg_image": trg_img,
            "trg_caption": trg_caption
        }


def collate_fn(batch):
    ref_imgs = [item["ref_image"] for item in batch]
    mod_texts = [item["mod_text"] for item in batch]
    trg_imgs = [item["trg_image"] for item in batch]
    trg_captions = [item["trg_caption"] for item in batch]

    return {
        "ref_images": ref_imgs,              # (B, 3, H, W)
        "mod_texts": mod_texts,     # list of str
        "trg_images": trg_imgs,          # (B, 3, H, W)
        "trg_captions": trg_captions # list of str
    }