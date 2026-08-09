# @title
import os
import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, images_dir, csv_path, max_retries=5):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.max_retries = max_retries
        self._warned_paths = set()

    def _load_image(self, filename):
        path = os.path.join(self.images_dir, filename)
        img = Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ✅ لو صورة مفقودة، نتخطاها ونجرب مثال عشوائي تاني بدل ما ننهار
        for attempt in range(self.max_retries):
            row = self.df.iloc[idx]
            try:
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
            except (FileNotFoundError, OSError) as e:
                bad_path = str(row.get("ref_path", "?")) + " / " + str(row.get("trg_path", "?"))
                if bad_path not in self._warned_paths:
                    print(f"[FashionDataset] ⚠️ صورة مفقودة، تخطّي: {bad_path} ({e})")
                    self._warned_paths.add(bad_path)
                idx = random.randint(0, len(self.df) - 1)

        raise RuntimeError(
            f"[FashionDataset] فشلنا نلاقي صورة صالحة بعد {self.max_retries} محاولات. "
            f"تأكد من image_dir صح ومن سلامة الداتاست."
        )


def collate_fn(batch):
    ref_imgs = [item["ref_image"] for item in batch]
    mod_texts = [item["mod_text"] for item in batch]
    trg_imgs = [item["trg_image"] for item in batch]
    trg_captions = [item["trg_caption"] for item in batch]

    return {
        "ref_images": ref_imgs,
        "mod_texts": mod_texts,
        "trg_images": trg_imgs,
        "trg_captions": trg_captions
    }