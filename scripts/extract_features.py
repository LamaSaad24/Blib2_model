#scripts/preprocess_features.py
"""
تمرير الصور على نموذج clip
extract patch-level features
save in meta.csv
Usage:
    python scripts/extract_features.py --csv data/sample/train.csv \
        --image_root data/raw/images --save_dir models/clip_features/train \
        --device auto --batch_size 8 --image_size 224
"""

import os
import argparse # parsing command-line arguments
from PIL import Image, UnidentifiedImageError # open images 
from tqdm import tqdm  # progress bar
import pandas as pd
import torch # pyTorch, tensors , save and load .pt, GPU/CPU

import sys
# insert the parent directory to the path to import from src 
# os.path.abspath : absolute path
# os.path.join : join paths
# os.path.dirname(__file__)  : current file directory
# __file__ : current file path
# '..' : parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src import CLIPVisionEncoder
from src import extract_clip_patch_features


def extract_and_save(csv_path, image_root, save_dir, device_str='auto', batch_size=8, image_size=224):
    
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    if device_str == 'auto' or device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("Requested CUDA but not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)

    print("Using device:", device)

    # load CLIP (vision encoder) model and processor (resize, normalize, convert PIL to tensor)
    clip = CLIPVisionEncoder().to(device)
    processor = clip.processor  # الـprocessor جاهز داخل الكلاس

    ref_feats = []
    trg_feats = []
    meta = []

    rows = df.to_dict(orient='records') # convert df to list of dicts
    print("df",df)
    print("rows",rows)

    # batching by images (we keep PIL images here and use processor inside flush)
    #نخزن صور مؤقتا ولما يمتلأ الباتش نمررها على النموذج
    batch_ref_imgs = []
    batch_trg_imgs = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_ref_imgs, batch_trg_imgs, batch_meta
        if len(batch_ref_imgs) == 0: # إذا لاتوجد صور لا تعمل شي
            return
        
        # استخراج ميزات CLIP لكل batch
        # batch_ref_imgs list of PLI images 
        # processor resize 224*224 , normalize
        # clip extract patch-level features
        out_refs = extract_clip_patch_features(clip, batch_ref_imgs, device=device)
        out_trgs = extract_clip_patch_features(clip, batch_trg_imgs, device=device)
        """ 
        out_refs.shape = (B, L, C)
        B = batch size
        L = number of patches
        C = embedding dim
        """

        # Move each sample to CPU and append to lists (free GPU memory quickly)
        # نمر على كل صورة في باتش batch
        for i in range(out_refs.shape[0]):
            ref_feats.append(out_refs[i].cpu()) # extract tensor from GPU (.cpu())
            trg_feats.append(out_trgs[i].cpu()) # نخزنه في الرام ونحرر GPU
            meta.append(batch_meta[i]) # save image info (index, path)

        # clear batch lists and ready to next batch
        batch_ref_imgs = []
        batch_trg_imgs = []
        batch_meta = []

    skipped = 0
    for i, row in enumerate(tqdm(rows, desc="Rows")):
        ref_path = os.path.join(image_root, row.get('ref_path', '')) # the path for image
        trg_path = os.path.join(image_root, row.get('trg_path', ''))

        # try open images as PIL and convert to RGB
        try:
            ref_im = Image.open(ref_path).convert("RGB")
            trg_im = Image.open(trg_path).convert("RGB")
        # not found image skipped
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e: 
            skipped += 1
            continue

        batch_ref_imgs.append(ref_im)
        batch_trg_imgs.append(trg_im)
        batch_meta.append({
            'row_idx': i, 
            'ref_path': row.get('ref_path'), 
            'trg_path': row.get('trg_path')
            })

        # flush if batch full
        if len(batch_ref_imgs) >= batch_size:
            flush_batch()

    # flush remaining
    flush_batch()

    # save outputs (قوائم من tensors، كل عنصر شكل (L,C))
    ref_out_path = os.path.join(save_dir, 'ref_feats_list.pt')
    trg_out_path = os.path.join(save_dir, 'trg_feats_list.pt')
    meta_out_path = os.path.join(save_dir, 'meta.csv')

    torch.save([f.cpu() for f in ref_feats], ref_out_path)
    torch.save([f.cpu() for f in trg_feats], trg_out_path)
    pd.DataFrame(meta).to_csv(meta_out_path, index=False)

    print("Saved features and meta in", save_dir)
    print(f"Extracted {len(meta)} pairs. Skipped {skipped} pairs due to missing/corrupt images.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/sample/val.csv")
    p.add_argument("--image_root", default="data/raw/images")
    p.add_argument("--save_dir", default="models/clip_features/val")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p.add_argument("--batch_size", type=int, default=8, help="batch size for images")
    p.add_argument("--image_size", type=int, default=224, help="resize images to this size (square) — handled by processor")
    args = p.parse_args()
    extract_and_save(args.csv, args.image_root, args.save_dir, args.device, args.batch_size, args.image_size)
