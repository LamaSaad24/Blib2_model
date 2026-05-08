import os
import pandas as pd

def load_available_images(images_root):
    available = set()
    for root, dirs, files in os.walk(images_root):
        for f in files:
            # نخزن المسار النسبي فقط
            rel = os.path.relpath(os.path.join(root, f), images_root)
            available.add(rel.replace("\\", "/"))
    return available


def find_missing_images(csv_path, available_images, chunksize=20000):
    missing = set()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for col in ['ref_path', 'trg_path']:
            for p in chunk[col]:
                if p not in available_images:
                    missing.add(p)
    return missing


if __name__ == "__main__":
    base = "data/processed"
    images_root = "data/raw/images"

    print("🚀 Loading image list from disk... (this happens once)")
    available_images = load_available_images(images_root)
    print(f"📦 Found {len(available_images):,} images in dataset\n")

    files = ["train_triplets.csv"]
    
    total_missing = 0

    for f in files:
        csv_path = os.path.join(base, f)
        if not os.path.exists(csv_path):
            print(f"⚠️ {f} not found, skip.")
            continue

        print(f"🔍 Checking: {f}")
        missing = find_missing_images(csv_path, available_images)
        count_missing = len(missing)
        total_missing += count_missing

        if count_missing:
            out_file = csv_path.replace(".csv", "_missing.txt")
            with open(out_file, "w") as out:
                out.write("\n".join(missing))
            print(f"❌ {count_missing} missing → saved to {out_file}\n")
        else:
            print(f"✅ {f}: all images found.\n")
    
    print(f"\n✨ Total missing images: {total_missing}")
