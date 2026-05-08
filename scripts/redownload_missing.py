import os
import requests
from tqdm import tqdm

# الملفات المطلوبة
missing_file = "data/processed/train_triplets_missing.txt"
urls_file = "data/raw/image_urls.txt"
save_root = "data/raw/images"

# تحميل جميع الروابط في dict
print("🔍 Loading URLs file...")
urls = {}
with open(urls_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            urls[parts[0]] = parts[1]

# قراءة الملفات الناقصة
with open(missing_file, "r", encoding="utf-8") as f:
    missing_paths = [line.strip().split("\t")[0] for line in f if line.strip()]

# إعادة التنزيل
print(f"📦 Redownloading {len(missing_paths)} missing images...")

for path in tqdm(missing_paths):
    if path not in urls:
        continue
    url = urls[path]
    save_path = os.path.join(save_root, path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"⚠️ Failed: {url}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

print("✅ Done redownloading missing images.")
