import pandas as pd
import os

# مسارات الملفات وحجم العينة
files_info = {
    "train": ("data/processed/train_triplets.csv", 100),
    "val": ("data/processed/val_triplets.csv", 50),
    "test": ("data/processed/test_triplets.csv", 50)
}

output_dir = "data/sample/"
image_dir = "data/raw/images"
os.makedirs(output_dir, exist_ok=True)

for split, (file_path, sample_size) in files_info.items():
    # قراءة CSV
    df = pd.read_csv(file_path)

    # أخذ عينة أكبر قليلًا لضمان وجود الصور
    extra_factor = 3  # خذ 3x من العدد المطلوب لتجنب نقص الصور
    initial_sample_size = min(len(df), sample_size * extra_factor)
    sample_df = df.sample(initial_sample_size, random_state=42)

    # تصفية الصفوف بحيث الصور موجودة
    def images_exist(row):
        return os.path.exists(os.path.join(image_dir, row['ref_path'])) and \
               os.path.exists(os.path.join(image_dir, row['trg_path']))

    sample_df = sample_df[sample_df.apply(images_exist, axis=1)]

    # خذ فقط العدد المطلوب
    final_sample = sample_df.head(sample_size)

    # حفظ العينة
    output_file = os.path.join(output_dir, f"{split}.csv")
    final_sample.to_csv(output_file, index=False)
    print(f"{split} sample saved: {output_file} ({len(final_sample)} rows)")
