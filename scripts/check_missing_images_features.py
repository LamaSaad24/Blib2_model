import pandas as pd

# === عدّلي المسارات حسب جهازك ===
train_csv = "data/sample/train.csv"
meta_csv  = "models/clip_features/train/meta.csv"

# ================================

train = pd.read_csv(train_csv)
meta  = pd.read_csv(meta_csv)

# نستخدم ref_path + trg_path كهوية لكل صف
train_pairs = set(zip(train["ref_path"], train["trg_path"]))
meta_pairs  = set(zip(meta["ref_path"], meta["trg_path"]))

# الصفوف الموجودة في train ولم تظهر في meta (سببها الصور الناقصة)
missing_rows = train_pairs - meta_pairs

print("\n===== rows missing numbers =====")
print(len(missing_rows))

if len(missing_rows) == 0:
    print("no rows missing ✓")
else:
    print("\n===== السطور الناقصة نفسها (أول 20) =====")
    for ref, trg in list(missing_rows)[:20]:
        print(f"ref_path: {ref}   |   trg_path: {trg}")

    # إذا بدك نطبعها كلها
    print("\n===== تفاصيل السطور الناقصة بالكامل =====")
    for ref, trg in missing_rows:
        row = train[(train["ref_path"] == ref) & (train["trg_path"] == trg)]
        print(row.to_string(index=False))
