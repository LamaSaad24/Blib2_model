# Fashion200K Data Preparation

1- python scripts/download_images.py

2- python scripts/preprocess_fashion200k.py

3- python scripts/check_missing_images.py 

4-python scripts/redownload_missing.py 

5- python scripts/create_small_triplet_subset.py 

------------------------
1- venv\Scripts\activate.ps1
---------------
استخراج ميزات CLIP (على العينة الصغرى)

5- python scripts/extract_features.py --csv data/sample/test.csv --image_root data/raw/images --save_dir outputs/clip_features/test --device auto --batch_size 14

الناتج:


models/clip_features/train/ref_feats_list.pt (قائمة من tensors كل واحد شكل (L, C)).

models/clip_features/train/trg_feats_list.pt

models/clip_features/train/meta.csv (ربط صفوف CSV بالصور).

ملاحظة: هذه الخطوة قد تأخذ وقتًا حسب عدد الصور. على 80 صورة صغيرة ستنتهي بسرعة.
----------------
تنزيله محليا
huggingface-cli download google/flan-t5-large

6- python scripts/train.py
6- python scripts/main.py

7- python scripts/eval_retrieval.py
>>> Calculating Recall@K...

===== Evaluation Results =====
R@1: 0.2000
R@10: 1.0000
R@50: 1.0000