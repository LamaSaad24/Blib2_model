import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
import yaml
from src import FullModel

# =====================
# تحميل الإعدادات
# =====================
cfg = yaml.safe_load(open("config.yaml", "r"))

# =====================
# تحميل BLIP-2
# =====================
print("⏳ Loading BLIP-2...")
processor = Blip2Processor.from_pretrained(cfg['blip2_model_id'])
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    cfg['blip2_model_id'],
    torch_dtype=torch.bfloat16,
)
print("✅ BLIP-2 Ready")

# =====================
# بناء النموذج
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FullModel(processor, blip2, cfg['embed_dim'])

# =====================
# استرجاع آخر checkpoint
# =====================
latest_step = 5000  # آخر checkpoint عندك
checkpoint_path = f"outputs/Blip2_model_Checkpoints_small/step_{latest_step}"

print(f"⏳    Restore model from step {latest_step}...")
model.blip2.language_model = PeftModel.from_pretrained(
    model.blip2.language_model,
    os.path.join(checkpoint_path, "lora")
)
model.retrieval_head.load_state_dict(
    torch.load(
        os.path.join(checkpoint_path, "retrieval_head.pth"),
        weights_only=True
    )
)
model = model.to(device, dtype=torch.bfloat16)
model.eval()
print("✅ Restored model successfully ")

#دالة البحث
import torch.nn.functional as F
import pandas as pd
import numpy as np

def get_query_embedding(model, processor, image, text, device):
    """
    يحول صورة + نص لـ embedding
    """
    with torch.no_grad():
        embedding = model.get_embedding(
            images=[image],
            texts=[text]
        )
    return embedding

def get_target_embeddings(model, processor, image_paths, device, batch_size=8):
    """
    يحسب embeddings لكل الصور في الـ dataset
    """
    all_embeddings = []
    all_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                all_paths.append(path)
            except:
                continue

        if not batch_images:
            continue

        with torch.no_grad():
            emb = model.get_embedding(
                images=batch_images,
                texts=[""] * len(batch_images)
            )
        all_embeddings.append(emb.cpu().float())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_paths


def search(query_emb, target_embs, target_paths, top_k=5):
    """
    يبحث عن أقرب صور للـ query
    """
    # حساب التشابه
    similarities = F.cosine_similarity(
        query_emb.cpu().float(),
        target_embs,
        dim=-1
    )

    # أعلى نتائج
    top_scores, top_indices = similarities.topk(top_k)

    results = []
    for score, idx in zip(top_scores, top_indices):
        results.append({
            "path": target_paths[idx],
            "score": score.item()
        })
    return results

    #تجربة البحث
    # =====================
# تحميل بيانات الاختبار
# =====================
test_df = pd.read_csv("data/processed/test_triplets.csv")
IMAGE_DIR = cfg['image_dir']

# نأخذ sample صغير للاختبار السريع
sample = test_df.sample(n=200, random_state=42)

# =====================
# نجهز الصور المستهدفة
# =====================
print("⏳ calculate embeddings for images...")
target_paths = [
    os.path.join(IMAGE_DIR, p) 
    for p in sample["trg_path"].tolist()
]
target_embs, valid_paths = get_target_embeddings(
    model, processor, target_paths, device
)
print(f"✅ {len(valid_paths)}  image ready")

# =====================
# اختبار مثال واحد
# =====================
row = test_df.iloc[0]

ref_image = Image.open(
    os.path.join(IMAGE_DIR, row["ref_path"])
).convert("RGB")
mod_text   = row["mod_text"]
trg_path   = os.path.join(IMAGE_DIR, row["trg_path"])

print(f"\n🔍 Refrence image : {row['ref_path']}")
print(f"📝  text modification: {mod_text}")
print(f"🎯  Target Image: {row['trg_path']}")

# البحث
query_emb = get_query_embedding(model, processor, ref_image, mod_text, device)
results   = search(query_emb, target_embs, valid_paths, top_k=5)

# =====================
# عرض النتائج
# =====================
fig, axes = plt.subplots(1, 7, figsize=(24, 4))

# الصورة المرجعية
axes[0].imshow(ref_image)
axes[0].set_title(f"refrence\n{mod_text[:20]}", fontsize=8)
axes[0].axis("off")

# الصورة الهدف
axes[1].imshow(Image.open(trg_path))
axes[1].set_title(" true target ✅", fontsize=8, color="green")
axes[1].axis("off")

# النتائج
for i, result in enumerate(results):
    try:
        img = Image.open(result["path"])
        is_correct = result["path"] == trg_path
        color = "green" if is_correct else "black"
        axes[i+2].imshow(img)
        axes[i+2].set_title(
            f"#{i+1} {'✅' if is_correct else ''}\n{result['score']:.3f}",
            fontsize=8,
            color=color
        )
        axes[i+2].axis("off")
    except:
        axes[i+2].axis("off")

plt.suptitle(f" Text modification: '{mod_text}'", fontsize=11)
plt.tight_layout()
plt.show()


#حساب recall
def evaluate_recall(model, processor, test_df, image_dir, device, top_k=[1, 10, 50]):
    """
    يحسب Recall@K على بيانات الاختبار
    """
    hits = {k: 0 for k in top_k}
    total = 0

    # نأخذ sample للتجربة السريعة
    sample_df = test_df.sample(n=500, random_state=42)

    for _, row in sample_df.iterrows():
        try:
            ref_image = Image.open(
                os.path.join(image_dir, row["ref_path"])
            ).convert("RGB")
            mod_text = row["mod_text"]
            trg_path = os.path.join(image_dir, row["trg_path"])

            # query embedding
            query_emb = get_query_embedding(
                model, processor, ref_image, mod_text, device
            )

            # نبحث
            results = search(query_emb, target_embs, valid_paths, top_k=max(top_k))
            result_paths = [r["path"] for r in results]

            # نتحقق
            for k in top_k:
                if trg_path in result_paths[:k]:
                    hits[k] += 1

            total += 1
        except:
            continue

    print("\n📊  Evaluation Results:")
    for k in top_k:
        recall = hits[k] / total * 100
        print(f"  Recall@{k:2d} = {recall:.2f}%")

    return {k: hits[k]/total*100 for k in top_k}

# شغيلي التقييم
results = evaluate_recall(
    model, processor, test_df, IMAGE_DIR, device
)