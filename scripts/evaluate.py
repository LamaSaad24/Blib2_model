import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model(model, test_dataloader, device="cuda", recall_ks=(1, 10, 50)):
    model.eval()
    model.to(device, dtype=torch.bfloat16)

    all_query_embs = []
    all_target_embs = []

    print("Extracting features for Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            image_ref = batch["ref_images"]
            image_target = batch["trg_images"]
            text = batch["mod_texts"]

            # نص فارغ للهدف
            empty_texts = [""] * len(text)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # 🌟 السر هنا: استخدمنا get_embedding مباشرة لنتجاوز دالة forward تماماً
                query_emb = model.get_embedding(images=image_ref, texts=text)
                target_emb = model.get_embedding(images=image_target, texts=empty_texts)

            # توحيد المتجهات
            query_emb = F.normalize(query_emb, p=2, dim=-1)
            target_emb = F.normalize(target_emb, p=2, dim=-1)

            # تخزين المتجهات
            all_query_embs.append(query_emb.cpu().float())
            all_target_embs.append(target_emb.cpu().float())

    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_target_embs = torch.cat(all_target_embs, dim=0)

    num_queries = all_query_embs.shape[0]

    # ============================================================
    # ✅ سياج أمان: لو أعلى K أكبر من عدد الاستعلامات الفعلي، بنحذّر
    # (هاد بالضبط سبب مشكلة Recall@50=100% الوهمية يلي واجهناها سابقاً)
    # ============================================================
    max_k = max(recall_ks)
    if num_queries <= max_k:
        print(
            f"⚠️ Warning: Number of queries ({num_queries}) is less than or equal to "
            f" highest K ({max_k})! Recall@{max_k} will misleadingly appear close to 100% "
            f"regardless of the model's actual performance. Consider using a larger evaluation set."
        )

    print(f"\nComputing Similarities and Recall@K for {num_queries} queries...")

    recall_counts = {k: 0 for k in recall_ks}
    max_k_capped = min(max_k, num_queries)  # ما نطلب topk أكبر من العدد الموجود فعلياً

    for i in tqdm(range(num_queries), desc="Ranking"):
        q = all_query_embs[i].unsqueeze(0)
        sims = torch.matmul(q, all_target_embs.T).squeeze(0)

        top_indices = sims.topk(max_k_capped, largest=True).indices
        correct_target_index = i

        for k in recall_ks:
            k_capped = min(k, max_k_capped)
            if correct_target_index in top_indices[:k_capped]:
                recall_counts[k] += 1

    results = {k: (recall_counts[k] / num_queries) * 100 for k in recall_ks}

    print("\n" + "="*35)
    print(" 🎯 EVALUATION RESULTS 🎯 ")
    print("="*35)
    for k in recall_ks:
        print(f"Recall@{k}:   {results[k]:.2f}%")

    sorted_ks = sorted(recall_ks)
    if len(sorted_ks) >= 2:
        avg = (results[sorted_ks[-2]] + results[sorted_ks[-1]]) / 2
        print(f"Average({sorted_ks[-2]},{sorted_ks[-1]}): {avg:.2f}%")
    print("="*35)

    # نرجع نفس الشكل السابق (r1, r10, r50, r_avg) لو موجودين، للتوافق مع main_v2.py
    r1 = results.get(1, 0.0)
    r10 = results.get(10, 0.0)
    r50 = results.get(50, 0.0)
    r_avg = (r10 + r50) / 2

    return r1, r10, r50, r_avg