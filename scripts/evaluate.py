import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model(model, test_dataloader, device="cuda"):
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
    
    print(f"\nComputing Similarities and Recall@K for {num_queries} queries...")
    
    recall_at_1 = 0
    recall_at_10 = 0
    recall_at_50 = 0

    for i in tqdm(range(num_queries), desc="Ranking"):
        q = all_query_embs[i].unsqueeze(0) 
        sims = torch.matmul(q, all_target_embs.T).squeeze(0) 
        
        top50_indices = sims.topk(50, largest=True).indices
        correct_target_index = i
        
        if correct_target_index == top50_indices[0]:
            recall_at_1 += 1
            
        if correct_target_index in top50_indices[:10]:
            recall_at_10 += 1
            
        if correct_target_index in top50_indices:
            recall_at_50 += 1

    r1 = (recall_at_1 / num_queries) * 100
    r10 = (recall_at_10 / num_queries) * 100
    r50 = (recall_at_50 / num_queries) * 100
    r_avg = (r10 + r50) / 2 

    print("\n" + "="*35)
    print(" 🎯 EVALUATION RESULTS 🎯 ")
    print("="*35)
    print(f"Recall@1:   {r1:.2f}%")
    print(f"Recall@10:  {r10:.2f}%")
    print(f"Recall@50:  {r50:.2f}%")
    print(f"Average(10,50): {r_avg:.2f}%")
    print("="*35)

    return r1, r10, r50, r_avg