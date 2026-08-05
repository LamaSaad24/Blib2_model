# @title
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from utils import (save_model_checkpoint,EarlyStopping)
from src import (ComposedRetrievalLoss)

def train(
    model,
    trainloader,
    valloader,
    epochs,
    lr,
    weight_decay,
    warmup_steps,
    device,
    start_step=0, # قيمة افتراضيه
    resume_state=None,  # جديد: dict فيها optimizer/scheduler/logit_scale/xbm_memory محفوظين سابقاً
    push_to_hub=False,          
    hub_repo_id=None,
):
    model = model.to(device,  dtype=torch.bfloat16)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        , weight_decay=weight_decay
        )"""
    
    لازم تفصلي الـ 
    learning rate 
    بين 
    LoRA و retrieval_head، 
    لأنهم بحالة مختلفة تماماً —
      LoRA
      بيعدّل أوزان مُدرَّبة مسبقاً (يحتاج خطوات صغيرة)，
      لكن retrieval_head
        طبقة جديدة تماماً بأوزان عشوائية (لازم تتعلم من الصفر، فتحتاج خطوات أكبر).
    """
    """
    optimizer = AdamW([
                {"params": [p for n,p in model.named_parameters() if p.requires_grad and "retrieval_head" not in n], "lr": lr},   # LoRA
                {"params": model.retrieval_head.parameters(), "lr": lr*5},   # رأس الاسترجاع — LR أعلى بـ10x
            ], weight_decay=weight_decay)"""

    total_steps = epochs * len(trainloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Loss Function (L = L_LM + w * L_InfoNCE)
    # use_xbm=True بيفعّل الذاكرة اللي بتزوّد عدد الـ negatives بدون batch أكبر
    criterion = ComposedRetrievalLoss(
        omega=1.0,
        use_xbm=True,
        xbm_capacity=65536,   # زي الورقة بالضبط (memory.py تبعك)
        embed_dim=768,        # لازم يطابق embed_dim بالـ config.yaml
        device=device,
    )
    criterion = criterion.to(device)

    # ============================================================
    # استئناف حالة التدريب (optimizer + scheduler + logit_scale + XBM)
    # ============================================================
    if resume_state is not None:
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            print("[Resume] ✅ optimizer state restored")
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
            print("[Resume] ✅ scheduler state restored")
        if "logit_scale" in resume_state:
            criterion.info_nce.logit_scale.data = resume_state["logit_scale"].to(device)
            print("[Resume] ✅ logit_scale (temperature) restored")
        if "xbm_memory" in resume_state:
            criterion.info_nce.load_state_dict_extra(resume_state["xbm_memory"])
            print("[Resume] ✅ XBM memory restored")

    #early_stopping = EarlyStopping(patience=3)
    #torch.autograd.set_detect_anomaly(True) # Added for debugging NaNs

    #save checkpoint settings
    global_step = start_step
    save_every_n_steps = 250  # نحفظ كل 250 خطوة


    for epoch in range(epochs):
        model.train()
        epoch_lm_loss = 0
        epoch_contrastive_loss = 0
        epoch_total_loss = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # نتخطى الدفعات التي تدربنا عليها سابقاً (فقط في الجولة الأولى من الاستئناف)
            if epoch == 0 and batch_idx < start_step:
                continue

            image_ref = batch["ref_images"]
            image_target = batch["trg_images"]
            text = batch["mod_texts"]        # text queries
            target_text = batch["trg_captions"]  # target descriptions

            optimizer.zero_grad(set_to_none=True)# before forward

            outputs = model(
                image_ref=image_ref,
                image_target=image_target,
                modification_text=text,
                target_captions=target_text # لتوليد Text Loss
            )

            query_emb = outputs["query_embedding"]
            target_emb = outputs["target_embedding"]
            lm_loss_from_model = outputs["language_model_loss"]

            # حساب الخسارة الكلية باستخدام ComposedRetrievalLoss
            loss, contrastive_loss, lm_loss = criterion(
                query_emb,
                target_emb,
                lm_loss_from_model
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            # save checkpoint
            global_step += 1 # زيادة العداد بمقدار 1 مع كل دفعة (Batch)

            # ✅ الآن منمرر optimizer/scheduler/criterion عشان يتحفظوا كمان
            if global_step % save_every_n_steps == 0:
                save_model_checkpoint(
                    model, global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    push_to_hub=push_to_hub,
                    hub_repo_id=hub_repo_id
                )

            # Logging
            epoch_total_loss += loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()

            pbar.set_postfix({
                "total loss": f"{loss.item():.4f}",
                "lm loss": f"{lm_loss.item():.4f}",
                "contrastive loss (InfoNCE)": f"{contrastive_loss.item():.4f}"
            })

        print("starting validation......")
        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        val_total_loss = 0
        val_lm_loss = 0
        val_contrastive_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(valloader, desc="Validating")
            for batch in val_pbar:
                image_ref = batch["ref_images"]
                image_target = batch["trg_images"]
                text = batch["mod_texts"]
                target_text = batch["trg_captions"]

                outputs = model(
                    image_ref=image_ref,
                    image_target=image_target,
                    modification_text=text,
                    target_captions=target_text
                )

                query_emb = outputs["query_embedding"]
                target_emb = outputs["target_embedding"]
                lm_loss_from_model = outputs["language_model_loss"]

                loss, contrastive_loss, lm_loss = criterion(
                    query_emb,
                    target_emb,
                    lm_loss_from_model
                )

                val_total_loss += loss.item()
                val_lm_loss += lm_loss.item()
                val_contrastive_loss += contrastive_loss.item()

                val_pbar.set_postfix({
                    "val_loss"  : f"{loss.item():.4f}",
                    "val_lm"    : f"{lm_loss.item():.4f}",
                    "val_InfoNCE": f"{contrastive_loss.item():.4f}"
                })

        val_total_loss /= len(valloader)
        val_lm_loss /= len(valloader)
        val_contrastive_loss /= len(valloader)

        print(f"\n[Validation] Total Loss: {val_total_loss:.4f} | LM Loss: {val_lm_loss:.4f} | InfoNCE: {val_contrastive_loss:.4f}")

        print(
                f"[Epoch {epoch+1}] "
                f"Train Total: {epoch_total_loss/len(trainloader):.4f} | "
                f"LM: {epoch_lm_loss/len(trainloader):.4f} | "
                f"InfoNCE: {epoch_contrastive_loss/len(trainloader):.4f}"
            )
    print("ending of train")

    # نرجع criterion عشان main.py يقدر يحفظ logit_scale النهائي بالـ save_final_model
    return criterion
