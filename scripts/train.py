# @title
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from utils import (save_model_checkpoint,EarlyStopping)
from src import (ComposedRetrievalLoss)


def build_scheduler(scheduler_name, optimizer, warmup_steps, total_steps):
    """
    ✅ جديد: بتقرأ scheduler_name فعلياً من config.yaml بدل ما تكون
    القيمة "زينة" غير مستخدمة. بتدعم القيمتين الشائعتين:
    - "cosine_with_warmup": نزول تدريجي بمنحنى جيبي (أنعم من الخطي، شائع بموديلات كبيرة)
    - "linear_with_warmup" (أو أي قيمة تانية): نزول خطي (كان الافتراضي الثابت سابقاً)
    """
    scheduler_name = (scheduler_name or "linear_with_warmup").lower()

    if scheduler_name == "cosine_with_warmup":
        print(f"[Scheduler] ✅ استخدام cosine_with_warmup (warmup={warmup_steps}, total={total_steps})")
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        print(f"[Scheduler] ✅ استخدام linear_with_warmup (warmup={warmup_steps}, total={total_steps})")
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )


def train(
    model,
    trainloader,
    valloader,
    epochs,
    lr,
    weight_decay,
    warmup_steps,
    device,
    start_step=0,
    resume_state=None,
    push_to_hub=False,
    hub_repo_id=None,
    scheduler_name="linear_with_warmup",
):
    model = model.to(device,  dtype=torch.bfloat16)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )

    steps_per_epoch = len(trainloader)
    total_steps = epochs * steps_per_epoch

    scheduler = build_scheduler(scheduler_name, optimizer, warmup_steps, total_steps)

    criterion = ComposedRetrievalLoss(
        omega=1.0, use_xbm=True, xbm_capacity=65536, embed_dim=768, device=device,
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

    # ============================================================
    # ✅ إصلاح جوهري: حساب صحيح لـ "أي epoch" و"أي batch داخلها"
    # بدل الاعتماد على شرط epoch==0 (كان يفشل لو start_step > حجم epoch وحدة)
    # ============================================================
    start_epoch = start_step // steps_per_epoch
    start_batch_in_epoch = start_step % steps_per_epoch

    if start_step > 0:
        print(f"[Resume] 📍 هيبلش من epoch {start_epoch + 1}, batch {start_batch_in_epoch} داخلها "
              f"(global_step={start_step})")

    global_step = start_step
    save_every_n_steps = 250

    for epoch in range(start_epoch, epochs):
        # ✅ لو هاي أول epoch نبلش فيها (ممكن تكون فيها تخطي جزئي)، وإلا كاملة عادية
        skip_until = start_batch_in_epoch if epoch == start_epoch else 0

        model.train()
        epoch_lm_loss = 0
        epoch_contrastive_loss = 0
        epoch_total_loss = 0
        counted_batches = 0  # ✅ لعد الـ batches الفعلية (بدون المتخطاة) لحساب المتوسط صح
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx < skip_until:
                continue

            image_ref = batch["ref_images"]
            image_target = batch["trg_images"]
            text = batch["mod_texts"]
            target_text = batch["trg_captions"]

            optimizer.zero_grad(set_to_none=True)

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
                query_emb, target_emb, lm_loss_from_model
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            counted_batches += 1

            if global_step % save_every_n_steps == 0:
                save_model_checkpoint(
                    model, global_step,
                    optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                    push_to_hub=push_to_hub, hub_repo_id=hub_repo_id
                )

            epoch_total_loss += loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.2f}",
                "lm": f"{lm_loss.item():.2f}",
                "nce": f"{contrastive_loss.item():.2f}"
            })

        # ============================================================
        # ✅ لو هاي الـ epoch كانت متخطاة بالكامل (مافي أي batch فعلي اتدرب)،
        # منتخطى الـ validation كمان -- ما في داعي نعيد تقييم موديل ما تغيّر
        # ============================================================
        if counted_batches == 0:
            print(f"[Epoch {epoch+1}] كانت متخطاة بالكامل (استئناف) -- تخطينا الـ validation كمان.")
            continue

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
                    query_emb, target_emb, lm_loss_from_model
                )

                val_total_loss += loss.item()
                val_lm_loss += lm_loss.item()
                val_contrastive_loss += contrastive_loss.item()

                val_pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_lm": f"{lm_loss.item():.4f}",
                    "val_InfoNCE": f"{contrastive_loss.item():.4f}"
                })

        val_total_loss /= len(valloader)
        val_lm_loss /= len(valloader)
        val_contrastive_loss /= len(valloader)

        print(f"\n[Validation] Total Loss: {val_total_loss:.4f} | LM Loss: {val_lm_loss:.4f} | InfoNCE: {val_contrastive_loss:.4f}")
        print(
            f"[Epoch {epoch+1}] "
            f"Train Total: {epoch_total_loss/counted_batches:.4f} | "
            f"LM: {epoch_lm_loss/counted_batches:.4f} | "
            f"InfoNCE: {epoch_contrastive_loss/counted_batches:.4f}"
        )
    print("ending of train")

    return criterion
