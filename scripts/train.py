# @title
import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from utils import (save_model_checkpoint, EarlyStopping)
from src import (ComposedRetrievalLoss)

LOG_CSV_PATH = "outputs/training_log.csv"
LOG_CSV_COLUMNS = [
    "timestamp", "global_step", "epoch_label",
    "val_total_loss", "val_lm_loss", "val_infonce",
    "train_total_loss", "train_lm_loss", "train_infonce",
]


def _download_existing_log(hub_repo_id):
    """✅ جديد: لو فيه سجل قديم على HF Hub، نحمّله أول شي عشان نكمل عليه (مش نبدأ من الصفر)."""
    if hub_repo_id is None:
        return
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=hub_repo_id, filename="training_log.csv", repo_type="model")
        os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
        import shutil
        shutil.copy(downloaded, LOG_CSV_PATH)
        print(f"[Log] ✅ استرجعنا سجل نتائج سابق من HF Hub ({sum(1 for _ in open(LOG_CSV_PATH)) - 1} صف)")
    except Exception:
        print("[Log] مافي سجل سابق على HF Hub -- رح نبدأ سجل جديد")


def _append_log_row(row_dict, hub_repo_id, push_to_hub):
    """✅ جديد: يضيف صف جديد لملف CSV محلي، وبيرفعه لـ HF Hub (نسخة دائمة عبر الجلسات)."""
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_CSV_PATH)

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

    print(f"[Log] ✅ نتيجة اتسجلت بـ {LOG_CSV_PATH}")

    if push_to_hub and hub_repo_id is not None:
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=LOG_CSV_PATH,
                path_in_repo="training_log.csv",
                repo_id=hub_repo_id,
                repo_type="model",
            )
            print(f"[Log] ☁️ السجل اترفع لـ HF Hub: {hub_repo_id}/training_log.csv")
        except Exception as e:
            print(f"[Log] ⚠️ فشل رفع السجل (رح يضل محفوظ محلياً بس): {e}")


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


def _run_validation(model, valloader, criterion, desc="Validating"):
    """دالة مشتركة للـ validation (نستخدمها بالحلقة العادية وبحالة الاسترجاع)."""
    model.eval()
    val_total_loss = 0
    val_lm_loss = 0
    val_contrastive_loss = 0

    with torch.no_grad():
        val_pbar = tqdm(valloader, desc=desc)
        for batch in val_pbar:
            outputs = model(
                image_ref=batch["ref_images"],
                image_target=batch["trg_images"],
                modification_text=batch["mod_texts"],
                target_captions=batch["trg_captions"]
            )
            loss, contrastive_loss, lm_loss = criterion(
                outputs["query_embedding"], outputs["target_embedding"],
                outputs["language_model_loss"]
            )
            val_total_loss += loss.item()
            val_lm_loss += lm_loss.item()
            val_contrastive_loss += contrastive_loss.item()
            val_pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "val_lm": f"{lm_loss.item():.4f}",
                "val_InfoNCE": f"{contrastive_loss.item():.4f}"
            })

    n = len(valloader)
    model.train()
    return val_total_loss / n, val_lm_loss / n, val_contrastive_loss / n


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
    model = model.to(device, dtype=torch.bfloat16)

    # ✅ جديد: نحمّل سجل النتائج القديم (لو موجود) قبل ما نبلش
    _download_existing_log(hub_repo_id)

    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters()
                     if p.requires_grad and "retrieval_head" not in n], "lr": lr},
        {"params": model.retrieval_head.parameters(), "lr": lr * 5},
    ], weight_decay=weight_decay)

    steps_per_epoch = len(trainloader)
    total_steps = epochs * steps_per_epoch

    scheduler = build_scheduler(scheduler_name, optimizer, warmup_steps, total_steps)

    criterion = ComposedRetrievalLoss(
        omega=1.0, use_xbm=True, xbm_capacity=4096, embed_dim=768, device=device,
        xbm_warmup_steps=1000,
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
            print(f"[DEBUG] xbm_warmup_steps = {criterion.info_nce.xbm_warmup_steps}")
            print(f"[DEBUG] _step_count (بعد الاسترجاع) = {criterion.info_nce._step_count}")
            print(f"[DEBUG] mem_target size = {criterion.info_nce.memory_target.size}")
            print(f"[DEBUG] logit_scale (بعد exp) = {criterion.info_nce.logit_scale.exp().item():.4f}")

    if start_step > 0:
        criterion.info_nce._step_count = start_step
        print(f"[Fix] ✅ _step_count = {start_step} (مزامنة مع start_step)")

    start_epoch = start_step // steps_per_epoch
    start_batch_in_epoch = start_step % steps_per_epoch

    if start_step > 0:
        print(f"[Resume] 📍 هيبلش من epoch {start_epoch + 1}, batch {start_batch_in_epoch} داخلها "
              f"(global_step={start_step})")

    # ✅ استرجاع validation ضائعة (لو الاستئناف صار بالضبط عند حد epoch)
    if start_step > 0 and start_batch_in_epoch == 0:
        missed_epoch_num = start_epoch
        print(f"[Recovery] ⚠️ يبدو إنه validation تبع Epoch {missed_epoch_num} ضاعت -- عم نعملها هلق...")

        v_total, v_lm, v_nce = _run_validation(
            model, valloader, criterion, desc=f"Validating (recovered Epoch {missed_epoch_num})"
        )
        print(f"\n[Validation - Recovered Epoch {missed_epoch_num}] "
              f"Total Loss: {v_total:.4f} | LM Loss: {v_lm:.4f} | InfoNCE: {v_nce:.4f}")

        # ✅ تسجيل بالملف
        _append_log_row({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "global_step": start_step,
            "epoch_label": f"Epoch {missed_epoch_num} (recovered)",
            "val_total_loss": f"{v_total:.4f}",
            "val_lm_loss": f"{v_lm:.4f}",
            "val_infonce": f"{v_nce:.4f}",
            "train_total_loss": "", "train_lm_loss": "", "train_infonce": "",
        }, hub_repo_id, push_to_hub)

    global_step = start_step
    save_every_n_steps = 250

    for epoch in range(start_epoch, epochs):
        # ✅ لو هاي أول epoch نبلش فيها (ممكن تكون فيها تخطي جزئي)، وإلا كاملة عادية
        skip_until = start_batch_in_epoch if epoch == start_epoch else 0

        model.train()
        epoch_lm_loss = 0
        epoch_contrastive_loss = 0
        epoch_total_loss = 0
        counted_batches = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx < skip_until:
                continue

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                image_ref=batch["ref_images"],
                image_target=batch["trg_images"],
                modification_text=batch["mod_texts"],
                target_captions=batch["trg_captions"]
            )

            loss, contrastive_loss, lm_loss = criterion(
                outputs["query_embedding"], outputs["target_embedding"],
                outputs["language_model_loss"]
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
        v_total, v_lm, v_nce = _run_validation(model, valloader, criterion, desc="Validating")

        train_total = epoch_total_loss / counted_batches
        train_lm = epoch_lm_loss / counted_batches
        train_nce = epoch_contrastive_loss / counted_batches

        print(f"\n[Validation] Total Loss: {v_total:.4f} | LM Loss: {v_lm:.4f} | InfoNCE: {v_nce:.4f}")
        print(f"[Epoch {epoch+1}] Train Total: {train_total:.4f} | LM: {train_lm:.4f} | InfoNCE: {train_nce:.4f}")

        # ✅ تسجيل بالملف بعد كل epoch كاملة
        _append_log_row({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "global_step": global_step,
            "epoch_label": f"Epoch {epoch+1}",
            "val_total_loss": f"{v_total:.4f}",
            "val_lm_loss": f"{v_lm:.4f}",
            "val_infonce": f"{v_nce:.4f}",
            "train_total_loss": f"{train_total:.4f}",
            "train_lm_loss": f"{train_lm:.4f}",
            "train_infonce": f"{train_nce:.4f}",
        }, hub_repo_id, push_to_hub)

    print("ending of train")
    return criterion