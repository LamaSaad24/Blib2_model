# @title
import os
import torch


def save_model_checkpoint(model, step, optimizer=None, scheduler=None,
                           criterion=None, base_dir="outputs/Blip2_model_Checkpoints_small", push_to_hub=False, hub_repo_id=None):
    """
    تابع لحفظ حالة التدريب الكاملة بشكل مرحلي:
    1. أوزان LoRA
    2. رأس الاسترجاع (Retrieval Head)
    3. optimizer state (زخم Adam)   <-- جديد
    4. scheduler state (مكان الـ warmup/decay)  <-- جديد
    5. logit_scale المتعلّم + حالة ذاكرة XBM (جوا criterion)  <-- جديد
    6. رقم الخطوة الحالية (global_step)  <-- جديد

    ملاحظة: غيّرت اسم المجلد الافتراضي رجّعته "Blip2_model_Checkpoints_small"
    (بدون _v2) عشان يطابق المسار اللي main_v2.py بيدور فيه وقت الاسترجاع.
    """
    """
    يحفظ حالة التدريب الكاملة محلياً، وبشكل اختياري يرفعها لـ HuggingFace Hub
    كنسخة احتياطية -- مفيد جداً على Kaggle لأن الجلسة ممكن تنقطع بأي لحظة
    وتخسر كل شي بـ /kaggle/working لو ما رفعتها لمكان خارجي.
    """
    save_path = os.path.join(base_dir, f"step_{step}")
    os.makedirs(save_path, exist_ok=True)

    # 1. أوزان LoRA
    model.blip2.language_model.save_pretrained(os.path.join(save_path, "lora"))

    # 2. رأس الاسترجاع
    torch.save(model.retrieval_head.state_dict(), os.path.join(save_path, "retrieval_head.pth"))

    # 3+4+5+6: كل شي تاني بملف واحد اسمه training_state.pth
    training_state = {"global_step": step}

    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()

    if criterion is not None:
        # logit_scale + ذاكرة XBM كلهم جوا criterion.info_nce
        training_state["logit_scale"] = criterion.info_nce.logit_scale.data
        training_state["xbm_memory"] = criterion.info_nce.state_dict_extra()

    torch.save(training_state, os.path.join(save_path, "training_state.pth"))

    print(f"\n[Checkpoint] ✅ the model was saved securely at {step} : {save_path}")


    
    # ============================================================
    # ✅ نسخة احتياطية على HuggingFace Hub (حماية من انقطاع Kaggle)
    # ============================================================
    if push_to_hub and hub_repo_id is not None:
        try:
            from huggingface_hub import upload_folder
            upload_folder(
                folder_path=save_path,
                repo_id=hub_repo_id,
                path_in_repo=f"step_{step}",
                repo_type="model",
            )
            print(f"[Checkpoint] ☁️ backed up to HF Hub: {hub_repo_id}/step_{step}")
        except Exception as e:
            # ما بدنا نوقف التدريب لو فشل الرفع (مشكلة إنترنت مؤقتة مثلاً)
            print(f"[Checkpoint] ⚠️ فشل الرفع لـ HF Hub (رح نكمل التدريب عادي): {e}")
 