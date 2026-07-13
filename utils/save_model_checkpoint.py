# @title
import os
import torch


def save_model_checkpoint(model, step, optimizer=None, scheduler=None,
                           criterion=None, base_dir="outputs/Blip2_model_Checkpoints_small"):
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