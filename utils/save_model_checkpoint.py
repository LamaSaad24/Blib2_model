# @title
import os
import torch

def save_model_checkpoint(model, step, base_dir="outputs/Blip2_model_Checkpoints"):
    """
    تابع لحفظ أوزان النموذج (LoRA + Retrieval Head) بشكل مرحلي
    """
    # إنشاء اسم المجلد المخصص لهذه الخطوة
    save_path = os.path.join(base_dir, f"step_{step}")
    os.makedirs(save_path, exist_ok=True)

    # 1. حفظ أوزان LoRA
    model.blip2.language_model.save_pretrained(os.path.join(save_path, "lora"))

    # 2. حفظ رأس الاسترجاع
    torch.save(model.retrieval_head.state_dict(), os.path.join(save_path, "retrieval_head.pth"))

    print(f"\n[Checkpoint] ✅ تم حفظ النموذج بأمان عند الخطوة {step} في: {save_path}")