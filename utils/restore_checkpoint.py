import os
import torch
from peft import PeftModel


def restore_checkpoint(model, cfg, device):
    """
    يسترجع آخر checkpoint محفوظ (لو موجود) حسب cfg['latest_step'].

    بترجع:
        model         : الموديل بعد ما تركّب عليه أوزان LoRA + retrieval_head
        resume_state  : dict فيها optimizer/scheduler/logit_scale/xbm_memory
                        (أو None إذا latest_step == 0 أو الـ checkpoint قديم بدون training_state.pth)

    مستخدمة من main.py و main_v2.py وأي سكربت تاني بده يسترجع نفس الطريقة
    (test_search.py مثلاً)، بدل ما نكرر نفس المنطق بكل ملف.
    """
    resume_state = None
    latest_step = cfg['latest_step']

    if latest_step <= 0:
        print("latest_step = 0 -> بدء تدريب جديد من الصفر (بدون استرجاع)")
        return model, resume_state

    latest_checkpoint = f"{cfg['model_dir']}/Blip2_model_Checkpoints_small/step_{latest_step}"
    print(f"Restor the model from step: {latest_checkpoint}")

    if not os.path.isdir(latest_checkpoint):
        raise FileNotFoundError(
            f"❌ مافي checkpoint بهاد المسار: {latest_checkpoint}\n"
            f"تأكد من latest_step بـ config.yaml، أو حط latest_step: 0 لتدريب من الصفر."
        )

    # 1. تركيب أوزان LoRA المحفوظة فوق نموذج BLIP-2 الأساسي
    model.blip2.language_model = PeftModel.from_pretrained(
        model.blip2.language_model,
        os.path.join(latest_checkpoint, "lora")
    )

    # 2. استعادة أوزان رأس الاسترجاع (Retrieval Head)
    model.retrieval_head.load_state_dict(
        torch.load(os.path.join(latest_checkpoint, "retrieval_head.pth"), weights_only=True)
    )

    # 3. استعادة حالة التدريب الكاملة (optimizer + scheduler + logit_scale + XBM)
    training_state_path = os.path.join(latest_checkpoint, "training_state.pth")
    if os.path.exists(training_state_path):
        resume_state = torch.load(training_state_path, weights_only=False)
        print("[Resume] ✅ training_state.pth loaded (optimizer/scheduler/logit_scale/xbm)")
    else:
        print("[Resume] ⚠️ ما في training_state.pth بهاد الـ checkpoint (checkpoint قديم من قبل التحديث) "
              "-- رح نكمل بس بأوزان الموديل، والـ optimizer/scheduler/temperature رح يبلشوا من جديد.")

    # 4. نقل النموذج كاملاً إلى كرت الشاشة بالصيغة الصحيحة
    model = model.to(device, dtype=torch.bfloat16)

    print("The model was successfully restored")
    return model, resume_state