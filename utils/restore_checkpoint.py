import os
import re
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


def _find_latest_local_step(base_dir):
    if not os.path.isdir(base_dir):
        return None
    steps = []
    for name in os.listdir(base_dir):
        m = re.match(r"^step_(\d+)$", name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def _find_latest_hub_step(hub_repo_id):
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(hub_repo_id, repo_type="model")
        steps = set()
        for f in files:
            m = re.match(r"^step_(\d+)/", f)
            if m:
                steps.add(int(m.group(1)))
        return max(steps) if steps else None
    except Exception:
        return None


def restore_checkpoint(model, cfg, device, hub_repo_id=None, auto_detect=True):
    """
    ✅ إصلاح جوهري: model.blip2.language_model هو أصلاً PeftModel
    (تم تطبيق get_peft_model عليه بـ FullModel.__init__). لو استخدمنا
    PeftModel.from_pretrained(model.blip2.language_model, path) هون،
    هاد بيغلّف الموديل بـ LoRA جديدة فوق LoRA الموجودة أصلاً (LoRA داخل LoRA)!
    النتيجة: أوزان LoRA المدرّبة فعلياً ما بتنحمّل أبداً (mismatch بأسماء
    المفاتيح)، والموديل يستمر بأوزان LoRA عشوائية بصمت.

    الحل الصح: نحمّل أوزان LoRA مباشرة *داخل* بنية الـ PeftModel الموجودة
    أصلاً، باستخدام set_peft_model_state_dict (بدل from_pretrained).
    """
    resume_state = None
    base_dir = f"{cfg['model_dir']}/Blip2_model_Checkpoints_small"

    if auto_detect:
        latest_step = _find_latest_local_step(base_dir)
        if latest_step is not None:
            print(f"[Auto-Resume] 🔎 لقيت checkpoint محلي أعلى step: {latest_step}")
        elif hub_repo_id is not None:
            latest_step = _find_latest_hub_step(hub_repo_id)
            if latest_step is not None:
                print(f"[Auto-Resume] 🔎 مافي محلياً، لقيت على HF Hub أعلى step: {latest_step}")

        if latest_step is None:
            print("[Auto-Resume] مافي أي checkpoint محلي ولا على HF Hub -> بدء تدريب جديد من الصفر")
            return model, resume_state, 0
    else:
        latest_step = cfg['latest_step']
        if latest_step <= 0:
            print("latest_step = 0 -> بدء تدريب جديد من الصفر (بدون استرجاع)")
            return model, resume_state, 0

    latest_checkpoint = os.path.join(base_dir, f"step_{latest_step}")
    print(f"Restor the model from step: {latest_checkpoint}")

    if not os.path.isdir(latest_checkpoint) and hub_repo_id is not None:
        print(f"[Restore] المسار المحلي مش موجود، جاري السحب من HF Hub: {hub_repo_id}/step_{latest_step} ...")
        try:
            from huggingface_hub import snapshot_download
            downloaded_path = snapshot_download(
                repo_id=hub_repo_id,
                repo_type="model",
                allow_patterns=[f"step_{latest_step}/*"],
            )
            latest_checkpoint = os.path.join(downloaded_path, f"step_{latest_step}")
            print(f"[Restore] ✅ تم السحب من HF Hub بنجاح: {latest_checkpoint}")
        except Exception as e:
            raise FileNotFoundError(
                f"❌ ما قدرنا نلاقي الـ checkpoint لا محلياً ولا على HF Hub.\n"
                f"تفاصيل الخطأ: {e}"
            )

    if not os.path.isdir(latest_checkpoint):
        raise FileNotFoundError(f"❌ مافي checkpoint بهاد المسار: {latest_checkpoint}")

    # ============================================================
    # ✅ الجزء المصحح: تحميل أوزان LoRA *داخل* الـ PeftModel الموجود
    # بدل ما نغلّفه بـ LoRA جديدة فوق القديمة
    # ============================================================
    lora_path = os.path.join(latest_checkpoint, "lora")
    adapter_weights_path = os.path.join(lora_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_weights_path):
        raise FileNotFoundError(f"❌ مافي adapter_model.safetensors بـ {lora_path}")

    adapter_state_dict = load_file(adapter_weights_path)
    set_peft_model_state_dict(model.blip2.language_model, adapter_state_dict)
    print(f"[Restore] ✅ أوزان LoRA تحمّلت صح داخل الموديل (بدون تكرار/تغليف)")

    model.retrieval_head.load_state_dict(
        torch.load(os.path.join(latest_checkpoint, "retrieval_head.pth"), weights_only=True)
    )

    training_state_path = os.path.join(latest_checkpoint, "training_state.pth")
    if os.path.exists(training_state_path):
        resume_state = torch.load(training_state_path, weights_only=False)
        print("[Resume] ✅ training_state.pth loaded (optimizer/scheduler/logit_scale/xbm)")
    else:
        print("[Resume] ⚠️ ما في training_state.pth بهاد الـ checkpoint -- رح نكمل بس بأوزان الموديل.")

    model = model.to(device, dtype=torch.bfloat16)
    print("The model was successfully restored")

    return model, resume_state, latest_step