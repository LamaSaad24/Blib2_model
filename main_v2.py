import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
from src import (FashionDataset, collate_fn, FullModel)
from utils import (save_final_model)

from scripts.train import train
from scripts.evaluate import evaluate_model

from huggingface_hub import login
login("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")


cfg = yaml.safe_load(open("config.yaml", "r"))


processor = Blip2Processor.from_pretrained(cfg['blip2_model_id'])
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    cfg['blip2_model_id'],
    torch_dtype=torch.bfloat16,
)

print("blip2 model loaded successfull")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullModel(processor, blip2, cfg['embed_dim'])
print(f"model dtype: {model.blip2.dtype}")

# ============================================================
# استرجاع آخر checkpoint (لو موجود)
# ✅ صار المسار موحّد مع save_model_checkpoint.py (بدون _v2)
# ✅ صار latest_step ياخد من config.yaml بس، مش مكتوب يدوياً مرتين
# ============================================================
resume_state = None
latest_step = cfg['latest_step']

if latest_step > 0:
    latest_checkpoint = f"{cfg['model_dir']}/Blip2_model_Checkpoints_small/step_{latest_step}"
    print(f"Restor the model from step: {latest_checkpoint}")

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
    #    ✅ هاد الجزء كان ناقص بالكامل سابقاً
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
else:
    print("latest_step = 0 -> بدء تدريب جديد من الصفر (بدون استرجاع)")


dataset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/train_triplets.csv")
valset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/val_triplets.csv")

print("dataset created successfull")

dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4)
valloader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4)

print("dataloader created successfull")

criterion = train(
    model,
    dataloader,
    valloader,
    epochs=cfg['epochs'],
    lr=cfg['lr'],
    weight_decay=cfg['weight_decay'],
    warmup_steps=cfg['warmup_steps'],
    device=device,
    start_step=latest_step,
    resume_state=resume_state,   # ✅ جديد
)

print("trained successfull")

# حفظ النموذج النهائي (شامل logit_scale الآن)
save_final_model(model, criterion=criterion, save_dir="outputs/Final_Fashion_Model")

r1, r10, r50, r_avg = evaluate_model(model, valloader, device="cuda")

print("evaluate successfull")