import yaml
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src import (FashionDataset, collate_fn, FullModel)
from utils import (save_final_model, restore_checkpoint)

from scripts.train import train
from scripts.evaluate import evaluate_model

from huggingface_hub import login

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = "x"

login(hf_token)

HUB_REPO_ID = "lama65/train_BLIP2_model_2_activate_XBM_gradually"
PUSH_TO_HUB = True

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
# ✅ auto_detect=True: بيكتشف تلقائياً أعلى step محفوظ (محلياً أو HF Hub)
# لاحظ: صار في 3 قيم إرجاع الآن (model, resume_state, detected_step)
# ============================================================
model, resume_state, detected_step = restore_checkpoint(
    model, cfg, device, hub_repo_id=HUB_REPO_ID, auto_detect=True
)

dataset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/train_triplets.csv")
valset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/val_triplets.csv")

print("dataset created successfull")

dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2)
valloader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2)

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
    start_step=detected_step,   # ✅ مكتشف تلقائياً، مش من config.yaml
    resume_state=resume_state,
    push_to_hub=PUSH_TO_HUB,
    hub_repo_id=HUB_REPO_ID,
    scheduler_name=cfg.get('scheduler', 'linear_with_warmup'),
)

print("trained successfull")

save_final_model(model, criterion=criterion, save_dir="outputs/Final_Fashion_Model")

r1, r10, r50, r_avg = evaluate_model(model, valloader,
                                      device="cuda",
                                        recall_ks=tuple(cfg.get('recall_ks', [1, 10, 50]))
                                     )

print("evaluate successfull")