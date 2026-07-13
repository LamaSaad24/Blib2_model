import yaml
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src import (FashionDataset, collate_fn, FullModel)
from utils import (save_final_model,restore_checkpoint)

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

# استرجاع آخر checkpoint (لو موجود) 
# ============================================================
model, resume_state = restore_checkpoint(model, cfg, device)


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
    start_step=cfg['latest_step'],
    resume_state=resume_state,
)

print("trained successfull")

save_final_model(model, criterion=criterion, save_dir="outputs/Final_Fashion_Model")

r1, r10, r50, r_avg = evaluate_model(model, valloader, device="cuda")

print("evaluate successfull")