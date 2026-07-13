import yaml
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src import (FashionDataset,collate_fn,FullModel)
from utils import (save_final_model)

from scripts.train import train
from scripts.evaluate import evaluate_model

from huggingface_hub import login
login("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")


cfg = yaml.safe_load(open("config.yaml", "r"))


processor = Blip2Processor.from_pretrained(cfg['blip2_model_id'])
blip2 = Blip2ForConditionalGeneration.from_pretrained(cfg['blip2_model_id'],
    torch_dtype=torch.bfloat16, # <--- السر هنا! يقلص الحجم للنصف ويحل مشكلة الذاكرة NAN
    #device_map="auto"          # يوزع النموذج بذكاء على كرت الشاشة
)

print("blip2 model loaded successfull")

dataset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/train_triplets.csv")
valset = FashionDataset(cfg['image_dir'], f"{cfg['data_dir']}/val_triplets.csv")

print("dataset created successfull")

dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn,num_workers=4)
valloader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn,num_workers=4)

print("dataloader created successfull")



device = "cuda" if torch.cuda.is_available() else "cpu"
model = FullModel(processor,blip2,cfg['embed_dim'])
print(f"model dtype: {model.blip2.dtype}")



train(
    model,
    dataloader,
    valloader,
    epochs=cfg['epochs'],
    lr=cfg['lr'],
    weight_decay=cfg['weight_decay'],
    warmup_steps=cfg['warmup_steps'],
    device=device,
    start_step= 0
     ## <--- إخبار الدالة أن تبدأ من الخطوة 3000 متجاهلة ما قبلها!
)

print("trained successfull")

# 2. حفظ النموذج النهائي مباشرة بعد انتهاء الـ train
save_final_model(model, save_dir="outputs/Final_Fashion_Model")

# تأكدي من تمرير test_dataloader (مجموعة البيانات المخصصة للاختبار والتي لم يراها النموذج أثناء التدريب)
r1, r10, r50, r_avg = evaluate_model(model, valloader, device="cuda")

print("evaluate successfull")