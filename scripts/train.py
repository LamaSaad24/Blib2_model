# @title
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from utils import (save_model_checkpoint,EarlyStopping)
from src import (ComposedRetrievalLoss)

def train(
    model,
    trainloader,
    valloader,
    epochs,
    lr,
    weight_decay,
    warmup_steps,
    device,
    start_step=0 # قيمة افتراضيه
):
    model = model.to(device,  dtype=torch.bfloat16)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        , weight_decay=weight_decay
        )

    total_steps = epochs * len(trainloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Loss Function (L = L_LM + w * L_InfoNCE)
    criterion = ComposedRetrievalLoss(omega=1.0)

    #early_stopping = EarlyStopping(patience=3)
    #torch.autograd.set_detect_anomaly(True) # Added for debugging NaNs

    #save checkpoint settings
    global_step = start_step
    save_every_n_steps = 250  # نحفظ كل 250 خطوة


    for epoch in range(epochs):
        model.train()
        epoch_lm_loss = 0
        epoch_contrastive_loss = 0
        epoch_total_loss = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # نتخطى الدفعات التي تدربنا عليها سابقاً (فقط في الجولة الأولى من الاستئناف)
            if epoch == 0 and batch_idx < start_step:
                continue

            image_ref = batch["ref_images"]
            image_target = batch["trg_images"]
            text = batch["mod_texts"]        # text queries
            target_text = batch["trg_captions"]  # target descriptions

            #helper
            #print("Texts:", text)
            #print("Target Text:", target_text)
            #print("Batch size:", len(text))

            optimizer.zero_grad(set_to_none=True)# before forward

            #with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(
                image_ref=image_ref,
                image_target=image_target,
                modification_text=text,
                target_captions=target_text # لتوليد Text Loss
            )


            query_emb = outputs["query_embedding"]
            target_emb = outputs["target_embedding"]
            lm_loss_from_model = outputs["language_model_loss"]

            # حساب الخسارة الكلية باستخدام ComposedRetrievalLoss
            loss, contrastive_loss, lm_loss = criterion(
                query_emb,
                target_emb,
                lm_loss_from_model
            )
            #helper function
            #check_tensor("query_emb", query_emb)
            #check_tensor("target_emb", target_emb)
            #print("LM Loss:", lm_loss_from_model.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # save checkpoint
            global_step += 1 # زيادة العداد بمقدار 1 مع كل دفعة (Batch)

            # استدعاء التابع بكل أناقة هنا! 👇
            if global_step % save_every_n_steps == 0:
                save_model_checkpoint(model, global_step)

            #فحص الاخطاء
            for name, param in model.named_parameters():
              if param.grad is not None:
                  if torch.isnan(param.grad).any():
                      print(f"NaN gradient in {name}")
                      break


            # Logging
            epoch_total_loss += loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()


            pbar.set_postfix({
                "total loss": f"{loss.item():.4f}",
                "lm loss": f"{lm_loss.item():.4f}",
                "contrastive loss (InfoNCE)": f"{contrastive_loss.item():.4f}"
            })

            #helper function
            #print("Total Loss:", loss.item())
            #print("Contrastive Loss:", contrastive_loss.item())

            if not torch.isfinite(loss):
              print("LOSS EXPLODED!")
              print("Texts:", text)
              print("Target Texts:", target_text)
              break

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        val_total_loss = 0
        val_lm_loss = 0
        val_contrastive_loss = 0

        with torch.no_grad():
            for batch in valloader:
                image_ref = batch["ref_images"]
                image_target = batch["trg_images"]
                text = batch["mod_texts"]
                target_text = batch["trg_captions"]

                #with torch.cuda.amp.autocast(dtype=dtype):
                outputs = model(
                    image_ref=image_ref,
                    image_target=image_target,
                    modification_text=text,
                    target_captions=target_text
                )

                query_emb = outputs["query_embedding"]
                target_emb = outputs["target_embedding"]
                lm_loss_from_model = outputs["language_model_loss"]

                loss, contrastive_loss, lm_loss = criterion(
                    query_emb,
                    target_emb,
                    lm_loss_from_model
                )

                val_total_loss += loss.item()
                val_lm_loss += lm_loss.item()
                val_contrastive_loss += contrastive_loss.item()

        val_total_loss /= len(valloader)
        val_lm_loss /= len(valloader)
        val_contrastive_loss /= len(valloader)

        print(f"\n[Validation] Total Loss: {val_total_loss:.4f} | LM Loss: {val_lm_loss:.4f} | InfoNCE: {val_contrastive_loss:.4f}")

        # ---- EARLY STOPPING ----
        """early_stopping(val_total_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break"""

        print(
                f"[Epoch {epoch+1}] "
                f"Train Total: {epoch_total_loss/len(trainloader):.4f} | "
                f"LM: {epoch_lm_loss/len(trainloader):.4f} | "
                f"InfoNCE: {epoch_contrastive_loss/len(trainloader):.4f}"
            )