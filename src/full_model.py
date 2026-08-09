# @title
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from peft import get_peft_model, LoraConfig, TaskType
from .retrieval_head import RetrievalHead

cfg = yaml.safe_load(open("config.yaml", "r"))


class FullModel(nn.Module):
    def __init__(self, processor,blip2, embed_dim):
        super().__init__()

        self.processor = processor
        self.blip2 = blip2

        # 2. تجميد جميع الأوزان أولاً
        for param in self.blip2.parameters():
            param.requires_grad = False

        # 3. تطبيق Lora
        lora_config = LoraConfig(
            r=cfg["lora_rank"], #rank of the low-rank decomposition
            lora_alpha=cfg["lora_alpha"],  #scaling factor
            target_modules=["q", "v"],  #Attention layers: Q (query), V (value)
            lora_dropout=cfg["lora_dropout"], #dropout rate for LoRA layers
            bias="none", # كيفية التعامل مع الانحيازات (none, all, lora_only)
            task_type=TaskType.SEQ_2_SEQ_LM #  نوع المهمة (يمكن أن يكون CAUSAL_LM أو SEQ_2_SEQ_LM بناءً على نوع النموذج)
        )

        # تطبيق LoRA على النموذج
        self.blip2.language_model = get_peft_model(self.blip2.language_model, lora_config)

        # gradient checkpointing لتقليل استهلاك الذاكرة، مع use_cache=False
        # (ضرورية معاً -- بدون use_cache=False كان LM loss يتجمّد على
        # نفس القيمة تماماً كل epoch لأن الـ backward pass ما كان يشتغل صح)
        # enable_input_require_grads() ضرورية مع LoRA + gradient checkpointing
        # ============================================================
        self.blip2.language_model.gradient_checkpointing_enable()
        self.blip2.language_model.enable_input_require_grads()
        self.blip2.language_model.config.use_cache = False
        # ============================================================

        # 3. رأس الاسترجاع
        # أخذ الحالات الخفية من نموذج اللغة وتمريرها لرأس الاسترجاع
        t5_hidden_size = self.blip2.config.text_config.d_model #2048
        self.retrieval_head = RetrievalHead(hidden_size=t5_hidden_size, embed_dim=embed_dim)



    def get_embedding(self, images, texts):

      device = next(self.parameters()).device

      # 1. processor
      # تجهيز البيانات من خلال المعالج وتحويل لتنسور لتقديمها ل للقارئ
      #encoder T5
      inputs = self.processor( images=images, text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
      #print("input",type(input)) input <class 'method'>

      # فقط tensors الناتجة
      inputs = {k: v.to(device) for k, v in inputs.items()}

      batch_size = len(images)
      # torch.full() تنشأ مصفوفة بحجم 1
      # طول النص كلمة واحدة فقط T=1
      # قيمة هذه الكلمة
      #pad_token_id
      #رمز لبدء الكلمة الأولى
      decoder_input_ids = torch.full(
          (batch_size, 1),
          self.processor.tokenizer.pad_token_id,
          device=device
      )
      # تشغيل النموذج مررنا ماقرأه encoder T5
      outputs = self.blip2(
          **inputs,
          decoder_input_ids=decoder_input_ids, #ورقة بيضاء يحاول إكمالها Decoder
          output_hidden_states=True, # احتفظ بنسخه لتفكيرك
          return_dict=True
        )

      # 3. extract hidden_states
      hidden_states = outputs.language_model_outputs.decoder_hidden_states[-1]

      # 4. pass to retrieval_head
      embedding = self.retrieval_head(hidden_states)

      return embedding

    def compute_lm_loss(self, images, texts, target_captions):
        """
        Composed captioning loss: model generates the target caption
        from (reference image + modification text).
        """
        device = next(self.parameters()).device

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = self.processor.tokenizer(
            target_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(device)

        labels = labels.to(device)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self.blip2(
            **inputs,
            labels=labels,
            return_dict=True,
        )

        return outputs.loss

    def forward(
        self,
        image_ref,
        image_target,
        modification_text,
        target_captions=None
    ):
        # 1. حساب متجه الاستعلام (صورة مرجعية + نص التعديل)
        query_embedding = self.get_embedding(images=image_ref, texts=modification_text)

        # 2. حساب متجه الهدف (صورة فقط مع نص فارغ)
        batch_size = len(modification_text)
        empty_texts = [""] * batch_size
        target_embedding = self.get_embedding(images=image_target, texts=empty_texts)

        # 2) LM loss (required for full paper)
        if target_captions is None:
            raise ValueError("For full-paper training you must pass target_captions (trg_caption).")

        lm_loss = self.compute_lm_loss(
            images=image_ref,
            texts=modification_text,
            target_captions=target_captions,
        )

        return {
            "query_embedding": query_embedding,
            "target_embedding": target_embedding,
            "language_model_loss": lm_loss
        }