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

# image -> vision encoder -> visual features-> Qformer(أهم معلومات المرئية)->t5 encoder نموذج اللغة (يبدأ بتوكن بداية بعدين بيكمل الجملة)
# text -> t5 encoder 
# تمثيل الموحد للصورة والنص ثم نموذج اللغة المفكك T5 decoder

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

        # تطبيق LoRA على نموذج اللغة
        self.blip2.language_model = get_peft_model(self.blip2.language_model, lora_config)

        # gradient checkpointing 
        # لتقليل استهلاك الذاكرة
        # ، مع 
        # use_cache=False
        # (ضرورية معاً -- بدون 
        # use_cache=False 
        # كان 
        # LM loss 
        # يتجمّد على
        # نفس القيمة تماماً كل 
        # epoch
        # لأن 
        # الـ 
        # backward pass 
        # ما كان يشتغل صح)
        # enable_input_require_grads() 
        # ضرورية مع 
        # LoRA + gradient checkpointing
        # ============================================================
        # enable gradient checkpointing for the language model to save memory
        self.blip2.language_model.gradient_checkpointing_enable()
        #تقليل استهلاك الذكره يلي بيحتاجه بيرجع بيحفظه
        # enable input gradients for LoRA layers
        self.blip2.language_model.enable_input_require_grads()
        # disable caching to ensure gradients are computed correctly
        self.blip2.language_model.config.use_cache = False

        # 3. رأس الاسترجاع
        # أخذ الحالات الخفية من نموذج اللغة وتمريرها لرأس الاسترجاع
        t5_hidden_size = self.blip2.config.text_config.d_model #2048
        self.retrieval_head = RetrievalHead(hidden_size=t5_hidden_size, embed_dim=embed_dim)


    def get_embedding(self, images, texts):

      device = next(self.parameters()).device

      # 1. processor
      # تجهيز البيانات من خلال المعالج وتحويل لتنسور لتقديمها ل للقارئ
      #encoder T5
      inputs = self.processor( images=images, text=texts, 
                            return_tensors="pt", 
                            #red [PAD] [PAD] [PAD] 
                            #make the shirt red [PAD]
                            #change the shirt color to red
                            padding=True, 
                            # long text x
                            truncation=True).to(device)
      #print("input",type(input)) input <class 'method'> pixel values, attention mask, input ids

      # فقط tensors الناتجة 
      #نقل التنسور على الجهاز
      inputs = {k: v.to(device) for k, v in inputs.items()}

      batch_size = len(images)
      # torch.full() تنشأ مصفوفة بحجم 1
      # طول النص كلمة واحدة فقط T=1
      # قيمة هذه الكلمة
      #pad_token_id
      #رمز لبدء الكلمة الأولى
    #   نريد تشغيل   Blip2 للحصول على decoder hidden states
    # ليس هدفنا توليد caption كامل
    # نقول ل decoder ابدأ من توكين واحد وانا اريد  
    # hidden representation الناتج
    #تهيئة الإدخال الاول وتملا كل تنسور بنفس القيمة والحجم انا بححده
    #نقطة بداية مو نص حقيقي بس لناخد الحالة المخفية hidden state
    #نقطة بداية لكل عينة بالباتش  
    # توكن بداية لكل عينة
    #ليحسب decoder
    #pad token id
    #hidden state لنحسب
    #لنحسب بطريقة منظمة وصحيحة
      decoder_input_ids = torch.full(
          (batch_size, 1), # tensore shape
          self.processor.tokenizer.pad_token_id,
          device=device
      )
      #  t5 encoder(بيستقبل النص) decoder (بيولد النص)
      # تشغيل النموذج مررنا ماقرأه encoder T5
      outputs = self.blip2(
          **inputs, # ** بيفك قاموس وبيعت كل وسيط مع قيمته pixel values = input.pixel values
          decoder_input_ids=decoder_input_ids, #ورقة بيضاء يحاول إكمالها Decoder
          output_hidden_states=True, # احتفظ بنسخه لتفكيرك
          return_dict=True
        )

      # 3. extract hidden_states
      hidden_states = outputs.language_model_outputs.decoder_hidden_states[-1]
      # استخراج اخر حالة 
      # الطبقات الاخيرة من المحول تكون اقرب للتمثيل النهائي
      #hidden state تمثيل رقمي عالي الأبعاد لما فهمه النموذج

      # 4. pass to retrieval_head
      embedding = self.retrieval_head(hidden_states)

      return embedding

    def compute_lm_loss(self, images, texts, target_captions):
        """
        Composed captioning loss: model generates the target caption
        from (reference image + modification text).
        language model loss خسارة نموذج اللغة لتوليد اللغة
        cross entropy loss  
        مقارنة كابشن يلي ولده الموديل مع كابشن الهدف 
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

        # تحضير كابشن الهدف ك تنسور ليدخل للنموذج
        labels = self.processor.tokenizer(
            target_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(device)

        labels = labels.to(device)

        # استبدل كل pad_token_id = -100
        # الخسارة تحسب فقط للكلمات الحقيقة 
        # cross entropy loss  تتجاهل -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100 # ignore padding

        outputs = self.blip2(
            **inputs,
            labels=labels,
            return_dict=True,
        )
        

        return outputs.loss

    def forward(self, image_ref, image_target, modification_text, target_captions=None ):
        # 1. حساب متجه الاستعلام (صورة مرجعية + نص التعديل)
        # فستان أحمر + بدي أسود = خصئص بصرية للفستان بس لون أسود
        query_embedding = self.get_embedding(images=image_ref, texts=modification_text)

        # 2. حساب متجه الهدف (صورة فقط مع نص فارغ)
        # منتج نهائي الصورة الصحيحه مابدي عليها تعديل لهيك بمرق النص الفارغ
        batch_size = len(modification_text)
        empty_texts = [""] * batch_size
        target_embedding = self.get_embedding(images=image_target, texts=empty_texts)

        if target_captions is None:
            raise ValueError("For full-paper training you must pass target_captions (trg_caption).")

        # 3) LM loss (required for full paper)
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