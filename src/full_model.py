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


      """
      print(outputs.keys())
      #odict_keys(['logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs'])
      print(type(outputs))
      #<class 'transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput'>
      print(type(outputs.language_model_outputs))
      #<class 'transformers.modeling_outputs.Seq2SeqLMOutput'>

      print(outputs.language_model_outputs.keys())
      #odict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
      """

      # 3. extract hidden_states
      #hidden_states = outputs.language_model_outputs.decoder_last_hidden_state
      # decoder_hidden_states خلايا دماغ الكاتب
      # -1 الطبقة الاخيره والاعمق تفكير  (B,1,2048)
      hidden_states = outputs.language_model_outputs.decoder_hidden_states[-1]
      #print("get embedding: hidden_states nan:", torch.isnan(hidden_states).any())
      #print("get embedding: hidden_states min:", hidden_states.min().item())
      #print("get embedding: hidden_states max:", hidden_states.max().item())

      # hidden_states (B, T, H) (Batch, Sequence_Length, Hidden_Size)
      #print("hidden_states",hidden_states)
      #tensor([[[....]],[[.....]]],device='cuda:0', dtype=torch.float16)
      #print(" shape hidden_states",hidden_states.shape)
      #torch.Size([2, 8, 2048]) (B, T, H)
      """
      B batch size
      T (sequence length عدد التوكنز
      H hidden size for T5 is 2048
      يعني:
      (2,1,2048)
      لدينا مثالين
      كل منها يحوي 1 توكين
      وكل توكن تمثل ب2048 ميزة

      T =1 seq_len = 1
      decoder_input_ids = [pad/start token واحد]
      نطلب أول خطوة من تمثيل decoder
      لا نأخذ كابشن كامل
      """

      # 4. pass to retrieval_head
      # تحويل الخلاصة لباركود 768
      embedding = self.retrieval_head(hidden_states)
      """
      mean(dim=1)
      T=1 لايتغير شيء
      الناتج (2, 768)
      """

      return embedding

    def compute_lm_loss(self, images, texts, target_captions):
        """
        Composed captioning loss: model generates the target caption
        from (reference image + modification text).
        هون بدنا نتأكد أذا عنجد فهم حقا كيف يدمج الصورة المرجعية و النص
        الورقة اقترحت ان نجعله يكتب
        Composed Captioning
        نعطية صورة فستان أزرق + نص اجعله احمر = فستان احمر
        إذا اخطا نعاقبه
        """
        device = next(self.parameters()).device

        # Encoder >> Inputs: image ref + mod text
        # نعطي القارئ الصورة المرجعية ونص التعديل
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}



        # Labels: target caption tokens
        # (استخدام tokenizer داخل الـprocessor)
        # تجهيز الاجابة النموذجية للمصحح لدينا نحن الوصف الصحيح لنقارنه مع وصف النموذج
        labels = self.processor.tokenizer(
            target_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(device)

        labels = labels.to(device)


        # مهم: تجاهل padding في اللوس
        # الجمل ليست بنفس الطول نضيف اخرها pad
        # تقول للمصحح عاقب النموذج إذا أخطا بالكلمات الاساسية أما الكلمات الفارغة -100 يتجاهلها
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # إدخال كل شي للنموذج وحساب الخطأ
        #ه المره مررناله labels
        # تلقائيا سيكتب  الكلمة ويقارنها بالإجابه النموذجية
        outputs = self.blip2(
            **inputs,
            labels=labels,
            return_dict=True,
        )

        #print("lm outputs loss:", outputs.loss)
        #print("lm logits nan:", torch.isnan(outputs.logits).any())

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