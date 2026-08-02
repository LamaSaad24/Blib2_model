# @title
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (CrossBatchMemory)

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with learnable temperature, using cosine similarity.
    (This represents the retrieval term in Equation 4 of the paper)

    ✨ محدّثة: تدعم Cross-Batch Memory (XBM) عن طريق memory.py
    لزيادة عدد الـ negatives بدون الحاجة لـ batch size كبير.
    """
    def __init__(self, init_temperature=0.07, use_xbm=False,
                 xbm_capacity=65536, embed_dim=768, device="cuda"):
        super().__init__()
        # Learnable scaling parameter (tau) as mentioned in the paper
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))

        # ---- XBM setup ----
        self.use_xbm = use_xbm
        if self.use_xbm:
            self.memory_target = CrossBatchMemory(embedding_dim=embed_dim, capacity=xbm_capacity, device=device)
            self.memory_query = CrossBatchMemory(embedding_dim=embed_dim, capacity=xbm_capacity, device=device)

    def forward(self, emb_a, emb_b):
        """
        emb_a: (B, D) -- anchor embeddings (Query: Reference Image + Mod Text)
        emb_b: (B, D) -- positive embeddings (Target Image + Empty Text)
        """
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        batch_size = emb_a.size(0)
        scale = self.logit_scale.exp().clamp(max=100.0)

        # 1) التشابه ضمن الـ batch الحالي
        logits_curr = torch.matmul(emb_a, emb_b.t()) * scale  # (B, B)

        # 2) التشابه مع الذاكرة (XBM) -- negatives إضافية فقط
        if self.use_xbm:
            mem_target = self.memory_target.get()
            mem_query = self.memory_query.get()
        else:
            mem_target = None
            mem_query = None

        if mem_target is not None and mem_target.size(0) > 0:
            mem_target = mem_target.to(emb_a.device, emb_a.dtype)
            logits_mem_a = torch.matmul(emb_a, mem_target.t()) * scale
            full_logits_a = torch.cat([logits_curr, logits_mem_a], dim=1)
        else:
            full_logits_a = logits_curr

        logits_curr_t = logits_curr.t()
        if mem_query is not None and mem_query.size(0) > 0:
            mem_query = mem_query.to(emb_a.device, emb_a.dtype)
            logits_mem_b = torch.matmul(emb_b, mem_query.t()) * scale
            full_logits_b = torch.cat([logits_curr_t, logits_mem_b], dim=1)
        else:
            full_logits_b = logits_curr_t

        targets = torch.arange(batch_size).to(emb_a.device)

        loss_a = F.cross_entropy(full_logits_a, targets)
        loss_b = F.cross_entropy(full_logits_b, targets)
        loss = (loss_a + loss_b) / 2

        # 3) تحديث الذاكرة -- فقط أثناء التدريب
        if self.use_xbm and self.training:
            self.memory_target.enqueue(emb_b)
            self.memory_query.enqueue(emb_a)

        return loss

    # ---- لحفظ/استرجاع حالة الذاكرة والـ temperature مع الـ checkpoint ----
    def state_dict_extra(self):
        extra = {}
        if self.use_xbm:
            extra["memory_target"] = self.memory_target.memory
            extra["memory_target_ptr"] = self.memory_target.ptr
            extra["memory_target_size"] = self.memory_target.size
            extra["memory_query"] = self.memory_query.memory
            extra["memory_query_ptr"] = self.memory_query.ptr
            extra["memory_query_size"] = self.memory_query.size
        return extra

    def load_state_dict_extra(self, extra):
        if self.use_xbm and extra:
            self.memory_target.memory = extra["memory_target"]
            self.memory_target.ptr = extra["memory_target_ptr"]
            self.memory_target.size = extra["memory_target_size"]
            self.memory_query.memory = extra["memory_query"]
            self.memory_query.ptr = extra["memory_query_ptr"]
            self.memory_query.size = extra["memory_query_size"]


class ComposedRetrievalLoss(nn.Module):
    """
    Final Multi-task Objective combining Language Modeling and Retrieval.
    Equation 5: L = L_LM + ω * L_InfoNCE
    """
    def __init__(self, omega=1.0, init_temperature=0.07,
                 use_xbm=True, xbm_capacity=65536, embed_dim=768, device="cuda"):
        super().__init__()
        self.omega = omega
        self.info_nce = InfoNCELoss(
            init_temperature=init_temperature,
            use_xbm=use_xbm,
            xbm_capacity=xbm_capacity,
            embed_dim=embed_dim,
            device=device,
        )

    def forward(self, query_embedding, target_embedding, lm_loss):
        loss_info_nce = self.info_nce(query_embedding, target_embedding)
        total_loss = lm_loss + (self.omega * loss_info_nce)
        return total_loss, loss_info_nce, lm_loss