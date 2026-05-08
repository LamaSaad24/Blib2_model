# @title
import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrievalHead(nn.Module):
    def __init__(self, hidden_size, embed_dim=768):
        """hidden_size: input hidden size (H), embed_dim: output embedding size (default 768)"""
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor (B, T, H) -- decoder hidden states
        Returns:
            final_embeds: Tensor (B, embed_dim) -- L2-normalized embeddings
        """
        #أخذ المتوسط >> ثم طبقة خطية1024 ثم>> دالة تفعيل ثم >> طبقة خطية786 ثم>> تسوية

        #hidden_states = hidden_states.to(self.proj1.weight.dtype)
        # 1. Average pooling over T
        # Mean pooling على sequence
        pooled = hidden_states.mean(dim=1)      # (B, H)
        # 2. LayerNorm
        normed = self.norm(pooled)        # (B, H)
        # 3. MLP projection:= Linear -> ReLU -> Linear
        x = self.mlp(normed)                  # (B, 1024)
        # (B, embed_dim)
        # 4. L2 Normalization
        final_embeds = F.normalize(x, p=2, dim=-1)  # (B, embed_dim)
        return final_embeds
