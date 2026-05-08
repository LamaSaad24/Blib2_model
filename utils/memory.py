import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBatchMemory:
    """
    FIFO memory bank to store embeddings for contrastive/retrieval objectives.
    Usage: feed .enqueue(new_embeddings), call .get() for all memories.
    """
    def __init__(self, embedding_dim, capacity=65536, device="cpu"):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.device = device
        self.memory = torch.zeros(capacity, embedding_dim, dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def enqueue(self, embeddings):
        """
        embeddings: (B, D)
        Adds new embeddings to the memory.
        """
        B = embeddings.size(0)
        embeddings = embeddings.detach()
        if B >= self.capacity:
            self.memory = embeddings[-self.capacity:].to(self.device)
            self.ptr = 0
            self.size = self.capacity
            return
        # If overflow, wrap around
        end = self.ptr + B
        if end <= self.capacity:
            self.memory[self.ptr:end] = embeddings.to(self.device)
        else:
            first = self.capacity - self.ptr
            self.memory[self.ptr:] = embeddings[:first].to(self.device)
            self.memory[:end % self.capacity] = embeddings[first:].to(self.device)
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def get(self):
        """
        Returns (num_items, D) tensor of all items in memory (order not guaranteed).
        """
        if self.size < self.capacity:
            return self.memory[:self.size]
        # If wrapped, memory is unordered; you can re-order if needed
        return self.memory

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.memory.zero_()

