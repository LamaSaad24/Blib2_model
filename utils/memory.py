import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBatchMemory:
    """
    FIFO memory bank to store embeddings for contrastive/retrieval objectives.
    Usage: feed .enqueue(new_embeddings), call .get() for all memories.
    تخزين تضمينات من باتش السابقة لاستخدامها كعينات سالبة
    """
    def __init__(self, embedding_dim, capacity=65536, device="cpu"):
        self.capacity = capacity
        self.embedding_dim = embedding_dim # أبعاد تضمينات
        self.device = device
        # مصفوفة  أصفار بحجم سعة الذاكرة
        self.memory = torch.zeros(capacity, embedding_dim, dtype=torch.float32, device=device)
        # pointer مؤشر بيقول وين نكتب التضمين الجايي
        self.ptr = 0
        self.size = 0 # حجم الذاكرة بالبداية صفر إذا ضفنا اكتر بينحذف من القديم

    def enqueue(self, embeddings):
        """
        embeddings: (B, D)
        Adds new embeddings to the memory.
        يضيف تضمينات الجداد للذاكرة
        """
        B = embeddings.size(0) # حساب عدد التضمينات بالباتش الحالي
        # مانحسب التدرجات للتضمينات القديمة
        embeddings = embeddings.detach() # فصل عن الغراف الحساب يلي بحسب  تدرجات وقت التدري لتقليل استهلاك الذاكرة
        
        # لما تكون التضمينات أكبر من الذاكرة نضيف أخر 
        if B >= self.capacity: 
            # نستبدل محتوى الذاكرة بالكامل  أخر تضمينات
            self.memory = embeddings[-self.capacity:].to(self.device)
            self.ptr = 0
            self.size = self.capacity
            return # توقيف تنفيذ دالة enqueue
        # If overflow, wrap around
        end = self.ptr + B 
        # أصغر من سعة الذاكرة
        if end <= self.capacity:
            # إضافة تضمينات من 0 ل حجم تضمينات B
            self.memory[self.ptr:end] = embeddings.to(self.device)
        else:
            # قدي فيه يكتب 
            # كم خانة فاضية 
            first = self.capacity - self.ptr
            # نعبي أخر خانات فاضية
            self.memory[self.ptr:] = embeddings[:first].to(self.device)
            # ضل تضمينات مانكتبت منستبدل الاول FiFo
            self.memory[:end % self.capacity] = embeddings[first:].to(self.device)
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def get(self):
        """
        Returns (num_items, D) tensor of all items in memory (order not guaranteed).
        """
        if self.size < self.capacity:
            # بترجع يلي أمتلأ
            return self.memory[:self.size]
        # If wrapped, memory is unordered; you can re-order if needed
        # الذاكرة كلها تعبت بترجع كل الذاكرة
        return self.memory

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.memory.zero_()

