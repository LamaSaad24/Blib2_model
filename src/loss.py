# @title
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with learnable temperature, using cosine similarity.
    (This represents the retrieval term in Equation 4 of the paper)
    """
    def __init__(self, init_temperature=0.07):
        super().__init__()
        # Learnable scaling parameter (tau) as mentioned in the paper
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))
         # معامل التحجيم للـ temperature
                                   #log(tensor(0.0700)):= -2.6593
        #Parameter containing: tensor(-2.6593, requires_grad=True)
    def forward(self, emb_a, emb_b):
        """
        emb_a: (B, D) -- anchor embeddings (Query: Reference Image + Mod Text)
        emb_b: (B, D) -- positive embeddings (Target Image + Empty Text)
        Returns: scalar loss
        (Batch,dimension)
        """
        # Normalize embeddings to unit hypersphere (Equation 2 constraint)
        """
        F.normalize >>>>>> اتفاقية الأشعة (التطبيع): >>>>>>
        يعمل على وضع المتجهات في فضاء كروي الموحد
        (Unit Hypersphere)
        هذا يجعل النموذج يعتمد على "الزاوية" بين المتجهات (تشابه جيب التمام)
         وليس على حجمها
        """
        emb_a = F.normalize(emb_a, p=2, dim=-1) # يتحول كل تضمين إلى شعاع وحدة بطول 1
        emb_b = F.normalize(emb_b, p=2, dim=-1)  # ||x|| = ||y|| = 1 => dot product = cosine similarity
        #النموذج قد يغش عبر تكبير القيم بدل تحسين الاتجاه.
        """print("infoNCE:emb_a",emb_a)
        print("infoNCE:emb_a shape",emb_a.shape)
        print("infoNCE:",emb_b)
        print("emb_b.t()",emb_b.t())
        print("infoNCE shape:",emb_b.shape)"""
        # Compute pairwise cosine similarity
        # logits : مصفوفة التشابه
        # القطر الرئيسي الصحيح والباقي خاطئ
        logits = torch.matmul(emb_a, emb_b.t())  # (B, B)
        """print("logits",logits)
        print("logits shape",logits.shape)"""
        """
        batch size =2
            b1    b2
        a1  sim  sim
        a2  sim  sim

        positive pairs  a1b1  a2b2
        """

        # Scale by learnable temperature
        scale = self.logit_scale.exp().clamp(max=100.0)
        """print("self.logit_scale",self.logit_scale)
        print("self.logit_scale exp",self.logit_scale.exp())
        print("scale",scale)"""

        #helper function
        #check_tensor("emb_a", emb_a)
        #check_tensor("emb_b", emb_b)
        #check_tensor("logits_before_scale", logits)
        #check_tensor("logits_after_scale", logits * scale)
        #print("scale:", scale.item())


        logits = logits * scale
        #print("logits * scale",logits)
        # similarity/t  or  similarity * 1/t
        # temp تتحكم بحدة softmax

        # Targets: diagonal elements are the positive pairs
        # العناصر القطرية هي الأزواج الموجبة
        targets = torch.arange(logits.size(0)).to(logits.device)

        # Cross-entropy over rows (anchor->positive)
        loss_a = F.cross_entropy(logits, targets)
        # لكل صف تقول للنموذج ضع أعلى احتمال على العنصر القطري

        # Cross-entropy over cols (positive->anchor)
        loss_b = F.cross_entropy(logits.t(), targets) #الاتجاه العكسي Symmetric Contrastive Loss
        #الاستعلام يجد الهدف وأيضا الهدف يجد الاستعلام


        # InfoNCE is the symmetric average المتوسط المتماثل
        #Symmetric Contrastive Learning
        # الخسارة المتماثلة الاستعلام يبحث عن الهدف والهدف يبحث عن الهدف
        loss = (loss_a + loss_b) / 2
        return loss #InfoNCE Retrieval Loss


class ComposedRetrievalLoss(nn.Module):
    """
    Final Multi-task Objective combining Language Modeling and Retrieval.
    Equation 5: L = L_LM + ω * L_InfoNCE
    يجمع الخسارات
    """
    def __init__(self, omega=1.0, init_temperature=0.07):
        super().__init__()
        self.omega = omega  #ω is the weight for the retrieval loss term
        self.info_nce = InfoNCELoss(init_temperature=init_temperature) # Instantiate the InfoNCE loss module

    def forward(self, query_embedding, target_embedding, lm_loss):
        """
        query_embedding: Output from the Q-Former -> T5 -> Retrieval Head
        target_embedding: Output from the Q-Former -> T5 -> Retrieval Head
        lm_loss: Language modeling loss directly from T5
        """
        # 1. Calculate Retrieval Loss (InfoNCE)
        loss_info_nce = self.info_nce(query_embedding, target_embedding) # Calculate the InfoNCE loss between query and target embeddings

        # 2. Combine with Language Modeling Loss
        # lm_loss : قدرة النموذج على وصف الصورة بعد التعديل
        # loss_info_nce: تقريب التضمين مع التضمين الصحيح للصورة الهدف وإبعاده عن الخاطئة
        total_loss = lm_loss + (self.omega * loss_info_nce) # Final loss as per Equation 5 in the paper

        return total_loss, loss_info_nce, lm_loss