import os
import torch

def save_final_model(model, save_dir="outputs/Final_Fashion_Model"):
    """
    دالة لحفظ النسخة النهائية من النموذج بعد انتهاء التدريب بالكامل.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. حفظ أوزان LoRA
    lora_path = os.path.join(save_dir, "lora")
    model.blip2.language_model.save_pretrained(lora_path)
    
    # 2. حفظ رأس الاسترجاع (Retrieval Head)
    head_path = os.path.join(save_dir, "retrieval_head.pth")
    torch.save(model.retrieval_head.state_dict(), head_path)
    
    print("\n" + "="*40)
    print(f" 🎉 the final model saved successfull! 🎉 ")
    print(f" 📁 path: {save_dir} ")
    print("="*40)