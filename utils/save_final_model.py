import os
import torch


def save_final_model(model, criterion=None, save_dir="outputs/Final_Fashion_Model"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. حفظ أوزان LoRA
    lora_path = os.path.join(save_dir, "lora")
    model.blip2.language_model.save_pretrained(lora_path)

    # 2. حفظ رأس الاسترجاع (Retrieval Head)
    head_path = os.path.join(save_dir, "retrieval_head.pth")
    torch.save(model.retrieval_head.state_dict(), head_path)

    # 3. حفظ logit_scale المتعلّم (جديد -- كان بيضيع سابقاً)
    if criterion is not None:
        torch.save(
            criterion.info_nce.logit_scale.data,
            os.path.join(save_dir, "logit_scale.pth")
        )

    print("\n" + "="*40)
    print(f" 🎉 the final model saved successfull! 🎉 ")
    print(f" 📁 path: {save_dir} ")
    print("="*40)