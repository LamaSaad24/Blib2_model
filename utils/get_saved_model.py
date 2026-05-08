# @title
import torch
from peft import PeftModel

# 1. تحديد أحدث مجلد قمنا بحفظه

latest_step = x
latest_checkpoint = f"/content/drive/MyDrive/Blip2_model_Checkpoints/step_{latest_step}"

print(f"جاري استعادة النموذج من الخطوة: {latest_checkpoint}")

# 2. تركيب أوزان LoRA المحفوظة فوق نموذج BLIP-2 الأساسي
model.blip2.language_model = PeftModel.from_pretrained(
    model.blip2.language_model,
    os.path.join(latest_checkpoint, "lora")
)

# 3. استعادة أوزان رأس الاسترجاع (Retrieval Head)
model.retrieval_head.load_state_dict(
    torch.load(os.path.join(latest_checkpoint, "retrieval_head.pth"))
)

# 4. نقل النموذج كاملاً إلى كرت الشاشة بالصيغة الصحيحة
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device, dtype=torch.bfloat16)

print("تم استعادة النموذج بنجاح! جاهز لإكمال التدريب.")