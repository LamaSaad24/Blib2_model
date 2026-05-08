#مكتبة HTTP غير متزامنة (asynchronous client) لإرسال GET وقراءة الردود.
import aiohttp
#إطار العمل الخاص بالبرمجة غير المتزامنة في بايثون.
import asyncio
#إنشاء مجلدات ومسارات الملفات.
import os
#لعرض شريط التقدّم في التيرمنال.
from tqdm import tqdm
#لقراءة/كتابة ملفات بشكل غير متزامن (حتى لا تحجب حلقة الـ event loop).
import aiofiles

# الحد الأقصى للتحميلات المتزامنة
#عدد المهام (requests) التي ستكون قيد التنفيذ في نفس الوقت.
MAX_CONCURRENT_DOWNLOADS = 300

async def download_image(session, sem, url, save_path):
    #sem(Semaphore) : يضمن ألا يتجاوز عدد الاتصالات المفتوحة MAX_CONCURRENT_DOWNLOADS.
    async with sem:  # نستخدم semaphore للحد من عدد الاتصالات المفتوحة
        try:
            async with session.get(url, timeout=10) as response: #جلب المحتوى
                if response.status == 200:
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(await response.read())  #كتابة البيانات بالملف بشكل غير متزامن
                else:
                    pass  # تجاهل الصور التي فشل تحميلها
        except Exception:
            pass  # تجاهل الأخطاء البسيطة (timeouts، إلخ)

async def main(url_file, save_dir, percent=1.0):
    #إنشاء مجلد للحفظ إذا لم يكن موجود
    os.makedirs(save_dir, exist_ok=True)

    # قراءة الروابط
    with open(url_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = int(len(lines) * percent)
    lines = lines[:total]

    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for line in lines:
            path, url = line.strip().split("\t") #تفكيك الجزئين 
            save_path = os.path.join(save_dir, path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True) #إنشاء مجلد الخاص ب المسار 
            tasks.append(download_image(session, sem, url, save_path)) #تنزيل

        # tqdm لدعم شريط التقدم
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f #ينتظر كل مهمة لتكتمل ويحدث Tqdm

if __name__ == "__main__":
    asyncio.run(main("data/raw/image_urls.txt", "data/raw/images", percent=1.0))
