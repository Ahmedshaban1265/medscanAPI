# خطوات النشر على Railway - ملخص سريع

## 1. تحضير الملفات (✅ جاهز)
- app.py (محسن للنشر)
- requirements.txt
- nixpacks.toml
- Procfile
- .gitattributes
- README.md

## 2. إعداد Git Repository
```bash
cd railway-flask-deployment
git init
git add .
git commit -m "Initial commit: Medical AI Flask API"
```

## 3. رفع على GitHub
```bash
# إنشاء repository جديد على GitHub أولاً
git remote add origin https://github.com/username/medical-ai-railway.git
git branch -M main
git push -u origin main
```

## 4. نشر على Railway
1. اذهب إلى https://railway.app
2. سجل دخول بـ GitHub
3. اضغط "Start a New Project"
4. اختر "Deploy from GitHub repo"
5. اختر repository الخاص بك
6. اضغط "Deploy Now"

## 5. الحصول على URL
1. في لوحة Railway، اذهب إلى "Settings"
2. اضغط على "Networking"
3. اضغط "Generate Domain"
4. ستحصل على URL مثل: https://your-app-name.up.railway.app

## 6. اختبار التطبيق
```bash
curl https://your-app-name.up.railway.app/health
```

## ملاحظات مهمة:
- تأكد من رفع ملفات النماذج (.h5) مع المشروع
- إذا كانت الملفات كبيرة، استخدم Git LFS
- راقب استهلاك الساعات المجانية (500 ساعة/شهر)
- التطبيق محسن لاستهلاك ذاكرة أقل عبر تحميل النماذج عند الطلب

