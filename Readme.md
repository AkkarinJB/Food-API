### 🍽 Food Recommendation System - Backend

🚀 **Backend ของระบบแนะนำอาหารสำหรับผู้ป่วยเบาหวาน**  
Backend พัฒนาโดยใช้ **FastAPI** พร้อม **Machine Learning (KNN + K-Means Clustering)** เพื่อแนะนำอาหารที่เหมาะสม

---

## 📌 **คุณสมบัติของโปรเจค**
✅ รับข้อมูลสุขภาพ เช่น อายุ, น้ำหนัก, ส่วนสูง, ระดับกิจกรรม  
✅ ใช้อัลกอริธึม **KNN และ K-Means Clustering** เพื่อแนะนำอาหารที่เหมาะสม  
✅ แสดงรายการอาหารแนะนำที่มี **Glycemic Index (GI) ต่ำ**  
✅ รองรับการเชื่อมต่อผ่าน API กับ **Frontend React**  
✅ Deploy บน **Railway** พร้อมรองรับการใช้งานผ่านอินเทอร์เน็ต  

---

## 🛠 **เทคโนโลยีที่ใช้**
- **FastAPI** (สร้าง REST API)  
- **Pandas & NumPy** (จัดการข้อมูล)  
- **Scikit-learn** (สร้างโมเดล Machine Learning)  
- **Uvicorn** (รัน FastAPI Server)  
- **Railway** (Deploy Backend API)  

---

## 📂 **โครงสร้างโปรเจค**
```
backend/
│── data/               # 📂 ไฟล์ข้อมูลโภชนาการอาหาร
│   ├── Food_and_Nutrition.csv
│   ├── pred_food.csv
│── main.py             # 🚀 API หลักของ FastAPI
│── model.py            # 🤖 โหลดและ train โมเดล KNN
│── requirements.txt    # 📦 รายการ dependencies
│── Dockerfile          # 🐳 สำหรับ deploy บน Railway
│── README.md           # 📖 คำอธิบายโปรเจค
```

---

## ⚙️ **การติดตั้งและใช้งาน**
### 🚀 **1. Clone Repo**
```sh
git clone https://github.com/AkkarinJB/Food-API.git
cd food-recommendation-backend
```

### 🔧 **2. สร้าง Virtual Environment และติดตั้ง Dependencies**
```sh
python -m venv env
source env/bin/activate  # สำหรับ macOS/Linux
env\Scripts\activate     # สำหรับ Windows
pip install -r requirements.txt
```

### 🏃‍♂️ **3. รันเซิร์ฟเวอร์ FastAPI**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
🔹 เปิด API Docs ได้ที่: `http://127.0.0.1:8000/docs`

---

## 🚀 **Deploy ขึ้น Railway**
### **ติดตั้ง Railway CLI**
```sh
curl -fsSL https://railway.app/install.sh | sh
```
### **ล็อกอินและสร้างโปรเจค**
```sh
railway login
railway init
railway up
```
✅ ระบบจะแสดง API URL เช่น:  
```
✅ Deployed at: https://your-api-url.up.railway.app
```

---

## 📚 **API ที่ใช้**
📌 **Backend API (FastAPI)**
| Method | Endpoint            | คำอธิบาย               |
|--------|-----------------|------------------------|
| `POST` | `/recommend`    | รับข้อมูลสุขภาพและแนะนำอาหาร |
| `GET`  | `/`             | ตรวจสอบ API ว่าทำงานอยู่ |

---

## 🛠 **การตั้งค่า Environment Variables**
📌 สร้างไฟล์ `.env` และเพิ่มค่า API URL
```env
PORT=8000
MODEL_PATH=model.pkl
```

---




