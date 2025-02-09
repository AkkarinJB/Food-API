# ใช้ Python 3.9 เป็น Base Image
FROM python:3.9

# กำหนด Working Directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดเข้า Docker Container
COPY . /app/

# ติดตั้ง Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# รันเซิร์ฟเวอร์ FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]