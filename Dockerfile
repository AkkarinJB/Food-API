
FROM python:3.9


WORKDIR /app


COPY . /app/

# ติดตั้ง Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# รันเซิร์ฟเวอร์ FastAPI
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]