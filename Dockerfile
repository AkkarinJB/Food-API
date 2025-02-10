
FROM python:3.9


WORKDIR /app


COPY . /app/

#Dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Runserver FastAPI
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]