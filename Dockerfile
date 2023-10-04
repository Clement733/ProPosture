FROM python:3.10.6-slim
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY MLOps/ /app/
EXPOSE 8080
CMD ["python", "main.py"]
