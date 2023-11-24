FROM python:3.10.6-slim
WORKDIR /app
COPY requirements.txt /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
RUN pip install --no-cache-dir -r requirements.txt
COPY MLOps/ /app/
EXPOSE 8080
CMD ["python", "main.py"]
