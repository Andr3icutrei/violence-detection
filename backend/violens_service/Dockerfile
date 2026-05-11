FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code/app

COPY ./shared_models/ ../shared_models/

COPY ./violens_service/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

RUN pip install ../shared_models

COPY ./violens_service .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "./key.pem", "--ssl-certfile", "./cert.pem"]
