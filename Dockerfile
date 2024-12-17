FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

USER appuser

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT