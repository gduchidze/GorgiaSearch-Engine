FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip install uvicorn

USER appuser

COPY . .

CMD [ "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}" ]