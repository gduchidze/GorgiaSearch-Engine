FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip install uvicorn

COPY . .

CMD [ "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}" ]