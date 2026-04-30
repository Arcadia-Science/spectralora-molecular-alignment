FROM python:3.13.1-slim

WORKDIR /app

COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/api/app /app/app
COPY apps/api/static /app/static
COPY scripts /app/scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
