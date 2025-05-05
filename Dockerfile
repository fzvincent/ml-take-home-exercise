FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./ /app/src

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "src/task1.py"]
CMD ["python", "src/task2.py"]
CMD ["python", "src/task4.py"]