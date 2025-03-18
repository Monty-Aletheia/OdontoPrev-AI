FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
COPY Aletheia.pkl ./
COPY app.py ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
