FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "mgmNeural.py", "--ip=0.0.0", "--port=8888", "--allow-root", "--no-browser", "--allow-root"]