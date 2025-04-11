FROM python:3.11-bullseye

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
# Install regular and test dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e ".[test]"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 