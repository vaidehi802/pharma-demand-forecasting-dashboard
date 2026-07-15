# RxForecast Dashboard - container image for Azure App Service / AWS App Runner / GCP Cloud Run
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY outputs/ ./outputs/
COPY dashboard/ ./dashboard/
COPY src/ ./src/

# Streamlit config: bind to all interfaces, use platform-provided PORT
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
EXPOSE 8501

CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
