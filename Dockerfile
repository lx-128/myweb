FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
# install playwright browsers (optional; if running scraper inside container)
RUN python -m playwright install --with-deps
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.headless", "true"]
