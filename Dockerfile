# One image for the Streamlit dashboard, the LangGraph agents and the local ChromaDB index.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY followup ./followup
COPY streamlit_app.py .
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser && mkdir -p /app/storage /app/vectorstore && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
