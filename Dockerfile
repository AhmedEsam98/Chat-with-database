# ==============================================================================
# Dockerfile: Talk2Tables Container Image
# Packages the FastAPI backend and Streamlit UI in a Python 3.11 slim environment.
#
# Build with:
#     docker build -t talk2tables .
#
# Run with:
#     docker run -p 8000:8000 --env-file .env talk2tables
# ==============================================================================

FROM python:3.11-slim


WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default: start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
