FROM python:3.12.15-slim

ARG GIT_COMMIT=unknown

ENV GIT_COMMIT=${GIT_COMMIT}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src

RUN mkdir -p models reports src/data/raw mlruns /home/appuser/.cache/torch \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.cache/torch \
    && chown -R appuser:appuser /app /home/appuser/.cache /home/appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]