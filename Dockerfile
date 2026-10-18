FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml ./
COPY src/ src/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPS_LIMITS_PATH=/tmp/ops-agent/limits.json \
    OPS_DATA_DIR=/app/data
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /tmp/ops-agent \
    && chown app:app /tmp/ops-agent
WORKDIR /app
COPY --chown=app:app src/ src/
COPY --chown=app:app data/ data/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/healthz', timeout=3).status == 200 else 1)"]
CMD ["sh", "-c", "exec uvicorn --factory ops_agent.api:create_app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
