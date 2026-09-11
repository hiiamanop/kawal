FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN groupadd --system --gid 10001 kawal && \
    useradd --system --uid 10001 --gid kawal --no-create-home --shell /usr/sbin/nologin kawal

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY contracts/ ./contracts/
COPY services/__init__.py ./services/__init__.py
COPY services/simulator/ ./services/simulator/

RUN chown -R kawal:kawal /app
USER kawal

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "services.simulator.app:app", "--host", "0.0.0.0", "--port", "8000"]
