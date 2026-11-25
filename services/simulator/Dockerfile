FROM python:3.12-alpine

# Alpine keeps the simulator image small; all runtime dependencies are pure
# Python or ship musl-compatible wheels.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN addgroup -S -g 10001 kawal && \
    adduser -S -u 10001 -G kawal -h /app -s /sbin/nologin kawal

COPY requirements-simulator.txt ./
RUN pip install --no-cache-dir -r requirements-simulator.txt

COPY contracts/ ./contracts/
COPY services/__init__.py ./services/__init__.py
COPY services/simulator/ ./services/simulator/

RUN chown -R kawal:kawal /app
USER kawal

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "services.simulator.app:app", "--host", "0.0.0.0", "--port", "8000"]
