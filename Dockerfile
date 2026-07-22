FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV U2NET_HOME=/opt/rembg-models
ENV READY_FILE=/tmp/telegram-bg-remover-ready
ENV HEARTBEAT_FILE=/tmp/telegram-bg-remover-heartbeat
ENV PRELOAD_MODELS=false
ENV PRELOAD_PRIMARY_MODEL=true
ENV OMP_NUM_THREADS=2
ENV SAVE_DEBUG_IMAGES=false

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p "$U2NET_HOME" \
    && python -c "from rembg import new_session; new_session('u2net'); new_session('isnet-general-use')"

RUN useradd --create-home --uid 10001 botuser \
    && chown -R botuser:botuser /app "$U2NET_HOME"

COPY --chown=botuser:botuser bot.py .

USER botuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,time; from pathlib import Path; ready=Path(os.environ['READY_FILE']); heartbeat=Path(os.environ['HEARTBEAT_FILE']); raise SystemExit(0 if ready.is_file() and heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime < 45 else 1)"

CMD ["python", "bot.py"]
