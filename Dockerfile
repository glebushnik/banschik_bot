FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY requirements.lock ./
COPY banschik ./banschik
COPY prompts ./prompts
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home bot \
    && mkdir -p /app/data /app/backups && chown -R bot:bot /app
USER bot
ENTRYPOINT ["banschik"]
CMD ["run"]
