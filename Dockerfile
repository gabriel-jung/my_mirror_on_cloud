FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN apt-get update && apt-get install -y build-essential g++ curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi --no-root && \
    apt-get purge -y --auto-remove build-essential g++ curl && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim AS runner

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .
COPY ./models /app/models
ENV HF_HOME="/app/models"

ENV PYTHONPATH="/app/src"

EXPOSE 8080

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
