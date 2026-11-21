# Imagem da API do assistente (FastAPI + uvicorn), usada no Azure Container Apps.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependências primeiro: esta camada só é refeita quando o pyproject muda (build mais rápido).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[pgvector,openai,azure]"

# Notas de exemplo, indexadas pela rota /ingest depois do deploy.
COPY data ./data

# Não roda como root: se a aplicação for comprometida, o estrago fica limitado.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

ENV INGEST_ROOT=/app/data
EXPOSE 8000
CMD ["python", "-m", "assistant", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
