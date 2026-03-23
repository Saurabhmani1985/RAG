# ──────────────────────────────────────────────────────────────────────────────
# Multimodal RAG — Makefile
# Usage: make <target>
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install dev-install run ingest query test test-cov lint fmt clean docker-build docker-run

PYTHON  := python3
UVICORN := uvicorn
PORT    := 8000
PDF     := sample_documents/Diagnostic_Document.pdf

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Multimodal RAG System — available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install      Install production dependencies"
	@echo "    make dev-install  Install dev + test dependencies"
	@echo ""
	@echo "  Run:"
	@echo "    make run          Start FastAPI server (port $(PORT))"
	@echo "    make ingest       Ingest sample PDF into vector store"
	@echo "    make query        Interactive query CLI"
	@echo ""
	@echo "  Test & Quality:"
	@echo "    make test         Run all tests with pytest"
	@echo "    make test-cov     Run tests with coverage report"
	@echo "    make lint         Run ruff linter"
	@echo "    make fmt          Auto-format with black"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build Build Docker image"
	@echo "    make docker-run   Run Docker container"
	@echo ""
	@echo "  Misc:"
	@echo "    make clean        Remove caches and temp files"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

dev-install: install
	$(PYTHON) -m pip install ruff black pytest pytest-asyncio pytest-cov httpx

# ── Run ──────────────────────────────────────────────────────────────────────
run:
	$(UVICORN) main:app --host 0.0.0.0 --port $(PORT) --reload

ingest:
	$(PYTHON) scripts/ingest_pdf.py $(PDF) --collection diagnostic_rag

query:
	$(PYTHON) scripts/query_cli.py

# ── Test ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short \
	  --cov=src --cov=main \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov

# ── Lint & Format ─────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src/ main.py tests/ scripts/

fmt:
	$(PYTHON) -m black src/ main.py tests/ scripts/ --line-length 100

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t multimodal-rag:latest .

docker-run:
	docker run -p $(PORT):$(PORT) --env-file .env \
	  -v $$(pwd)/data:/app/data \
	  multimodal-rag:latest

docker-compose-up:
	docker compose up --build

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned."
