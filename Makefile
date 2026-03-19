# -------------------------------------------------
#  VATA — Video Audio Text Analytics
#  Project management Makefile
# -------------------------------------------------

ENV_NAME = vata
PYTHON   = conda run -n $(ENV_NAME) python
PIP      = conda run -n $(ENV_NAME) pip

.DEFAULT_GOAL := help

# -- Help ------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "  VATA — project management commands"
	@echo ""
	@echo "  First run:"
	@echo "    make setup        — create environment and install dependencies"
	@echo "    make env-copy     — copy .env.example → .env"
	@echo ""
	@echo "  Daily work:"
	@echo "    make activate     — show environment activation command"
	@echo "    make ollama       — start Ollama in background"
	@echo "    make models       — download recommended Ollama models"
	@echo "    make run f=<file> — run transcription (f=path to .webm)"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make update       — update dependencies from environment.yml"
	@echo "    make check        — check environment and dependencies"
	@echo "    make clean        — remove temporary files"
	@echo "    make reset        — remove and recreate environment"
	@echo "    make info         — show environment info"
	@echo ""

# -- First run -------------------------------------
.PHONY: setup
setup:
	@echo "→ Creating conda environment '$(ENV_NAME)'..."
	conda env create -f environment.yml
	@echo ""
	@echo "✓ Done. Now run:"
	@echo "    conda activate $(ENV_NAME)"
	@echo "    make env-copy"

.PHONY: env-copy
env-copy:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env created. Open it and add your tokens:"; \
		echo "    HF_TOKEN and OPENROUTER_API_KEY"; \
	else \
		echo "! .env already exists, skipping."; \
	fi
	@if [ ! -f config.toml ]; then \
		cp config.toml.example config.toml; \
		echo "✓ config.toml created. Edit settings as needed."; \
	else \
		echo "! config.toml already exists, skipping."; \
	fi

# -- Daily work ------------------------------------
.PHONY: activate
activate:
	@echo ""
	@echo "  Run in terminal:"
	@echo "    conda activate $(ENV_NAME)"
	@echo ""
	@echo "  To deactivate:"
	@echo "    conda deactivate"
	@echo ""

.PHONY: ollama
ollama:
	@echo "→ Starting Ollama in background..."
	ollama serve &
	@sleep 2
	@echo "✓ Ollama is running at http://localhost:11434"

.PHONY: models
models:
	@echo "→ Downloading recommended models..."
	ollama pull llama3.2
	ollama pull mistral
	@echo "✓ Models ready"

.PHONY: run
run:
	@if [ -z "$(f)" ]; then \
		echo "✗ Specify a file: make run f=meeting.webm"; \
		exit 1; \
	fi
	$(PYTHON) main.py transcribe $(f)

# -- Maintenance -----------------------------------
.PHONY: update
update:
	@echo "→ Updating environment from environment.yml..."
	conda env update -n $(ENV_NAME) -f environment.yml --prune
	@echo "✓ Environment updated"

.PHONY: check
check:
	@echo "→ Checking environment..."
	$(PYTHON) main.py check

.PHONY: clean
clean:
	@echo "→ Removing temporary files..."
	find . -name "*.wav" -not -path "./.git/*" -delete
	find . -name "*.tmp" -not -path "./.git/*" -delete
	find . -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.git/*" -delete
	rm -rf tmp/
	@echo "✓ Temporary files removed"

.PHONY: reset
reset:
	@echo "→ Removing environment '$(ENV_NAME)'..."
	conda env remove -n $(ENV_NAME) -y
	@echo "→ Recreating..."
	conda env create -f environment.yml
	@echo "✓ Environment recreated"

.PHONY: info
info:
	@echo "─── Conda ────────────────────────────────"
	conda info
	@echo "─── Environment '$(ENV_NAME)' ─────────────"
	conda run -n $(ENV_NAME) conda list | grep -E "python|torch|whisper|pyannote|ffmpeg"
	@echo "─── Ollama ───────────────────────────────"
	ollama list 2>/dev/null || echo "  Ollama is not running"
