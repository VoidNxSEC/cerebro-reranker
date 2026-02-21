# CEREBRO Reranker - Development Commands

set shell := ["bash", "-uc"]
set dotenv-load

# Colors
RED := '\033[0;31m'
GREEN := '\033[0;32m'
YELLOW := '\033[1;33m'
BLUE := '\033[0;34m'
NC := '\033[0m'

# Default recipe
default:
    @just --list --unsorted

# ─── Development ──────────────────────────────────────────────

# Development server with hot reload
dev:
    #!/usr/bin/env bash
    echo -e "{{GREEN}}Starting CEREBRO Reranker in dev mode...{{NC}}"
    just _check-deps
    just _start-services
    nix develop -c watchexec -r -e py \
        'uvicorn src.reranker.server:app --reload --host 0.0.0.0 --port 8000'

# Watch mode - rebuild on changes
watch:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Watching for changes...{{NC}}"
    nix develop -c watchexec -r -e nix,py,rs \
        'just build && echo -e "{{GREEN}}Rebuild complete{{NC}}"'

# Interactive shell with all deps
shell:
    nix develop

# Python REPL with imports
repl:
    #!/usr/bin/env bash
    nix develop -c python -i -c "
    from src.reranker.hybrid_engine import HybridReranker
    from src.reranker.models import ModelRegistry
    from src.reranker.cache import IPFSCache
    import structlog
    log = structlog.get_logger()
    print('CEREBRO Reranker REPL ready')
    "

# ─── Build ────────────────────────────────────────────────────

# Build all packages
build:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Building packages...{{NC}}"
    nix build .#reranker-api -o result-api
    nix build .#trainer -o result-trainer
    nix build .#libscorer -o result-libscorer
    echo -e "{{GREEN}}Build complete{{NC}}"

# Build and load docker images
build-docker:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Building Docker images...{{NC}}"
    nix build .#reranker-api -o result-api
    docker load < result-api
    nix build .#trainer -o result-trainer
    docker load < result-trainer
    echo -e "{{GREEN}}Docker images loaded{{NC}}"
    docker images | grep cerebro-reranker

# ─── Test & Quality ──────────────────────────────────────────

# Run Python tests
test *ARGS='':
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Running tests...{{NC}}"
    nix develop -c pytest tests/ -v --cov=src {{ARGS}}

# Run Rust tests
test-rust:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Running Rust tests...{{NC}}"
    cd src/lib && cargo test

# Test with coverage report
test-cov:
    just test --cov-report=html
    echo -e "{{GREEN}}Coverage report: htmlcov/index.html{{NC}}"

# Lint code
lint:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Linting...{{NC}}"
    nix develop -c ruff check src/
    nix develop -c mypy src/
    echo -e "{{GREEN}}Lint complete{{NC}}"

# Format code
fmt:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Formatting...{{NC}}"
    nix develop -c ruff format src/
    cd src/lib && cargo fmt
    echo -e "{{GREEN}}Format complete{{NC}}"

# Type check
typecheck:
    nix develop -c mypy src/ --strict

# Security audit
audit:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Security audit...{{NC}}"
    nix develop -c pip-audit
    nix develop -c bandit -r src/

# ─── Benchmarks ──────────────────────────────────────────────

# Rust benchmarks (scorer, cache)
bench-rust:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Running Rust benchmarks...{{NC}}"
    cd src/lib && cargo bench

# Python benchmarks
bench-python:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Running Python benchmarks...{{NC}}"
    nix develop -c python scripts/benchmark.py

# All benchmarks
bench: bench-rust bench-python

# ─── Training ────────────────────────────────────────────────

# Train model on GCP
train:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Starting training pipeline...{{NC}}"
    if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo -e "{{RED}}Error: GOOGLE_APPLICATION_CREDENTIALS not set{{NC}}"
        exit 1
    fi
    nix run .#train

# Train locally (no GCP)
train-local:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Training locally...{{NC}}"
    export CEREBRO_DB="postgresql://cerebro@localhost/cerebro"
    nix develop -c python src/training/train.py

# Export model to ONNX + IPFS
export MODEL_PATH:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Exporting model...{{NC}}"
    nix develop -c python src/training/export.py {{MODEL_PATH}}

# ─── Deploy ──────────────────────────────────────────────────

# Deploy to NixOS
deploy:
    #!/usr/bin/env bash
    echo -e "{{GREEN}}Deploying to NixOS...{{NC}}"
    sudo nixos-rebuild switch --flake .#cerebro-reranker
    echo -e "{{GREEN}}Deployment complete{{NC}}"

# Deploy to NixOS (test mode, no activation)
deploy-test:
    sudo nixos-rebuild test --flake .#cerebro-reranker

# ─── Migration ───────────────────────────────────────────────

# Show current migration status
migration-status:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}CEREBRO Reranker Migration Status{{NC}}"
    echo ""
    MODE=$(nix eval .#nixosConfigurations.cerebro.config.services.cerebro.reranker.migration.mode --raw 2>/dev/null || echo "unknown")
    echo -e "{{YELLOW}}Current Mode:{{NC}} $MODE"
    echo ""
    psql -U cerebro -d cerebro -c "
        SELECT
            mode,
            COUNT(*) as requests,
            ROUND(AVG(agreement_score)::numeric, 3) as avg_agreement,
            ROUND(AVG(local_latency_ms)::numeric, 1) as avg_local_ms,
            ROUND(AVG(vertex_latency_ms)::numeric, 1) as avg_vertex_ms
        FROM cerebro.reranker_migration_metrics
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY mode;
    " 2>/dev/null || echo "No migration data yet"

# Progress to next migration stage
migration-progress:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Progressing migration stage...{{NC}}"
    systemctl start cerebro-migration-orchestrator
    journalctl -u cerebro-migration-orchestrator -f

# Run A/B variance analysis
migration-analyze:
    systemctl start cerebro-variance-analyzer
    journalctl -u cerebro-variance-analyzer --no-pager

# Compare results between backends
migration-compare QUERY:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Comparing backends for query: {{QUERY}}{{NC}}"
    echo ""
    echo "Local:"
    curl -s http://localhost:8001/v1/rerank \
        -H "Content-Type: application/json" \
        -d '{"query": "{{QUERY}}", "documents": ["doc1", "doc2"], "mode": "auto"}' \
        | jq '.results[:3]'
    echo ""
    echo "Integration layer:"
    curl -s http://localhost:8002/v1/rerank \
        -H "Content-Type: application/json" \
        -d '{"query": "{{QUERY}}", "documents": ["doc1", "doc2"]}' \
        | jq '.results[:3]'

# ─── Status & Monitoring ─────────────────────────────────────

# Check system status
status:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}System Status{{NC}}"
    echo ""
    for svc in cerebro-reranker cerebro-reranker-integration; do
        echo -e "{{YELLOW}}$svc:{{NC}}"
        systemctl is-active $svc 2>/dev/null || echo "not running"
    done
    echo ""
    echo -e "{{YELLOW}}Redis:{{NC}} $(redis-cli -p 6379 ping 2>/dev/null || echo 'not running')"
    echo -e "{{YELLOW}}IPFS:{{NC}} $(ipfs id 2>/dev/null | jq -r '.ID' || echo 'not running')"
    echo -e "{{YELLOW}}PostgreSQL:{{NC}} $(psql -U cerebro -d cerebro -tc 'SELECT 1' 2>/dev/null && echo 'ok' || echo 'not running')"

# Health check
health:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Health Check{{NC}}"
    curl -sf http://localhost:8000/health | jq '.' || echo -e "{{RED}}API not responding{{NC}}"
    echo ""
    echo -e "{{YELLOW}}Metrics:{{NC}}"
    curl -s http://localhost:8000/metrics | grep -E "rerank_requests_total" || echo "No metrics"

# View logs
logs SERVICE='cerebro-reranker':
    journalctl -u {{SERVICE}} -f --no-pager

# View metrics in terminal
metrics:
    watch -n 1 'curl -s http://localhost:8000/metrics | grep -E "(rerank|cache|gpu)"'

# ─── Database ────────────────────────────────────────────────

db-migrate:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Running migrations...{{NC}}"
    psql -U cerebro -d cerebro -f sql/schema.sql

db-seed:
    psql -U cerebro -d cerebro -f sql/seed.sql

db-console:
    psql -U cerebro -d cerebro

db-backup:
    #!/usr/bin/env bash
    BACKUP_FILE="backups/cerebro_$(date +%Y%m%d_%H%M%S).sql"
    mkdir -p backups
    pg_dump -U cerebro cerebro > "$BACKUP_FILE"
    echo -e "{{GREEN}}Backup saved: $BACKUP_FILE{{NC}}"

# ─── IPFS ─────────────────────────────────────────────────────

ipfs-status:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}IPFS Status{{NC}}"
    ipfs stats repo
    echo ""
    ipfs swarm peers | wc -l | xargs echo "Connected peers:"

ipfs-pin-models:
    systemctl start ipfs-pin-models

ipfs-cluster-status:
    ipfs-cluster-ctl status

# ─── Cache ────────────────────────────────────────────────────

cache-stats:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Cache Statistics{{NC}}"
    redis-cli -p 6379 INFO stats | grep -E "(keyspace|hits|misses)"

cache-clear:
    #!/usr/bin/env bash
    echo -e "{{RED}}Clearing cache...{{NC}}"
    redis-cli -p 6379 FLUSHDB
    echo -e "{{GREEN}}Cache cleared{{NC}}"

cache-snapshot:
    systemctl start cerebro-cache-snapshot

# ─── Models ───────────────────────────────────────────────────

models-list:
    curl -s http://localhost:8000/models | jq '.'

# Update a specific model CID in models.toml
models-update MODEL_ID CID:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Updating {{MODEL_ID}} to CID {{CID}}{{NC}}"
    nix develop -c python3 -c "
    import toml
    with open('configs/models.toml') as f:
        cfg = toml.load(f)
    if '{{MODEL_ID}}' not in cfg.get('models', {}):
        print('Model {{MODEL_ID}} not found in registry')
        exit(1)
    cfg['models']['{{MODEL_ID}}']['ipfs_cid'] = '{{CID}}'
    with open('configs/models.toml', 'w') as f:
        toml.dump(cfg, f)
    print('Updated')
    "
    systemctl restart cerebro-reranker

# ─── GPU ──────────────────────────────────────────────────────

gpu-watch:
    watch -n 1 nvidia-smi

gpu-stats:
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# ─── Profiling ────────────────────────────────────────────────

profile DURATION='30':
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Profiling for {{DURATION}}s...{{NC}}"
    nix develop -c py-spy record -o profile.svg -d {{DURATION}} -s --pid $(pgrep -f uvicorn)
    echo -e "{{GREEN}}Profile saved: profile.svg{{NC}}"

load-test CONCURRENCY='10':
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Load testing with {{CONCURRENCY}} concurrent users{{NC}}"
    nix develop -c locust -f tests/load_test.py --headless \
        -u {{CONCURRENCY}} -r 10 --run-time 60s --host http://localhost:8000

# ─── CI & Release ────────────────────────────────────────────

ci: lint typecheck test test-rust
    echo -e "{{GREEN}}CI checks passed{{NC}}"

pre-commit: fmt lint
    echo -e "{{GREEN}}Pre-commit checks passed{{NC}}"

release VERSION:
    #!/usr/bin/env bash
    echo -e "{{YELLOW}}Releasing v{{VERSION}}...{{NC}}"
    just ci
    git tag -a "v{{VERSION}}" -m "Release v{{VERSION}}"
    just build
    git push origin "v{{VERSION}}"
    echo -e "{{GREEN}}Release v{{VERSION}} complete{{NC}}"

# ─── Cleanup ─────────────────────────────────────────────────

clean:
    #!/usr/bin/env bash
    rm -rf result result-api result-trainer result-libscorer
    rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
    find . -name "*.pyc" -delete
    echo -e "{{GREEN}}Cleanup complete{{NC}}"

clean-all: clean
    #!/usr/bin/env bash
    rm -rf /tmp/cerebro-*
    cd src/lib && cargo clean
    echo -e "{{GREEN}}Deep cleanup complete{{NC}}"

update:
    nix flake update

# ─── Bootstrap ────────────────────────────────────────────────

bootstrap:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}Bootstrapping environment...{{NC}}"
    if ! command -v just &> /dev/null; then
        nix profile install nixpkgs#just
    fi
    echo '#!/usr/bin/env bash' > .git/hooks/pre-commit
    echo 'just pre-commit' >> .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    mkdir -p backups logs data
    just _start-services
    echo -e "{{GREEN}}Bootstrap complete{{NC}}"

# ─── Internal ─────────────────────────────────────────────────

_check-deps:
    #!/usr/bin/env bash
    MISSING=()
    command -v redis-cli &>/dev/null || MISSING+=("redis")
    command -v ipfs &>/dev/null || MISSING+=("ipfs")
    command -v psql &>/dev/null || MISSING+=("postgresql")
    if [ ${#MISSING[@]} -ne 0 ]; then
        echo -e "{{RED}}Missing: ${MISSING[*]} — run: nix develop{{NC}}"
        exit 1
    fi

_start-services:
    #!/usr/bin/env bash
    if ! redis-cli -p 6379 ping &>/dev/null; then
        echo -e "{{YELLOW}}Starting Redis...{{NC}}"
        redis-server --daemonize yes --port 6379
    fi
    if ! ipfs id &>/dev/null 2>&1; then
        echo -e "{{YELLOW}}Starting IPFS...{{NC}}"
        ipfs daemon &>/dev/null &
        sleep 2
    fi
    echo -e "{{GREEN}}Services running{{NC}}"
