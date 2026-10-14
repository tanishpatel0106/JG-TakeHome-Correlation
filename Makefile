# Stock Correlation Explorer. Run from the repository root.
# Windows users: the plain commands are listed in README.md ("Quickstart").

PY   ?= python
ZIP  ?= data/raw/stock_data.zip
# Store root: CURRENT + snapshots/<id>/ + overlays/ + ingest_log.jsonl (see corrlib.snapshots)
OUT  ?= data/processed
ROOT ?= $(OUT)
PORT ?= 8000
# make ingest FILES="path/20240909.csv ..." [REASON="..."] [ROOT=...]
FILES  ?=
REASON ?=
# make snapshots ARGS="list" | "show current" | "verify --all" | "rollback ID" | "prune --keep 3" | "unlock"
ARGS   ?= list

# Synthetic demo dataset (no private data needed): python -m corrlib.demo
DEMO_DIR   ?= data/demo
DEMO_STORE ?= $(DEMO_DIR)/store

.PHONY: install install-py install-web audit build-data ingest snapshots api web web-build test bench all clean clean-store \
	demo-data demo-update demo-api demo

install: install-py install-web

install-py:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

install-web:
	cd app/web && npm ci

audit:
	$(PY) -m corrlib.audit --zip $(ZIP) --out reports/data_audit.md

# Full build: publishes a new snapshot in the store root $(ROOT) (earlier snapshots are kept).
build-data:
	$(PY) -m corrlib.build --zip $(ZIP) --out $(ROOT)

# New dates / corrections as a new snapshot: make ingest FILES="20240909.csv" REASON="..."
ingest:
	@test -n "$(FILES)" || { echo 'usage: make ingest FILES="path/YYYYMMDD.csv ..." [REASON="..."] [ROOT=...]'; exit 2; }
	$(PY) -m corrlib.ingest --root $(ROOT) $(FILES) $(if $(REASON),--reason "$(REASON)")

# Snapshot tooling: make snapshots ARGS="list" (default) | "verify --all" | "rollback ID" | "prune --keep 3" | "unlock"
snapshots:
	$(PY) -m corrlib.snapshots --root $(ROOT) $(ARGS)

api:
	CORR_DATA_DIR=$(ROOT) $(PY) -m uvicorn app.api.main:app --port $(PORT)

web:
	cd app/web && npm run dev

web-build:
	cd app/web && npm run build

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/bench.py --data $(ROOT) --out reports/bench.md

all: install audit build-data test bench web-build

# --- demo: deterministic synthetic data, clearly labelled "Synthetic demo data" ---

# Generate the synthetic dataset and publish a full_build snapshot into $(DEMO_STORE).
demo-data:
	$(PY) -m corrlib.demo --out $(DEMO_DIR)
	$(PY) -m corrlib.build --zip $(DEMO_DIR)/demo_stock_data.zip --out $(DEMO_STORE)

# Two update snapshots: the next business day, then a historical price correction (re-running is a no-op).
demo-update:
	$(PY) -m corrlib.ingest --root $(DEMO_STORE) $(DEMO_DIR)/updates/append/*.csv --reason "demo: append the next business day"
	$(PY) -m corrlib.ingest --root $(DEMO_STORE) $(DEMO_DIR)/updates/correction/*.csv --reason "demo: vendor price correction"

demo-api:
	CORR_DATA_DIR=$(DEMO_STORE) $(PY) -m uvicorn app.api.main:app --port $(PORT)

demo: demo-data web-build demo-api

# Regenerable outputs only. The store root is not removed: it holds the snapshot history,
# overlays/ (copies of every ingested file) and the audit log ingest_log.jsonl.
clean:
	rm -rf $(DEMO_DIR) app/web/dist .pytest_cache

# Deletes every snapshot, overlay and the audit log of $(ROOT). Explicit on purpose.
clean-store:
	rm -rf $(ROOT)
