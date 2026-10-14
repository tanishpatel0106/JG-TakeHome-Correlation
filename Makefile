# Stock Correlation Explorer. Run from the repository root.
# Windows users: the plain commands are listed in README.md ("Quickstart").

PY   ?= python
ZIP  ?= data/raw/stock_data.zip
OUT  ?= data/processed
PORT ?= 8000

.PHONY: install install-py install-web audit build-data api web web-build test bench all clean

install: install-py install-web

install-py:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

install-web:
	cd app/web && npm ci

audit:
	$(PY) -m corrlib.audit --zip $(ZIP) --out reports/data_audit.md

build-data:
	$(PY) -m corrlib.build --zip $(ZIP) --out $(OUT)

api:
	CORR_DATA_DIR=$(OUT) $(PY) -m uvicorn app.api.main:app --port $(PORT)

web:
	cd app/web && npm run dev

web-build:
	cd app/web && npm run build

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/bench.py --data $(OUT) --out reports/bench.md

all: install audit build-data test bench web-build

clean:
	rm -rf $(OUT) app/web/dist .pytest_cache
