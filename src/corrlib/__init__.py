"""corrlib: trailing-window stock return correlations.

Modules
-------
config   constants shared by build, API and tests
io       streaming reads of the daily CSVs inside the source zip
clean    validation / normalisation of one daily file, with issue counting
returns  wide [T x N] prices, uniform-price-day records + overrides, used prices -> simple returns
corr     correlation math (pure numpy)
store    processed artifacts on disk (format 2: write atomically, load with memmap)
audit    data-audit report (``python -m corrlib.audit``)
build    end-to-end build CLI (``python -m corrlib.build``)
demo     deterministic synthetic demo dataset (``python -m corrlib.demo``)
"""

__version__ = "0.2.0"
