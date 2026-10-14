"""corrlib: trailing-window stock return correlations.

Modules
-------
config   constants shared by build, API and tests
io       streaming reads of the daily CSVs inside the source zip
clean    validation / normalisation of one daily file, with issue counting
returns  long prices -> wide [T x N] prices -> simple returns
corr     correlation math (pure numpy)
store    processed artifacts on disk (write atomically, load with memmap)
audit    data-audit report (``python -m corrlib.audit``)
build    end-to-end build CLI (``python -m corrlib.build``)
"""

__version__ = "0.1.0"
