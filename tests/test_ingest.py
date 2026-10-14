"""Ingestion of new dates and corrections (corrlib.ingest).

The central property: an incremental update produces exactly the artifacts a
full rebuild from base + overlay files would produce, while recomputing only
the window-end dates whose windows contain a changed return row.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import daily_csv_text, write_zip

from corrlib import ingest, snapshots, store
from corrlib.build import run_build
from corrlib.ingest import IngestRejected, run_ingest

W = 5
BLOCK = 3
TICKERS = [f"S{k:02d}" for k in range(12)]
#: Business days; the base zip skips GAP (for the insert test) and stops before N_BASE + 1.
CALENDAR: list[date] = list(pd.bdate_range("2022-01-03", periods=48).date)
GAP = 20
N_BASE = 41
BASE_DATES: list[date] = CALENDAR[:GAP] + CALENDAR[GAP + 1 : N_BASE]
APPEND_DATE = CALENDAR[N_BASE]
BACKFILL_DATE = date(2021, 12, 30)

#: Files that a full rebuild compares byte for byte (inputs.json and manifest.json legitimately differ).
DATA_FILES = (
    "returns.npy",
    "tickers.json",
    "dates.json",
    "prices.parquet",
    "daily_stats.parquet",
    "hist.npy",
    "hist_edges.npy",
    "anomalies.json",
    "issues_by_date.parquet",
)


def make_prices(n_dates: int, n_tickers: int, seed: int = 11) -> np.ndarray:
    """One-factor price paths (betas from -1 to 1.5, so correlations of both signs), day 0 a uniform 20.0 seed."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.012, n_dates)
    noise = rng.normal(0.0, 0.010, (n_dates, n_tickers))
    steps = market[:, None] * np.linspace(-1.0, 1.5, n_tickers) + noise
    steps[0] = 0.0
    return np.round(20.0 * np.exp(np.cumsum(steps, axis=0)), 6)


PRICES = make_prices(len(CALENDAR), len(TICKERS))
PRICE_OF = {CALENDAR[t]: PRICES[t] for t in range(len(CALENDAR))}


def text_for(day: date, prices: np.ndarray | None = None, tickers: list[str] = TICKERS) -> str:
    return daily_csv_text(day, tickers, PRICE_OF[day] if prices is None else prices)


def write_csv(directory: Path, day: date, text: str, name: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{day.strftime('%Y%m%d')}.csv")
    path.write_bytes(text.encode("utf-8"))
    return path


def base_members() -> dict[date, str]:
    return {d: text_for(d) for d in BASE_DATES}


def build_root(path: Path, members: dict[date, str], overrides: Path | None = None) -> Path:
    """A store root with one full_build snapshot of ``members`` (date -> CSV text)."""
    zip_path = write_zip(path.parent / f"{path.name}.zip", {f"d/{d.strftime('%Y%m%d')}.csv": members[d] for d in sorted(members)})
    run_build(zip_path, path, window=W, block=BLOCK, overrides=overrides)
    return path


def current_dir(root: Path) -> Path:
    return store.snapshot_path(root)


def assert_same_data(a: Path, b: Path) -> None:
    """The data artifacts of two roots' CURRENT snapshots are byte-identical."""
    da, db = current_dir(a), current_dir(b)
    for name in DATA_FILES:
        assert (da / name).read_bytes() == (db / name).read_bytes(), name


def stats(root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(dates, n_valid, [mean, std] as uint64 bit patterns [T x 2], hist) of the CURRENT snapshot."""
    s = store.load_store(root)
    bits = np.stack([s.mean_corr, s.std_corr], axis=1).view(np.uint64)
    return s.dates, s.n_valid, bits, s.hist


@pytest.fixture(scope="module")
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_root(tmp_path_factory.mktemp("ingest_base") / "root", base_members())


@pytest.fixture
def root(base_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "root"
    shutil.copytree(base_root, copy)
    return copy


# --- identity: duplicates ------------------------------------------------------------------


def test_same_file_twice_is_a_no_op(root: Path, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    first = run_ingest(root, [path], reason="append")
    assert first.status == "published" and first.files[0].kind == "append"
    ids = snapshots.list_snapshot_ids(root)
    second = run_ingest(root, [path])
    assert second.status == "duplicate" and second.snapshot == first.snapshot
    assert second.files[0].kind == "duplicate"
    assert snapshots.list_snapshot_ids(root) == ids and snapshots.current_id(root) == first.snapshot
    assert store.load_store(root).dates.count(APPEND_DATE.isoformat()) == 1  # never appended twice


def test_file_identical_to_a_base_member_is_a_duplicate(root: Path, tmp_path: Path) -> None:
    day = BASE_DATES[10]
    path = write_csv(tmp_path / "in", day, text_for(day))
    before = snapshots.current_id(root)
    result = run_ingest(root, [path])
    assert result.status == "duplicate" and result.files[0].kind == "duplicate"
    assert "d/" in result.files[0].reason  # names the zip member it duplicates
    assert snapshots.current_id(root) == before and len(snapshots.list_snapshot_ids(root)) == 1
    assert not (root / "overlays").exists()
    assert snapshots.read_log(root)[-1]["status"] == "duplicate"


# --- append / correction / insert / new ticker == full rebuild -----------------------------------


def test_append_is_queryable_and_matches_a_full_rebuild(root: Path, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    result = run_ingest(root, [path], reason="next day")
    changes = result.changes
    assert changes["dates_added"] == [APPEND_DATE.isoformat()]
    assert changes["returns_changed_dates"] == [APPEND_DATE.isoformat()]
    assert changes["recomputed_window_dates"] == {"count": 1, "ranges": [[APPEND_DATE.isoformat()] * 2]}
    assert changes["copied_window_dates"] == len(BASE_DATES) - W
    assert changes["inputs"] == [
        {
            "date": APPEND_DATE.isoformat(),
            "kind": "append",
            "sha256": result.files[0].sha256,
            "file": f"overlays/{APPEND_DATE.strftime('%Y%m%d')}-{result.files[0].sha256[:12]}.csv",
            "original_name": path.name,
        }
    ]
    s = store.load_store(root)
    assert s.dates[-1] == APPEND_DATE.isoformat() and s.snapshot["kind"] == "update"
    assert s.snapshot["parent"] == result.parent and s.manifest["changes"]["reason"] == "next day"
    assert s.n_valid[-1] == len(TICKERS)
    assert (root / changes["inputs"][0]["file"]).read_bytes() == path.read_bytes()
    inputs = store.read_inputs(root)
    assert inputs["date_sources"][APPEND_DATE.isoformat()]["source"] == "overlay"
    assert list(inputs["date_sources"]) == s.dates
    rebuilt = build_root(tmp_path / "rebuilt", {**base_members(), APPEND_DATE: text_for(APPEND_DATE)})
    assert_same_data(root, rebuilt)
    assert snapshots.verify_snapshot(current_dir(root)) == []


def test_single_price_correction_recomputes_exactly_w_plus_1_window_ends(root: Path, tmp_path: Path) -> None:
    p = 25
    day = BASE_DATES[p]
    corrected = PRICE_OF[day].copy()
    corrected[3] *= 1.1
    before_dates, before_n, before_bits, before_hist = stats(root)
    result = run_ingest(root, [write_csv(tmp_path / "in", day, text_for(day, corrected))], reason="vendor fix")
    changes = result.changes
    iso = [d.isoformat() for d in BASE_DATES]
    assert result.files[0].kind == "correction"
    assert changes["dates_corrected"] == [iso[p]] and changes["dates_added"] == []
    assert changes["returns_changed_dates"] == [iso[p], iso[p + 1]]
    assert changes["recomputed_window_dates"] == {"count": W + 1, "ranges": [[iso[p], iso[p + W]]]}

    after_dates, after_n, after_bits, after_hist = stats(root)
    assert after_dates == before_dates
    ends = np.zeros(len(BASE_DATES), dtype=bool)
    ends[p : p + W + 1] = True
    stats_changed = (after_bits != before_bits).any(axis=1) | (after_n != before_n)
    hist_changed = (after_hist != before_hist).any(axis=1)
    assert not (stats_changed | hist_changed)[~ends].any()  # byte-identical elsewhere
    assert stats_changed[ends].all()  # every window holding the corrected price moved
    np.testing.assert_array_equal(after_hist[~ends], before_hist[~ends])
    assert store.read_inputs(root)["date_sources"][iso[p]]["source"] == "overlay"

    rebuilt = build_root(tmp_path / "rebuilt", {**base_members(), day: text_for(day, corrected)})
    assert_same_data(root, rebuilt)


def test_insert_and_backfill_match_a_full_rebuild(root: Path, tmp_path: Path) -> None:
    backfill = PRICE_OF[CALENDAR[1]] * 0.97
    files = [
        write_csv(tmp_path / "in", CALENDAR[GAP], text_for(CALENDAR[GAP])),
        write_csv(tmp_path / "in", BACKFILL_DATE, text_for(BACKFILL_DATE, backfill)),
    ]
    result = run_ingest(root, files, reason="late files")
    assert [f.kind for f in result.files] == ["insert", "insert"]
    assert result.changes["dates_added"] == [BACKFILL_DATE.isoformat(), CALENDAR[GAP].isoformat()]
    s = store.load_store(root)
    assert s.dates[0] == BACKFILL_DATE.isoformat() and CALENDAR[GAP].isoformat() in s.dates
    # The 20.0 seed day is no longer the first date: re-classified from its neighbours (still a re-base).
    [seed] = s.anomalies
    assert (seed["date"], seed["index"], seed["classification"], seed["look_ahead"]) == (BASE_DATES[0].isoformat(), 1, "rebase", True)
    assert result.changes["anomaly_changes"] == []
    rebuilt = build_root(
        tmp_path / "rebuilt",
        {**base_members(), CALENDAR[GAP]: text_for(CALENDAR[GAP]), BACKFILL_DATE: text_for(BACKFILL_DATE, backfill)},
    )
    assert_same_data(root, rebuilt)


def test_new_ticker_in_an_appended_day(root: Path, tmp_path: Path) -> None:
    tickers = [*TICKERS, "NEW1"]
    prices = np.append(PRICE_OF[APPEND_DATE], 42.0)
    text = text_for(APPEND_DATE, prices, tickers)
    result = run_ingest(root, [write_csv(tmp_path / "in", APPEND_DATE, text)])
    assert result.changes["tickers_added"] == ["NEW1"]
    s = store.load_store(root)
    assert s.tickers == sorted(tickers) and s.n_tickers == len(TICKERS) + 1
    raw, _ = store.read_prices(root)
    j = s.ticker_index["NEW1"]
    assert np.isnan(raw[:-1, j]).all() and raw[-1, j] == 42.0  # no invented history
    assert np.isnan(np.asarray(s.returns)[:, j]).all()
    rebuilt = build_root(tmp_path / "rebuilt", {**base_members(), APPEND_DATE: text})
    assert_same_data(root, rebuilt)


def test_incremental_equals_full_recompute(root: Path, tmp_path: Path) -> None:
    day = BASE_DATES[30]
    corrected = PRICE_OF[day].copy()
    corrected[[0, 7]] = np.nan  # two tickers missing that day
    files = [
        write_csv(tmp_path / "in", day, text_for(day, corrected)),
        write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE)),
    ]
    base_id = snapshots.current_id(root)
    incremental = run_ingest(root, files)
    snapshots.rollback(root, base_id)
    full = run_ingest(root, files, full=True)
    assert incremental.changes["mode"] == "incremental" and full.changes["mode"] == "full"
    assert full.changes["recomputed_window_dates"]["count"] == len(BASE_DATES) + 1 - W
    assert incremental.changes["recomputed_window_dates"]["count"] == W + 1 + 1  # p..p+W and the new day
    assert incremental.changes["affected_window_dates"] == full.changes["affected_window_dates"]
    a = snapshots.snapshots_path(root) / incremental.snapshot
    b = snapshots.snapshots_path(root) / full.snapshot
    for name in (*DATA_FILES, "inputs.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # Same content -> same hash part; ids differ by seq (v0003 after the rollback, not a second v0002).
    assert incremental.snapshot.split("-")[1] == full.snapshot.split("-")[1]
    assert full.snapshot.startswith("v0003-") and store.load_store(root).snapshot["parent"] == base_id


# --- anomaly re-classification and overrides -------------------------------------------------------


@pytest.fixture(scope="module")
def unresolved_members() -> dict[date, str]:
    """30 days whose last one is uniform (every price 7.0): unresolved until the next day arrives."""
    members = {d: text_for(d) for d in CALENDAR[:30]}
    members[CALENDAR[29]] = text_for(CALENDAR[29], np.full(len(TICKERS), 7.0))
    return members


@pytest.mark.parametrize("continues_from", ["previous", "uniform"])
def test_next_day_resolves_an_unresolved_last_date(
    continues_from: str, unresolved_members: dict[date, str], tmp_path: Path
) -> None:
    root = build_root(tmp_path / "root", unresolved_members)
    last = CALENDAR[29].isoformat()
    [record] = [r for r in store.load_store(root).anomalies if r["date"] == last]
    assert (record["classification"], record["status"], record["action"]) == ("unresolved", "unresolved", "excluded_pending_next_day")

    nxt = CALENDAR[30]
    prices = PRICE_OF[CALENDAR[28]] * 1.003 if continues_from == "previous" else np.full(len(TICKERS), 7.0) * PRICE_OF[nxt] / PRICE_OF[CALENDAR[29]]
    text = text_for(nxt, np.round(prices, 6))
    result = run_ingest(root, [write_csv(tmp_path / "in", nxt, text)])
    expected = "placeholder" if continues_from == "previous" else "rebase"
    assert result.changes["anomaly_changes"] == [
        {"date": last, "from": {"classification": "unresolved", "status": "unresolved"}, "to": {"classification": expected, "status": "inferred"}}
    ]
    [record] = [r for r in store.load_store(root).anomalies if r["date"] == last]
    assert record["look_ahead"] is True and record["observation_dates_used"][-1] == nxt.isoformat()
    # The parent snapshot keeps its interpretation: nothing published is rewritten.
    parent_records = store.read_anomalies(snapshots.snapshots_path(root) / result.parent)
    assert [r["classification"] for r in parent_records if r["date"] == last] == ["unresolved"]
    rebuilt = build_root(tmp_path / "rebuilt", {**unresolved_members, nxt: text})
    assert_same_data(root, rebuilt)


def overrides_file(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"overrides": entries}, indent=2) + "\n", encoding="utf-8")
    return path


def test_overrides_alone_publish_only_when_records_change(unresolved_members: dict[date, str], tmp_path: Path) -> None:
    root = build_root(tmp_path / "root", unresolved_members)
    last = CALENDAR[29].isoformat()
    path = overrides_file(
        tmp_path / "o.json", [{"date": last, "classification": "placeholder", "reason": "vendor confirmed", "author": "qa"}]
    )
    confirmed = run_ingest(root, [], overrides=path, reason="confirm the 7.0 day")
    assert confirmed.status == "published"
    assert confirmed.changes["anomaly_changes"] == [
        {"date": last, "from": {"classification": "unresolved", "status": "unresolved"}, "to": {"classification": "placeholder", "status": "confirmed"}}
    ]
    assert confirmed.changes["returns_changed_dates"] == [] and confirmed.changes["recomputed_window_dates"]["count"] == 0
    assert confirmed.changes["overrides"]["changed"] is True
    assert (current_dir(root) / "overrides.json").read_bytes() == path.read_bytes()

    again = run_ingest(root, [], overrides=path)
    assert again.status == "duplicate" and snapshots.current_id(root) == confirmed.snapshot

    reformatted = tmp_path / "o2.json"
    reformatted.write_text(json.dumps(json.loads(path.read_text())), encoding="utf-8")
    same_effect = run_ingest(root, [], overrides=reformatted)
    assert same_effect.status == "duplicate" and "do not change" in same_effect.record["reason"]

    # Without --overrides, the confirmed decision persists through the next update, even though
    # the rule alone would now say "rebase" (the next day continues from 7.0).
    nxt = CALENDAR[30]
    text = text_for(nxt, np.round(7.0 * (1.0 + 0.01 * np.linspace(-1.0, 1.0, len(TICKERS))), 6))
    result = run_ingest(root, [write_csv(tmp_path / "in", nxt, text)])
    [record] = [r for r in store.load_store(root).anomalies if r["date"] == last]
    assert (record["classification"], record["status"], record["evidence"]["rule_classification"]) == ("placeholder", "confirmed", "rebase")
    assert result.changes["anomaly_changes"] == []
    assert store.read_inputs(root)["overrides_sha256"] == confirmed.changes["overrides"]["sha256"]


def test_reclassifying_override_recomputes_and_matches_a_full_build_with_it(tmp_path: Path) -> None:
    members = {d: text_for(d) for d in CALENDAR[:30]}
    members[CALENDAR[15]] = text_for(CALENDAR[15], np.full(len(TICKERS), 1.0))  # a placeholder by the rule
    root = build_root(tmp_path / "root", members)
    day = CALENDAR[15].isoformat()
    path = overrides_file(tmp_path / "o.json", [{"date": day, "classification": "rebase", "reason": "treat as re-base", "author": "qa"}])
    result = run_ingest(root, [], overrides=path)
    assert result.changes["anomaly_changes"] == [
        {"date": day, "from": {"classification": "placeholder", "status": "inferred"}, "to": {"classification": "rebase", "status": "confirmed"}}
    ]
    assert result.changes["returns_changed_dates"] == [CALENDAR[16].isoformat()]
    assert result.changes["recomputed_window_dates"] == {"count": W, "ranges": [[CALENDAR[16].isoformat(), CALENDAR[20].isoformat()]]}
    rebuilt = build_root(tmp_path / "rebuilt", members, overrides=path)
    assert_same_data(root, rebuilt)


def test_override_of_a_non_uniform_day_is_rejected(root: Path, tmp_path: Path) -> None:
    path = overrides_file(tmp_path / "o.json", [{"date": BASE_DATES[9].isoformat(), "classification": "rebase", "reason": "r", "author": "a"}])
    before = snapshots.current_id(root)
    with pytest.raises(IngestRejected, match="not a detected uniform-price day"):
        run_ingest(root, [], overrides=path)
    assert snapshots.current_id(root) == before
    assert snapshots.read_log(root)[-1]["status"] == "rejected"


# --- validation (all-or-nothing) -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("bad_name", "YYYYMMDD.csv"),
        ("bad_calendar_date", "YYYYMMDD.csv"),
        ("bad_header", "unexpected header"),
        ("empty_file", "cannot parse CSV"),
        ("header_only", "no data rows"),
        ("no_valid_price", "no valid price"),
        ("conflicting_files", "two different files"),
    ],
)
def test_invalid_input_rejects_the_whole_invocation(case: str, match: str, root: Path, tmp_path: Path) -> None:
    good = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    bad_dir = tmp_path / "bad"
    day = CALENDAR[N_BASE + 1]
    if case == "bad_name":
        bad = write_csv(bad_dir, day, text_for(day), name="2022-03-02.csv")
    elif case == "bad_calendar_date":
        bad = write_csv(bad_dir, day, text_for(day), name="20220230.csv")
    elif case == "bad_header":
        bad = write_csv(bad_dir, day, text_for(day).replace("Ticker,Date,Price", "Symbol,Day,Close"))
    elif case == "empty_file":
        bad = write_csv(bad_dir, day, "")
    elif case == "header_only":
        bad = write_csv(bad_dir, day, "Ticker,Date,Price\n")
    elif case == "no_valid_price":
        bad = write_csv(bad_dir, day, f"Ticker,Date,Price\nS00,{day},abc\nS01,{day},\nS02,{day},-3\n")
    else:
        bad = write_csv(bad_dir, APPEND_DATE, text_for(APPEND_DATE, PRICE_OF[APPEND_DATE] * 1.01))
    before = (snapshots.current_id(root), snapshots.list_snapshot_ids(root))
    with pytest.raises(IngestRejected, match=match):
        run_ingest(root, [good, bad])
    assert (snapshots.current_id(root), snapshots.list_snapshot_ids(root)) == before
    assert not (root / "overlays").exists() and snapshots.staging_dirs(root) == []
    record = snapshots.read_log(root)[-1]
    assert record["status"] == "rejected" and match in record["reason"]
    assert {f["name"] for f in record["files"]} == {good.name, bad.name}
    assert ingest.main(["--root", str(root), str(good), str(bad)]) == 2


# --- failures, retries and the log ------------------------------------------------------------------


@pytest.mark.parametrize("where", ["aggregates", "pointer_swap"])
def test_failed_ingest_leaves_current_and_a_retry_succeeds(
    where: str, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    before = (snapshots.current_id(root), snapshots.list_snapshot_ids(root))

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full (simulated)")

    if where == "aggregates":
        monkeypatch.setattr(ingest, "daily_aggregates", boom)
    else:
        monkeypatch.setattr(snapshots, "write_current", boom)
    with pytest.raises(OSError, match="simulated"):
        run_ingest(root, [path])
    assert (snapshots.current_id(root), snapshots.list_snapshot_ids(root)) == before
    assert snapshots.staging_dirs(root) == [] and not (root / ".lock").exists()
    assert store.load_store(root).snapshot_id == before[0]
    failed = snapshots.read_log(root)[-1]
    assert failed["status"] == "failed" and "disk full" in failed["reason"] and failed["parent"] == before[0]

    monkeypatch.undo()
    retry = run_ingest(root, [path])
    assert retry.status == "published" and snapshots.current_id(root) == retry.snapshot
    assert run_ingest(root, [path]).status == "duplicate"  # re-running once more is harmless


def test_ingest_log_records_every_attempt(root: Path, tmp_path: Path) -> None:
    good = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    bad = write_csv(tmp_path / "bad", CALENDAR[N_BASE + 1], "Ticker,Date,Price\n")
    published = run_ingest(root, [good], reason="r1")
    run_ingest(root, [good])
    with pytest.raises(IngestRejected):
        run_ingest(root, [bad])
    with snapshots.locked(root, "someone else"), pytest.raises(snapshots.LockError):
        run_ingest(root, [good])
    records = snapshots.read_log(root)
    assert [(r["operation"], r["status"]) for r in records] == [
        ("build", "published"),
        ("ingest", "published"),
        ("ingest", "duplicate"),
        ("ingest", "rejected"),
        ("ingest", "failed"),
    ]
    assert len({r["attempt_id"] for r in records}) == len(records)
    assert records[1]["snapshot"] == published.snapshot and records[1]["update_reason"] == "r1"
    assert records[1]["files"] == [{"name": good.name, "date": APPEND_DATE.isoformat(), "sha256": published.files[0].sha256, "kind": "append"}]
    assert records[1]["summary"]["dates_added"] == 1
    assert records[2]["files"][0]["kind"] == "duplicate" and records[2]["snapshot"] == published.snapshot
    assert "LockError" in records[4]["reason"] and "someone else" in records[4]["reason"]
    for r in records[1:]:  # records[0] was logged where the base root was built (it was copied)
        assert r["root"] == str(root.resolve()) and r["at"].endswith("Z")


def test_ingest_requires_a_published_snapshot(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    with pytest.raises(snapshots.SnapshotError, match="no published snapshot"):
        run_ingest(tmp_path / "empty", [path])
    assert ingest.main(["--root", str(tmp_path / "empty2"), str(path)]) == 2


def test_cli_publishes_and_reports(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(tmp_path / "in", APPEND_DATE, text_for(APPEND_DATE))
    assert ingest.main(["--root", str(root), str(path), "--reason", "cli"]) == 0
    out = capsys.readouterr().out
    assert "Published v0002-" in out and "windows recomputed 1" in out
    assert ingest.main(["--root", str(root), str(path)]) == 0
    assert "Nothing published" in capsys.readouterr().out
    with snapshots.locked(root, "test"):
        assert ingest.main(["--root", str(root), str(path)]) == 3


# --- the demo update path (make demo-update) ---------------------------------------------------------


def test_demo_append_and_correction_match_a_full_rebuild(demo_files, demo_store_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "store"
    shutil.copytree(demo_store_dir, root)
    appended = run_ingest(root, [demo_files.append_csv], reason="demo: append")
    corrected = run_ingest(root, [demo_files.correction_csv], reason="demo: correction")
    window = store.load_store(root).window
    assert appended.changes["recomputed_window_dates"]["count"] == 1
    assert corrected.changes["recomputed_window_dates"]["count"] == window + 1 == 21
    assert corrected.changes["returns_changed_dates"] == [
        demo_files.correction_date.isoformat(),
        demo_files.dates[demo_files.dates.index(demo_files.correction_date) + 1].isoformat(),
    ]
    raw, _ = store.read_prices(root)
    s = store.load_store(root)
    t, j = s.date_index[demo_files.correction_date.isoformat()], s.ticker_index[demo_files.correction_ticker]
    assert raw[t, j] == demo_files.correct_price

    # Full rebuild from a zip holding the base members with the two overlay files applied.
    import zipfile

    rebuilt_zip = tmp_path / "rebuilt.zip"
    with zipfile.ZipFile(demo_files.zip_path) as src, zipfile.ZipFile(rebuilt_zip, "w") as dst:
        dst.comment = src.comment
        for info in src.infolist():
            if not info.filename.endswith(demo_files.correction_csv.name):
                dst.writestr(info.filename, src.read(info.filename))
        dst.writestr(f"demo_stock_data/{demo_files.correction_csv.name}", demo_files.correction_csv.read_bytes())
        dst.writestr(f"demo_stock_data/{demo_files.append_csv.name}", demo_files.append_csv.read_bytes())
    rebuilt = tmp_path / "rebuilt"
    run_build(rebuilt_zip, rebuilt)
    assert_same_data(root, rebuilt)
    issues = pq.read_table(current_dir(root) / "issues_by_date.parquet")
    assert issues.num_rows == len(demo_files.dates) + 1
