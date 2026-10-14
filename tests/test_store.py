from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import EXPECTED_COUNTS, FIXTURE_WINDOW, FixtureData

from corrlib import store
from corrlib.build import ingest, prepare_build
from corrlib.clean import clean_frame
from corrlib.config import RETURNS_FILE, build_policy
from corrlib.io import iter_daily_frames

REAL_V1 = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_round_trip_preserves_shapes_dtypes_axes(built_dir: Path, fixture_data: FixtureData) -> None:
    s = store.load_store(built_dir)
    T, N = len(fixture_data.dates), len(fixture_data.tickers)
    assert isinstance(s.returns, np.memmap)
    assert s.returns.shape == (T, N) and s.returns.dtype == np.float32
    assert s.tickers == fixture_data.tickers
    assert s.dates == [d.isoformat() for d in fixture_data.dates]
    assert s.ticker_index["NA"] == fixture_data.tickers.index("NA")
    assert s.hist.shape == (T, 40) and s.hist.dtype == np.int64
    assert s.hist_edges.shape == (41,)
    assert s.n_valid.shape == s.mean_corr.shape == s.std_corr.shape == (T,)
    assert s.window == FIXTURE_WINDOW
    # Windows exist from index W = 20, but the fixture's defects leave fewer than
    # two valid stocks until A's zero price (day 7) leaves the window at t = 28.
    assert s.n_valid[FIXTURE_WINDOW] == 1  # only MSFT
    assert s.first_valid_index == 28

    expected = ingest(fixture_data.zip_path).returns.astype(np.float32)
    np.testing.assert_array_equal(np.asarray(s.returns), expected)


def test_v2_store_fields(built_dir: Path, built_snapshot_dir: Path) -> None:
    s = store.load_store(built_dir)
    assert s.format_version == 2 and s.root == built_dir and s.data_dir == built_snapshot_dir
    assert built_snapshot_dir.parent == built_dir / "snapshots"
    assert re.fullmatch(r"v0001-[0-9a-f]{12}", s.snapshot_id)
    assert s.snapshot_id == built_snapshot_dir.name == json.loads((built_dir / "CURRENT").read_text())["snapshot"]
    assert s.snapshot == {
        "id": s.snapshot_id,
        "seq": 1,
        "parent": None,
        "created_at": s.manifest["snapshot"]["created_at"],
        "kind": "full_build",
    }
    # The id's hash part is derived from the artifact checksums.
    lines = sorted(f"{name} {entry['sha256']}" for name, entry in s.manifest["artifacts"].items())
    assert s.snapshot_id[6:] == hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
    # The same snapshot loaded directly (pinned) is the same data, without a root.
    pinned = store.load_store(built_snapshot_dir)
    assert pinned.snapshot_id == s.snapshot_id and pinned.root is None
    assert s.view == "retrospective"
    assert s.policy == build_policy(FIXTURE_WINDOW)
    assert s.dataset == {"kind": "source", "label": "Source data: stock_data.zip"}
    assert [(r["date"], r["classification"], r["status"]) for r in s.anomalies] == [("2021-01-04", "rebase", "inferred")]


def test_v2_manifest(built_snapshot_dir: Path) -> None:
    manifest = json.loads((built_snapshot_dir / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["snapshot"]["kind"] == "full_build" and manifest["changes"] is None
    assert manifest["hist_block_rows"] == 4
    assert set(manifest["artifacts"]) == set(store.ARTIFACT_FILES)
    assert manifest["window"] == FIXTURE_WINDOW
    assert manifest["issue_counts"] == EXPECTED_COUNTS
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["view"] == "retrospective"
    assert manifest["policy"]["reason_codes"][3]["name"] == "constant_returns"
    assert manifest["dataset"]["kind"] == "source"
    assert manifest["anomalies_summary"]["count"] == 1
    assert manifest["structural"]["rebase_dates"] == ["2021-01-04"]
    assert manifest["structural"]["unresolved_dates"] == []
    assert manifest["overrides"] is None
    for key in ("corrlib_version", "built_at", "first_date", "last_date", "first_valid_date", "hist", "timings_seconds"):
        assert key in manifest


def test_daily_stats_and_hist_are_consistent(built_dir: Path) -> None:
    s = store.load_store(built_dir)
    early = np.arange(s.n_dates) < s.window
    assert (s.n_valid[early] == 0).all() and np.isnan(s.mean_corr[early]).all()
    late = s.n_valid >= 2
    assert np.isfinite(s.mean_corr[late]).all()
    pairs = s.n_valid.astype(np.int64) * (s.n_valid - 1) // 2
    np.testing.assert_array_equal(s.hist.sum(axis=1), pairs)
    # FLAT never has a valid window; the 5 others do once their defects leave the window.
    assert s.n_valid.max() == 5


def test_prices_parquet_keeps_raw_and_used(built_dir: Path, built_snapshot_dir: Path, fixture_data: FixtureData) -> None:
    table = pq.read_table(built_snapshot_dir / "prices.parquet")
    assert table.column_names == ["date", "ticker", "price_raw", "price"]
    T, N = fixture_data.expected_prices.shape
    assert table.num_rows == T * N
    prices = table.column("price").to_numpy(zero_copy_only=False).reshape(T, N)
    np.testing.assert_allclose(prices, fixture_data.expected_prices, rtol=1e-12, equal_nan=True)
    assert table.column("ticker").to_pylist()[:N] == fixture_data.tickers
    raw, used = store.read_prices(built_dir)
    np.testing.assert_array_equal(used, prices)
    np.testing.assert_array_equal(raw, prices)  # no placeholder in the fixture: nothing removed


def test_issues_by_date_round_trip(built_dir: Path, fixture_data: FixtureData) -> None:
    days, counts = store.read_issues_by_date(built_dir)
    assert days == fixture_data.dates
    expected = []
    for file_date, frame in iter_daily_frames(fixture_data.zip_path):
        expected.append(clean_frame(file_date, frame)[1].as_dict())
    assert [c.as_dict() for c in counts] == expected
    total = counts[0]
    for k in range(1, len(counts)):
        total = total + counts[k]
    assert total.as_dict() == EXPECTED_COUNTS


def test_inputs_record_every_member_hash(built_dir: Path, fixture_data: FixtureData) -> None:
    inputs = store.read_inputs(built_dir)
    zip_bytes = fixture_data.zip_path.read_bytes()
    assert inputs["base"]["file_name"] == "stock_data.zip"
    assert inputs["base"]["sha256"] == hashlib.sha256(zip_bytes).hexdigest()
    assert inputs["base"]["size_bytes"] == len(zip_bytes)
    assert len(inputs["base"]["skipped_members"]) == 3
    assert inputs["overlays"] == [] and inputs["overrides_sha256"] is None
    sources = inputs["date_sources"]
    assert list(sources) == [d.isoformat() for d in fixture_data.dates]
    with zipfile.ZipFile(fixture_data.zip_path) as zf:
        for d in fixture_data.dates:
            entry = sources[d.isoformat()]
            assert entry["source"] == "base"
            assert entry["file"] == f"stock_data/{d.strftime('%Y%m%d')}.csv"
            assert entry["sha256"] == hashlib.sha256(zf.read(entry["file"])).hexdigest()


def test_manifest_checksums_verify(built_dir: Path, built_snapshot_dir: Path) -> None:
    assert store.verify_manifest(built_dir) == []  # a root: its CURRENT snapshot
    assert store.verify_manifest(built_snapshot_dir) == []


def test_tampering_is_detected(built_snapshot_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(built_snapshot_dir, copy)
    data = bytearray((copy / RETURNS_FILE).read_bytes())
    data[-1] ^= 0xFF
    (copy / RETURNS_FILE).write_bytes(bytes(data))
    assert store.verify_manifest(copy) == [f"{RETURNS_FILE}: checksum mismatch"]
    (copy / "anomalies.json").unlink()
    assert "anomalies.json: missing" in store.verify_manifest(copy)
    with pytest.raises(store.StoreError, match="missing artifacts"):
        store.load_store(copy)


def test_load_store_reports_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(store.StoreError, match="missing artifacts"):
        store.load_store(tmp_path)


# --- writer reuse, snapshot block and store roots -----------------------------------


@pytest.fixture(scope="module")
def prepared(fixture_data: FixtureData):
    return prepare_build(fixture_data.zip_path, window=FIXTURE_WINDOW, block=4)


def test_write_snapshot_into_any_empty_directory(prepared, tmp_path: Path, built_snapshot_dir: Path) -> None:
    seen: dict = {}

    def snapshot(artifacts: dict) -> dict:
        seen.update(artifacts)
        return {"id": "v0001-abcdefabcdef", "seq": 1, "parent": None, "created_at": "2026-01-01T00:00:00Z", "kind": "full_build"}

    target = tmp_path / "snap"
    target.mkdir()
    manifest = store.write_snapshot(target, prepared.content, prepared.extra, snapshot=snapshot)
    assert set(seen) == set(store.ARTIFACT_FILES)
    assert manifest["snapshot"]["id"] == "v0001-abcdefabcdef"
    assert store.verify_manifest(target) == []
    s = store.load_store(target)
    assert s.snapshot_id == "v0001-abcdefabcdef" and s.snapshot["seq"] == 1
    # Same content as the CLI build, byte for byte, for the arrays.
    for name in ("returns.npy", "hist.npy", "hist_edges.npy", "tickers.json", "dates.json", "anomalies.json"):
        assert (target / name).read_bytes() == (built_snapshot_dir / name).read_bytes(), name
    with pytest.raises(store.StoreError, match="not empty"):
        store.write_snapshot(target, prepared.content, prepared.extra)


def test_snapshot_content_check_rejects_bad_shapes(prepared, tmp_path: Path) -> None:
    content = prepared.content
    bad = store.SnapshotContent(**{**content.__dict__, "n_valid": content.n_valid[:-1]})
    with pytest.raises(store.StoreError, match="n_valid"):
        bad.check()
    unordered = store.SnapshotContent(**{**content.__dict__, "dates": content.dates[::-1]})
    with pytest.raises(store.StoreError, match="strictly increasing"):
        unordered.check()


def test_load_store_resolves_a_root_through_current(prepared, tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "snapshots" / "v0002-0123456789ab"
    target.mkdir(parents=True)
    store.write_snapshot(
        target,
        prepared.content,
        prepared.extra,
        snapshot=lambda artifacts: {"id": target.name, "seq": 2, "parent": "v0001-x", "created_at": "t", "kind": "update"},
    )
    (root / "CURRENT").write_text(json.dumps({"snapshot": target.name, "published_at": "2026-01-01T00:00:00Z"}))
    s = store.load_store(root)
    assert s.data_dir == target and s.root == root
    assert s.snapshot_id == target.name
    assert s.snapshot == {"id": target.name, "seq": 2, "parent": "v0001-x", "created_at": "t", "kind": "update"}

    (root / "CURRENT").write_text(json.dumps({"snapshot": "v0009-missing"}))
    with pytest.raises(store.StoreError, match="no manifest"):
        store.load_store(root)
    (root / "CURRENT").write_text(json.dumps({"snapshot": "../evil"}))
    with pytest.raises(store.StoreError, match="does not name"):
        store.load_store(root)
    (root / "CURRENT").write_text("{broken")
    with pytest.raises(store.StoreError, match="unreadable"):
        store.load_store(root)


# --- legacy format 1 ----------------------------------------------------------------


def make_v1_dir(built_dir: Path, target: Path) -> Path:
    """A format-1 directory as the previous build wrote it (flat layout, single price column)."""
    target.mkdir()
    for name in store.V1_ARTIFACT_FILES:
        if name != "prices.parquet":
            shutil.copy(built_dir / name, target / name)
    table = pq.read_table(built_dir / "prices.parquet").select(["date", "ticker", "price"])
    pq.write_table(table, target / "prices.parquet", compression="zstd")
    v2 = json.loads((built_dir / "manifest.json").read_text())
    keep = (
        "corrlib_version", "built_at", "source", "window", "validity_eps", "n_dates", "n_tickers",
        "first_date", "last_date", "first_valid_date", "hist", "issue_counts", "timings_seconds",
    )
    manifest = {"format_version": 1}
    for key in keep:
        manifest[key] = v2[key]
    manifest["structural"] = {
        "rebase_dates": v2["structural"]["rebase_dates"],
        "placeholder_dates": v2["structural"]["placeholder_dates"],
        "returns_nulled_at_rebase": 0,
        "prices_nulled_on_placeholder_days": 0,
    }
    manifest["artifacts"] = {}
    for name in store.V1_ARTIFACT_FILES:
        data = (target / name).read_bytes()
        manifest["artifacts"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def test_legacy_v1_directory_still_loads(built_dir: Path, built_snapshot_dir: Path, tmp_path: Path) -> None:
    legacy = make_v1_dir(built_snapshot_dir, tmp_path / "v1")
    s = store.load_store(legacy)
    v2 = store.load_store(built_dir)
    assert s.format_version == 1
    manifest_sha = hashlib.sha256((legacy / "manifest.json").read_bytes()).hexdigest()
    assert s.snapshot_id == "legacy-" + manifest_sha[:12]
    assert s.snapshot["kind"] == "legacy"
    assert s.policy == build_policy(FIXTURE_WINDOW) and s.view == "retrospective"
    assert s.dataset["kind"] == "source"
    np.testing.assert_array_equal(np.asarray(s.returns), np.asarray(v2.returns))
    np.testing.assert_array_equal(s.n_valid, v2.n_valid)
    [record] = s.anomalies
    assert (record["date"], record["classification"], record["action"]) == ("2021-01-04", "rebase", "none")
    assert record["evidence"]["median_abs_log_from_uniform"] is None and "legacy" in record["evidence"]["note"]
    assert store.verify_manifest(legacy) == []
    with pytest.raises(store.StoreError, match="price_raw"):
        store.read_prices(legacy)


def test_legacy_anomalies_reconstruct_actions_and_windows() -> None:
    dates = [f"2021-01-{d:02d}" for d in range(1, 31)]
    manifest = {"structural": {"rebase_dates": [dates[0], dates[20]], "placeholder_dates": [dates[10]]}}
    records = store.legacy_anomalies(manifest, dates, 5)
    assert [(r["index"], r["classification"], r["action"], r["look_ahead"]) for r in records] == [
        (0, "rebase", "none", False),
        (10, "placeholder", "prices_nulled", True),
        (20, "rebase", "return_into_day_nulled", True),
    ]
    assert records[1]["affected"]["window_end_dates"] == {"first": dates[10], "last": dates[15], "count": 6}
    assert records[2]["affected"]["window_end_dates"] == {"first": dates[20], "last": dates[24], "count": 5}


def test_current_wins_over_legacy_flat_artifacts_in_a_root(
    built_snapshot_dir: Path, fixture_data: FixtureData, tmp_path: Path
) -> None:
    """A legacy v1 directory later used as a build root: CURRENT is served, the flat files stay."""
    from corrlib.build import run_build

    root = make_v1_dir(built_snapshot_dir, tmp_path / "processed")
    legacy_manifest = (root / "manifest.json").read_bytes()
    assert store.load_store(root).format_version == 1
    manifest, _ = run_build(fixture_data.zip_path, root, window=FIXTURE_WINDOW, block=4)
    s = store.load_store(root)
    assert s.format_version == 2 and s.snapshot_id == manifest["snapshot"]["id"] and s.root == root
    assert (root / "manifest.json").read_bytes() == legacy_manifest  # left in place for running v1 readers
    assert store.read_manifest(root)["snapshot"]["id"] == s.snapshot_id
    assert store.read_prices(root)[0].shape == (s.n_dates, s.n_tickers)


def test_read_helpers_accept_root_or_snapshot(built_dir: Path, built_snapshot_dir: Path) -> None:
    assert store.read_anomalies(built_dir) == store.read_anomalies(built_snapshot_dir)
    assert store.read_inputs(built_dir) == store.read_inputs(built_snapshot_dir)
    assert store.read_manifest(built_dir) == store.read_manifest(built_snapshot_dir)
    assert store.snapshot_path(built_dir) == built_snapshot_dir == store.snapshot_path(built_snapshot_dir)


@pytest.mark.skipif(not (REAL_V1 / "manifest.json").is_file(), reason="real v1 artifacts not present")
def test_real_v1_artifacts_load_read_only() -> None:
    manifest = json.loads((REAL_V1 / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        pytest.skip("data/processed is not a format-1 directory")
    s = store.load_store(REAL_V1)
    assert s.format_version == 1 and s.snapshot_id.startswith("legacy-")
    assert (s.n_dates, s.n_tickers) == (900, 5000)
    assert [(r["date"], r["classification"]) for r in s.anomalies] == [
        ("2020-01-01", "rebase"),
        ("2020-12-31", "rebase"),
        ("2021-02-22", "placeholder"),
        ("2021-12-31", "rebase"),
        ("2023-01-02", "rebase"),
        ("2024-01-01", "rebase"),
    ]
    placeholder = s.anomalies[2]
    assert placeholder["affected"]["window_end_dates"]["count"] == 21


def test_write_issues_by_date_requires_one_row_per_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1 dates"):
        store.write_issues_by_date(tmp_path, [date(2021, 1, 4)], [])
