"""Streaming reads of the daily CSV files inside the source zip.

The zip is never extracted to disk. Each member is opened as a stream and
parsed into a DataFrame of *strings*, so every value can be validated in
:mod:`corrlib.clean` before it is interpreted. Reading everything as text with
``keep_default_na=False`` is what keeps the real ticker ``NA`` from turning
into a missing value.
"""

from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import EXPECTED_COLUMNS, MEMBER_PATTERN

log = logging.getLogger(__name__)

_MEMBER_RE = re.compile(MEMBER_PATTERN)


class SourceFormatError(ValueError):
    """The zip, or one of the CSV files in it, does not have the expected layout."""


@dataclass(frozen=True)
class MemberScan:
    """Result of scanning the zip directory.

    Attributes
    ----------
    members:
        ``(file_date, member_name)`` for every daily file, sorted by date.
    skipped:
        ``(member_name, reason)`` for every entry that was ignored.
    """

    members: list[tuple[date, str]]
    skipped: list[tuple[str, str]]


def scan_members(zip_path: str | Path) -> MemberScan:
    """Classify every entry of the zip as a daily file or a skipped entry.

    A daily file is any entry whose base name is ``YYYYMMDD.csv`` (with a real
    calendar date), optionally inside directories. Directories, ``__MACOSX/``
    metadata and anything else are skipped and logged.

    Raises
    ------
    SourceFormatError
        If two members map to the same date (the calendar would be ambiguous).
    """
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()

    members: list[tuple[date, str]] = []
    skipped: list[tuple[str, str]] = []
    seen: dict[date, str] = {}
    for k in range(len(infos)):
        info = infos[k]
        name = info.filename
        if info.is_dir():
            skipped.append((name, "directory"))
            continue
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
            skipped.append((name, "macOS metadata"))
            continue
        match = _MEMBER_RE.match(name)
        if match is None:
            skipped.append((name, "name is not YYYYMMDD.csv"))
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            skipped.append((name, "file name is not a valid date"))
            continue
        if file_date in seen:
            raise SourceFormatError(
                f"two zip members map to {file_date}: {seen[file_date]!r} and {name!r}"
            )
        seen[file_date] = name
        members.append((file_date, name))

    members.sort()
    for k in range(len(skipped)):
        log.info("skipping zip entry %r (%s)", skipped[k][0], skipped[k][1])
    return MemberScan(members=members, skipped=skipped)


def list_members(zip_path: str | Path) -> list[tuple[date, str]]:
    """Return ``(file_date, member_name)`` for each daily CSV, sorted by date.

    The date comes from the file name, which is authoritative for the calendar.
    Non-matching entries are skipped and logged (see :func:`scan_members`).
    """
    return scan_members(zip_path).members


def read_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Parse one zip member into a DataFrame of strings.

    Returns a frame with exactly the columns ``Ticker, Date, Price``. Every
    cell is text (empty string for an empty field). Nothing is coerced to NaN.

    Raises
    ------
    SourceFormatError
        If the file cannot be parsed or its header is not ``Ticker,Date,Price``
        (after trimming whitespace and a UTF-8 BOM).
    """
    with zf.open(name) as stream:
        try:
            df = pd.read_csv(
                stream,
                dtype=str,
                keep_default_na=False,
                na_values=[],
                encoding="utf-8-sig",
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise SourceFormatError(f"{name}: cannot parse CSV ({exc})") from exc

    columns = []
    for k in range(len(df.columns)):
        columns.append(str(df.columns[k]).strip().lstrip("﻿").strip())
    if tuple(columns) != EXPECTED_COLUMNS:
        raise SourceFormatError(
            f"{name}: unexpected header {','.join(columns)!r}; "
            f"expected {','.join(EXPECTED_COLUMNS)!r}"
        )
    df.columns = list(EXPECTED_COLUMNS)
    return df


def iter_daily_frames(
    zip_path: str | Path, members: list[tuple[date, str]] | None = None
) -> Iterator[tuple[date, pd.DataFrame]]:
    """Yield ``(file_date, raw_frame)`` for each daily file, in date order.

    Members are streamed from the open zip one at a time, so memory use is one
    daily file (~5,000 rows) regardless of the archive size. Frames are raw
    strings as returned by :func:`read_member`. ``members`` defaults to
    :func:`list_members`; pass it to avoid scanning the zip twice.
    """
    if members is None:
        members = list_members(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for k in range(len(members)):
            file_date, name = members[k]
            yield file_date, read_member(zf, name)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks (constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
