from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

SCORE_FIELDS = ("pos_scores", "neg_scores")


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def as_text_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return ["" if item is None else str(item) for item in value]


def as_score_list(value: Any, field_name: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [float(item) for item in value]


def validate_query_pos_neg_row(row: dict[str, Any]) -> None:
    if "query" not in row or row["query"] is None:
        raise ValueError("FlagEmbedding row must contain a non-empty query field")
    as_text_list(row.get("pos"), "pos")
    as_text_list(row.get("neg"), "neg")


def validate_flag_embedding_row(row: dict[str, Any]) -> None:
    validate_query_pos_neg_row(row)
    if "pos_scores" in row:
        as_score_list(row["pos_scores"], "pos_scores")
    if "neg_scores" in row:
        as_score_list(row["neg_scores"], "neg_scores")


def move_existing_scores(row: dict[str, Any], backup_prefix: str = "original_") -> None:
    if not backup_prefix:
        raise ValueError("backup_prefix must not be empty")
    for field_name in SCORE_FIELDS:
        if field_name not in row:
            continue
        backup_name = f"{backup_prefix}{field_name}"
        if backup_name not in row:
            row[backup_name] = row[field_name]


def write_jsonl_rows(path: str, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl_rows(path: str, skip_rows: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if emitted < skip_rows:
                emitted += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            yield line_number, row
            emitted += 1


def count_complete_jsonl_rows(path: str, truncate_invalid_tail: bool = False) -> int:
    if not os.path.exists(path):
        return 0

    count = 0
    last_good_offset = 0
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            current_offset = handle.tell()
            if not line.strip():
                last_good_offset = current_offset
                continue
            try:
                decoded = line.decode("utf-8")
                row = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(row, dict):
                break
            count += 1
            last_good_offset = current_offset

    if truncate_invalid_tail:
        size = os.path.getsize(path)
        if last_good_offset < size:
            with open(path, "ab") as handle:
                handle.truncate(last_good_offset)

    return count
