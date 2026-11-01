"""A dataset = ``cases.csv`` (one row per case) + ``metadata.json`` + ``cases/<case_id>/``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, default_metadata, validate_frame, validate_row


class Dataset:
    def __init__(self, root: str | Path, metadata: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv = self.root / "cases.csv"
        self.meta_path = self.root / "metadata.json"
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps(default_metadata(metadata), indent=2))
        elif metadata:
            meta = json.loads(self.meta_path.read_text())
            meta.update(metadata)
            self.meta_path.write_text(json.dumps(meta, indent=2))

    @property
    def cases_dir(self) -> Path:
        d = self.root / "cases"
        d.mkdir(exist_ok=True)
        return d

    def metadata(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def append(self, row: dict) -> None:
        validate_row(row)
        cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in row] + sorted(
            c for c in row if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS
        )
        df_new = pd.DataFrame([{c: row.get(c) for c in cols}])
        if self.csv.exists():
            df = pd.concat([pd.read_csv(self.csv), df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(self.csv, index=False)

    def extend(self, rows: list[dict]) -> None:
        for r in rows:
            self.append(r)

    def load(self, source: str | None = None, only_usable: bool = False) -> pd.DataFrame:
        if not self.csv.exists():
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        df = pd.read_csv(self.csv)
        validate_frame(df)
        if source:
            df = df[df["source"] == source]
        if only_usable:
            df = usable(df)
        return df.reset_index(drop=True)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows fit for training: converged and with finite coefficients."""
    conv = df["converged"].astype(str).str.lower().isin(["true", "1", "1.0"])
    finite = df[["cl", "cd", "cm"]].notna().all(axis=1)
    return df[conv & finite]
