"""Parse ``ShipsEar_meta.xlsx`` → ship-type labels + free-text descriptions.

This metadata is the project's differentiator vs. the four "rival" papers: none
of them use ShipsEar's native textual descriptions. The xlsx has one row per
recording with a ``Type`` column (the class) and a ``Notes`` column (situational
free text). ``text_enrich`` (评测 A/B) will contrast three variants:
(a) no text, (b) synthesized type descriptions (à la Cao), (c) these real notes.

Note: the local ShipsEar folder layout (A/B/C) does not map cleanly onto the 12
types, so ShipsEar manifest generation is handled separately from DeepShip and
is not part of the first runnable deliverable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

META_COLUMNS = {
    "Id", "Name", "Type", "Localization", "Date", "Notes",
    "Wind", "Distance", "Duration",
}


def load_meta(xlsx_path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Load the ShipsEar metadata spreadsheet, normalising column names."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    missing = META_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"ShipsEar meta missing expected columns: {missing}")
    df["Type"] = df["Type"].astype(str).str.strip()
    return df


def type_value_counts(xlsx_path: str | Path) -> pd.Series:
    """Convenience: number of recordings per ship type."""
    return load_meta(xlsx_path)["Type"].value_counts(dropna=False)


def type_to_label_id(xlsx_path: str | Path) -> dict[str, int]:
    """Map each ship ``Type`` string to a stable integer label id (sorted, 0-based)."""
    types = sorted(load_meta(xlsx_path)["Type"].dropna().unique())
    return {t: i for i, t in enumerate(types)}


def real_text_for_type(meta_df: pd.DataFrame, type_name: str) -> str:
    """Concatenated real ``Notes`` for a ship type → the "real text" variant."""
    notes = meta_df.loc[meta_df["Type"] == type_name, "Notes"].dropna().astype(str)
    return " ".join(notes.str.strip()).strip()
