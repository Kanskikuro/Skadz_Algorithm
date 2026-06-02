from pathlib import Path
from math import log

import numpy as np
import pandas as pd
from typing import cast

from scripts.config import MATCHUPS_FILE, REDUCED_MATCHUPS_FILE, SHRUNK_MATCHUPS_FILE


REQUIRED_COLUMNS = [
    "champ1",
    "role1",
    "type",
    "champ2",
    "role2",
    "win_rate",
    "delta",
    "sample_size",
]

VALID_TYPES = {"Synergy", "Counter"}


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "matchups.csv is missing required columns: " + ", ".join(missing)
        )


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize champion names, roles, and type values for stable grouping/lookups.
    """
    df = df.copy()

    for column in ["champ1", "champ2", "role1", "role2", "type"]:
        df[column] = df[column].astype(str).str.strip()

    df["role1"] = df["role1"].str.lower()
    df["role2"] = df["role2"].str.lower()

    df["type"] = (
        df["type"]
        .str.lower()
        .map({"synergy": "Synergy", "counter": "Counter"})
        .fillna(df["type"])
    )

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns safely and remove unusable rows.
    """
    df = df.copy()

    for column in ["win_rate", "delta", "sample_size"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)

    df = df.dropna(subset=["win_rate", "delta", "sample_size"])
    df = df[df["sample_size"] > 0]
    df = df[(df["win_rate"] > 0) & (df["win_rate"] < 100)]

    invalid_types = sorted(set(df["type"]) - VALID_TYPES)
    if invalid_types:
        print(f"Removed rows with invalid type values: {invalid_types}")
        df = df[df["type"].isin(VALID_TYPES)]

    after = len(df)

    if before != after:
        print(f"Removed {before - after} invalid rows.")

    return df


def canonicalize_row(row: pd.Series) -> tuple[str, str, str, str, str]:
    """
    Synergy is symmetric:
        A top + B jungle == B jungle + A top

    Counter is directional:
        A counters B != B counters A
    """
    champ1 = str(row["champ1"])
    role1 = str(row["role1"])
    row_type = str(row["type"])
    champ2 = str(row["champ2"])
    role2 = str(row["role2"])

    if row_type == "Synergy":
        pair1 = (champ1, role1)
        pair2 = (champ2, role2)
        sorted_pairs = sorted([pair1, pair2])

        return (
            sorted_pairs[0][0],
            sorted_pairs[0][1],
            "Synergy",
            sorted_pairs[1][0],
            sorted_pairs[1][1],
        )

    return champ1, role1, row_type, champ2, role2


def reduce_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate repeated rows and merge symmetric synergy rows.

    Aggregation:
    - win_rate: sample-size weighted average
    - delta: sample-size weighted average
    - sample_size: summed

    This avoids pandas groupby.apply typing warnings in VS Code/Pylance.
    """
    df = df.copy()
    df["canonical"] = df.apply(canonicalize_row, axis=1)

    rows: list[dict[str, object]] = []

    for canonical_raw, group in df.groupby("canonical", sort=True):
        canonical = cast(tuple[str, str, str, str, str], canonical_raw)

        total_samples = float(group["sample_size"].sum())

        if total_samples > 0:
            win_rate = float(np.average(group["win_rate"], weights=group["sample_size"]))
            delta = float(np.average(group["delta"], weights=group["sample_size"]))
        else:
            win_rate = float(group["win_rate"].mean())
            delta = float(group["delta"].mean())

        champ1, role1, row_type, champ2, role2 = canonical

        rows.append(
            {
                "champ1": champ1,
                "role1": role1,
                "type": row_type,
                "champ2": champ2,
                "role2": role2,
                "win_rate": win_rate,
                "delta": delta,
                "sample_size": int(total_samples),
            }
        )

    return pd.DataFrame(rows)


def shrink_win_rate(
    df: pd.DataFrame,
    global_win_rate: float = 50.0,
    m: int = 200,
) -> pd.DataFrame:
    """
    Bayesian shrinkage for raw win rate.

    Small sample sizes are pulled more strongly toward global_win_rate.
    """
    df = df.copy()

    df["win_rate_shrunk_bayes"] = (
        (df["win_rate"] * df["sample_size"] + global_win_rate * m)
        / (df["sample_size"] + m)
    )

    return df


def shrink_delta(
    df: pd.DataFrame,
    global_delta: float = 0.0,
    m: int = 200,
) -> pd.DataFrame:
    """
    Bayesian shrinkage for delta.

    Small sample sizes are pulled toward 0.
    """
    df = df.copy()

    df["delta_shrunk_bayes"] = (
        (df["delta"] * df["sample_size"] + global_delta * m)
        / (df["sample_size"] + m)
    )

    return df


def win_rate_to_log_odds(win_rate: float) -> float:
    p = win_rate / 100.0
    epsilon = 1e-6
    p = min(max(p, epsilon), 1.0 - epsilon)

    return log(p / (1.0 - p))


def add_log_odds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_odds_bayes"] = df["win_rate_shrunk_bayes"].apply(win_rate_to_log_odds)
    return df


def process_matchups(
    input_file: str | Path = MATCHUPS_FILE,
    reduced_file: str | Path = REDUCED_MATCHUPS_FILE,
    output_file: str | Path = SHRUNK_MATCHUPS_FILE,
    m: int = 200,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    input_file = Path(input_file)
    reduced_file = Path(reduced_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing {input_file}. Run: uv run python -m scripts.download_dataset"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if reduced_file.exists() and not force_rebuild:
        print(f"Loading reduced data from {reduced_file}")
        df = pd.read_csv(reduced_file)
    else:
        print(f"Loading raw matchup data from {input_file}")
        df = pd.read_csv(input_file)

        validate_input(df)

        df = normalize_text_columns(df)
        df = clean_numeric_columns(df)

        print("Reducing duplicate/symmetric matchup rows...")
        df = reduce_symmetry(df)

        reduced_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(reduced_file, index=False)
        print(f"Saved reduced data to {reduced_file}")

    print("Applying Bayesian shrinkage...")
    df = shrink_win_rate(df, global_win_rate=50.0, m=m)
    df = shrink_delta(df, global_delta=0.0, m=m)
    df = add_log_odds(df)

    output_columns = [
        "champ1",
        "role1",
        "type",
        "champ2",
        "role2",
        "win_rate",
        "sample_size",
        "delta",
        "win_rate_shrunk_bayes",
        "log_odds_bayes",
        "delta_shrunk_bayes",
    ]

    missing_output_columns = [column for column in output_columns if column not in df.columns]
    if missing_output_columns:
        raise ValueError(
            "Processed dataframe is missing columns: "
            + ", ".join(missing_output_columns)
        )

    df = df[output_columns]
    df.to_csv(output_file, index=False)

    print(f"Saved processed data to {output_file}")
    print(f"Rows: {len(df)}")

    return df


def main() -> None:
    process_matchups(
        input_file=MATCHUPS_FILE,
        reduced_file=REDUCED_MATCHUPS_FILE,
        output_file=SHRUNK_MATCHUPS_FILE,
        m=200,
        force_rebuild=False,
    )


if __name__ == "__main__":
    main()
