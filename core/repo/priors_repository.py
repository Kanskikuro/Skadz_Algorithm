import pandas as pd

from core.enums import ROLES


def load_champion_priors(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


class PriorsRepository:
    REQUIRED_COLUMNS = ["champion_name", *ROLES]

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._validate()

    @classmethod
    def from_csv(cls, path: str) -> "PriorsRepository":
        return cls(pd.read_csv(path))

    def get_df(self) -> pd.DataFrame:
        return self._df

    def champions(self) -> list[str]:
        return (
            self._df["champion_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    def _validate(self) -> None:
        missing = [
            column for column in self.REQUIRED_COLUMNS
            if column not in self._df.columns
        ]

        if missing:
            raise ValueError(f"Missing required priors columns: {missing}")

        self._df["champion_name"] = (
            self._df["champion_name"]
            .astype(str)
            .str.strip()
        )

        for role in ROLES:
            self._df[role] = pd.to_numeric(
                self._df[role],
                errors="coerce",
            ).fillna(0.0)

            self._df[role] = self._df[role].clip(lower=0.0)