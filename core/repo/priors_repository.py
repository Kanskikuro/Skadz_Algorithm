import pandas as pd

###############################################################################
# Champion priors (and full champion list) for Bayesian role-guessing
###############################################################################
def load_champion_priors(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class PriorsRepository:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    @classmethod
    def from_csv(cls, path: str): return cls(pd.read_csv(path))
    def get_df(self) -> pd.DataFrame: return self.df # Temporary
    def champions(self) -> list[str]: return list(self.df["champion_name"].unique())