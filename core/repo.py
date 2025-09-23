
import pandas as pd
###############################################################################
# Load synergy/counter data
###############################################################################
def load_matchup_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class MatchupRepository:
    def __init__(self, df: pd.DataFrame): 
        self.df = df
    @classmethod
    def from_csv(cls, path: str): return cls(pd.read_csv(path))
    def get_df(self) -> pd.DataFrame: return self.df # Temporary



###############################################################################
# Load champion priors (and full champion list) for Bayesian role-guessing
###############################################################################
def load_champion_priors(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class PriorsRepository:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    @classmethod
    def from_csv(cls, path: str): return cls(pd.read_csv(path))
    def get_df(self) -> pd.DataFrame: return self.df # Temporary