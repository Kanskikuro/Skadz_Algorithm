
import pandas as pd
###############################################################################
# Synergy/counter data
###############################################################################
class MatchupRepository:
    def __init__(self, df: pd.DataFrame): 
        self.df = df
    @classmethod
    def from_csv(cls, path: str): return cls(pd.read_csv(path))
    def get_df(self) -> pd.DataFrame: return self.df # Temporary
    def indexed(self) -> pd.DataFrame:
        # Create a multi-index based on columns commonly used in lookups.
        df = self.df.copy()
        df_indexed = df.set_index(['champ1', 'role1', 'type', 'champ2', 'role2'])
        return df_indexed.sort_index()  # Sort the index for optimal performance
        



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