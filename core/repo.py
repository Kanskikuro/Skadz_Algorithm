
import pandas as pd

from core.score import win_rate_to_log_odds
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

    def recalculate_matchups(self, m_value) -> None:
        """
        Recompute Bayesian-shrunk win rates (and optionally Δ), re-index DataFrame,
        save to CSV, then update overall win rates and recommendations.
        """
        # Shrink win_rate toward 0.5*100% by m_value "pseudo-samples"
        self.df['win_rate_shrunk_bayes'] = (
            (self.df['win_rate'] * self.df['sample_size'] +
                50.0 * m_value) / (self.df['sample_size'] + m_value)
        )
        self.df['log_odds_bayes'] = self.df['win_rate_shrunk_bayes'].apply(win_rate_to_log_odds)

        if 'delta' in self.df.columns:
            self.df['delta_shrunk_bayes'] = (
                (self.df['delta'] * self.df['sample_size'] +
                    0.0 * m_value) /
                (self.df['sample_size'] + m_value)
            )

        # Save relevant columns
        desired_columns = [
            "champ1", "role1", "type", "champ2", "role2",
            "win_rate", "sample_size",
            "win_rate_shrunk_bayes", "log_odds_bayes",
            "win_rate_shrunk_advi", "log_odds_advi",
            "win_rate_shrunk_hierarchical", "log_odds_hierarchical",
        ]
        if 'delta_shrunk_bayes' in self.df.columns:
            desired_columns.append('delta_shrunk_bayes')
        if 'delta' in self.df.columns:
            desired_columns.append('delta')

        columns_to_save = [c for c in desired_columns if c in self.df.columns]
        self.df.to_csv("data/matchups_shrunk.csv", columns=columns_to_save, index=False)
        



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