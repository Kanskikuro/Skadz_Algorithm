import pandas as pd
import numpy as np

from core.score import calculate_overall_win_rates
from core.role_guess import guess_enemy_roles

IDX_COLS = ['champ1', 'role1', 'type', 'champ2', 'role2']

###############################################################################
# Synergy/counter data
###############################################################################
class MatchupRepository:
    def __init__(self, df: pd.DataFrame): 
        self.df = df

        self._idx_cache: pd.DataFrame | None = None
        self._idx_key: str | None = None
    @classmethod
    def from_csv(cls, path: str): return cls(pd.read_csv(path))
    def get_df(self) -> pd.DataFrame: return self.df # Temporary

    ## INTERNAL
    def _with_log_odds(self, method: str) -> pd.DataFrame:
        df = self.df
        method = (method or "bayesian").lower()
        pref = f"log_odds_{method}"

        if "log_odds" in df.columns:
            return df

        if pref in df.columns:
            return df.assign(log_odds=df[pref])

        for c in ("log_odds_bayes", "log_odds_advi", "log_odds_hierarchical"):
            if c in df.columns:
                return df.assign(log_odds=df[c])

        if "win_rate" in df.columns:
            p = df["win_rate"].clip(1e-6, 100 - 1e-6) / 100.0
            lo = np.log(p / (1 - p))
            return df.assign(log_odds=lo)

        raise ValueError("No source for log_odds")

    def _create_column(self, method: str = "Bayesian") -> None:
        method = method.lower()
        log_col = f'log_odds_{method}'
        if log_col in self.df.columns:
            self.df['log_odds'] = self.df[log_col]
        else:
            self.df['log_odds'] = self.df['log_odds_bayes']


    ## PUBLIC
    def indexed(self, method: str = "Bayesian") -> pd.DataFrame:
        key = method.lower()
        if self._idx_cache is not None and self._idx_key == key:
            return self._idx_cache
        df = self._with_log_odds(method)
        idx = df.set_index(IDX_COLS).sort_index()
        self._idx_cache, self._idx_key = idx, key
        return idx

    def recalculate_matchups(self, m_value: int) -> None:
        """Recompute shrunk win rates and log-odds in-place."""
        df = self.df
        df["win_rate_shrunk_bayes"] = (
            (df["win_rate"] * df["sample_size"] + 50.0 * m_value)
            / (df["sample_size"] + m_value)
        )
        df["log_odds_bayes"] = df["win_rate_shrunk_bayes"].pipe(
            lambda s: np.log(
                s.clip(1e-6, 100 - 1e-6).div(100.0).pipe(lambda p: p / (1 - p))
            )
        )
        if "delta" in df.columns:
            df["delta_shrunk_bayes"] = (
                (df["delta"] * df["sample_size"] + 0.0 * m_value)
                / (df["sample_size"] + m_value)
            )
        # invalidate cache
        self._idx_cache = None
        self._idx_key = None

    def save(self, path: str = "data/matchups_shrunk.csv") -> None:
        cols = [
            "champ1","role1","type","champ2","role2",
            "win_rate","sample_size",
            "win_rate_shrunk_bayes","log_odds_bayes",
            "win_rate_shrunk_advi","log_odds_advi",
            "win_rate_shrunk_hierarchical","log_odds_hierarchical",
            "delta","delta_shrunk_bayes","log_odds"  # ok if missing
        ]
        self.df.to_csv(path, columns=[c for c in cols if c in self.df.columns], index=False)
    
    def update_overall_win_rates(self, priors_repo, enemy_list, ally_team, method: str = "Bayesian" ) -> tuple[float, float]:
        """
        Recompute ally vs. enemy team win rates and update the label.
        """
        method = method.lower()

        self._create_column(method)

        enemy_team = guess_enemy_roles(enemy_list, priors_repo)

        ally_pct, enemy_pct = calculate_overall_win_rates(
            self.indexed(method), ally_team, enemy_team
        )

        return ally_pct, enemy_pct

        
