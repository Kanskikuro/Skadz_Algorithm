import pandas as pd
import numpy as np

from core.score import calculate_overall_win_rates
from core.role_guess import guess_enemy_roles


IDX_COLS = ["champ1", "role1", "type", "champ2", "role2"]


METHOD_TO_LOG_COL = {
    "bayesian": "log_odds_bayes",
    "bayes": "log_odds_bayes",
    "advi": "log_odds_advi",
    "hierarchical": "log_odds_hierarchical",
}


###############################################################################
# Synergy/counter data
###############################################################################

class MatchupRepository:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._normalize_columns()

        self._idx_cache: pd.DataFrame | None = None
        self._idx_key: str | None = None

    @classmethod
    def from_csv(cls, path: str) -> "MatchupRepository":
        return cls(pd.read_csv(path))

    def get_df(self) -> pd.DataFrame:
        return self._df

    def indexed(self, method: str = "Bayesian") -> pd.DataFrame:
        key = self._normalize_method(method)

        if self._idx_cache is not None and self._idx_key == key:
            return self._idx_cache

        df = self._with_log_odds(key)
        idx = df.set_index(IDX_COLS).sort_index()

        self._idx_cache = idx
        self._idx_key = key

        return idx

    def update_adjustments(self, method: str = "Bayesian") -> None:
        """
        Public replacement for _create_column().
        Creates/updates the active 'log_odds' column.
        """
        key = self._normalize_method(method)
        log_col = METHOD_TO_LOG_COL.get(key)

        if log_col and log_col in self._df.columns:
            self._df["log_odds"] = self._df[log_col]
        elif "win_rate" in self._df.columns:
            p = self._df["win_rate"].clip(1e-6, 100 - 1e-6) / 100.0
            self._df["log_odds"] = np.log(p / (1 - p))
        else:
            raise ValueError(
                f"No valid log-odds source for method '{method}'. "
                f"Expected one of: {list(METHOD_TO_LOG_COL.values())}"
            )

        self._invalidate_cache()

    def recalculate_matchups(self, m_value: int) -> None:
        """
        Recompute shrunk win rates and Bayesian log-odds in-place.
        """
        df = self._df

        required = ["win_rate", "sample_size"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required matchup columns: {missing}")

        df["win_rate_shrunk_bayes"] = (
            (df["win_rate"] * df["sample_size"] + 50.0 * m_value)
            / (df["sample_size"] + m_value)
        )

        p = df["win_rate_shrunk_bayes"].clip(1e-6, 100 - 1e-6) / 100.0
        df["log_odds_bayes"] = np.log(p / (1 - p))

        if "delta" in df.columns:
            df["delta_shrunk_bayes"] = (
                (df["delta"] * df["sample_size"])
                / (df["sample_size"] + m_value)
            )

        self._invalidate_cache()

    def save(self, path: str = "data/matchups_shrunk.csv") -> None:
        cols = [
            "champ1", "role1", "type", "champ2", "role2",
            "win_rate", "sample_size",
            "win_rate_shrunk_bayes", "log_odds_bayes",
            "win_rate_shrunk_advi", "log_odds_advi",
            "win_rate_shrunk_hierarchical", "log_odds_hierarchical",
            "delta", "delta_shrunk_bayes", "log_odds",
        ]

        existing_cols = [col for col in cols if col in self._df.columns]
        self._df.to_csv(path, columns=existing_cols, index=False)

    def update_overall_scores(
        self,
        priors_repo,
        enemy_list: list[str],
        ally_team: dict[str, str],
        method: str = "Bayesian",
    ) -> tuple[float, float]:
        key = self._normalize_method(method)

        enemy_team = guess_enemy_roles(enemy_list, priors_repo)

        ally_team = {
            str(role).lower(): champ
            for role, champ in ally_team.items()
            if champ
        }

        ally_score, enemy_score = calculate_overall_win_rates(
            self.indexed(key),
            ally_team,
            enemy_team,
        )

        return ally_score, enemy_score

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _with_log_odds(self, method: str) -> pd.DataFrame:
        """
        Returns a dataframe with an active 'log_odds' column for the chosen method.
        Does not rely on a possibly stale existing 'log_odds' column.
        """
        key = self._normalize_method(method)
        log_col = METHOD_TO_LOG_COL.get(key)

        if log_col and log_col in self._df.columns:
            return self._df.assign(log_odds=self._df[log_col])

        if "win_rate" in self._df.columns:
            p = self._df["win_rate"].clip(1e-6, 100 - 1e-6) / 100.0
            return self._df.assign(log_odds=np.log(p / (1 - p)))

        raise ValueError(
            f"No source for log_odds using method '{method}'. "
            f"Available columns: {list(self._df.columns)}"
        )

    def _normalize_method(self, method: str) -> str:
        method = str(method or "Bayesian").strip().lower()

        aliases = {
            "bayesian": "bayesian",
            "bayes": "bayesian",
            "advi": "advi",
            "hierarchical": "hierarchical",
        }

        if method not in aliases:
            raise ValueError(
                f"Invalid method '{method}'. "
                f"Expected one of: {list(aliases.keys())}"
            )

        return aliases[method]

    def _normalize_columns(self) -> None:
        """
        Normalizes role/type/champion string columns.
        Does not lowercase champion names, because display names should be preserved.
        """
        missing = [col for col in IDX_COLS if col not in self._df.columns]

        if missing:
            raise ValueError(f"Missing required matchup index columns: {missing}")

        for col in ["role1", "role2", "type"]:
            self._df[col] = self._df[col].astype(str).str.strip().str.lower()

        self._df["type"] = self._df["type"].replace({
            "synergy": "Synergy",
            "counter": "Counter",
        })

        for col in ["champ1", "champ2"]:
            self._df[col] = self._df[col].astype(str).str.strip()

    def _invalidate_cache(self) -> None:
        self._idx_cache = None
        self._idx_key = None