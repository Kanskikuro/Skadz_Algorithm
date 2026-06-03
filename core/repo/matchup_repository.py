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

        self._fast_lookup_cache: dict[
            str,
            dict[tuple[str, str, str, str, str], float],
        ] = {}

        self._champion_roles_cache: set[tuple[str, str]] | None = None
        self._champion_to_roles_cache: dict[str, set[str]] | None = None

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

    def fast_lookup(
        self,
        method: str = "Bayesian",
    ) -> dict[tuple[str, str, str, str, str], float]:
        """
        Fast dictionary lookup for scoring.

        Key:
            (champ1, role1, type, champ2, role2)

        Value:
            log_odds for the selected adjustment method.
        """
        key = self._normalize_method(method)

        if key in self._fast_lookup_cache:
            return self._fast_lookup_cache[key]

        df = self._with_log_odds(key)

        lookup = {
            (champ1, role1, relation_type, champ2, role2): float(log_odds)
            for champ1, role1, relation_type, champ2, role2, log_odds
            in df[["champ1", "role1", "type", "champ2", "role2", "log_odds"]]
            .itertuples(index=False, name=None)
        }

        self._fast_lookup_cache[key] = lookup
        return lookup

    def fast_delta_lookup(
        self,
        method: str = "Bayesian",
    ) -> dict[tuple[str, str, str, str, str], float]:
        """
        Fast dictionary lookup for delta values.

        Key:
            (champ1, role1, type, champ2, role2)

        Value:
            delta_shrunk_bayes if available, otherwise 0.0.
        """
        key = self._normalize_method(method)
        cache_key = f"{key}:delta"

        if cache_key in self._fast_lookup_cache:
            return self._fast_lookup_cache[cache_key]

        df = self._with_log_odds(key)

        if "delta_shrunk_bayes" not in df.columns:
            lookup = {
                (champ1, role1, relation_type, champ2, role2): 0.0
                for champ1, role1, relation_type, champ2, role2
                in df[["champ1", "role1", "type", "champ2", "role2"]]
                .itertuples(index=False, name=None)
            }
        else:
            lookup = {
                (champ1, role1, relation_type, champ2, role2): float(delta)
                for champ1, role1, relation_type, champ2, role2, delta
                in df[
                    [
                        "champ1",
                        "role1",
                        "type",
                        "champ2",
                        "role2",
                        "delta_shrunk_bayes",
                    ]
                ].itertuples(index=False, name=None)
            }

        self._fast_lookup_cache[cache_key] = lookup
        return lookup

    def get_pair_score(
        self,
        champ1: str,
        role1: str,
        relation_type: str,
        champ2: str,
        role2: str,
        method: str = "Bayesian",
        default: float | None = 0.0,
    ) -> float | None:
        """
        Fast single score lookup.

        relation_type should usually be 'Synergy' or 'Counter'.
        """
        lookup = self.fast_lookup(method)

        return lookup.get(
            (
                str(champ1).strip(),
                str(role1).strip().lower(),
                self._normalize_relation_type(relation_type),
                str(champ2).strip(),
                str(role2).strip().lower(),
            ),
            default,
        )

    def get_pair_delta(
        self,
        champ1: str,
        role1: str,
        relation_type: str,
        champ2: str,
        role2: str,
        method: str = "Bayesian",
        default: float | None = 0.0,
    ) -> float | None:
        """
        Fast single delta lookup.
        Uses delta_shrunk_bayes if available, otherwise returns default.
        """
        lookup = self.fast_delta_lookup(method)

        return lookup.get(
            (
                str(champ1).strip(),
                str(role1).strip().lower(),
                self._normalize_relation_type(relation_type),
                str(champ2).strip(),
                str(role2).strip().lower(),
            ),
            default,
        )

    def champion_roles(self) -> set[tuple[str, str]]:
        """
        Returns all known valid (champion, role) pairs.
        Useful for MinimaxAllRoles so it does not test impossible roles.
        """
        if self._champion_roles_cache is not None:
            return self._champion_roles_cache

        pairs = {
            (champ, role)
            for champ, role in self._df[["champ1", "role1"]]
            .itertuples(index=False, name=None)
        }

        self._champion_roles_cache = pairs
        return pairs

    def roles_for_champion(self, champion: str) -> set[str]:
        """
        Returns valid roles for one champion.
        """
        if self._champion_to_roles_cache is None:
            mapping: dict[str, set[str]] = {}

            for champ, role in self.champion_roles():
                mapping.setdefault(champ, set()).add(role)

            self._champion_to_roles_cache = mapping

        return self._champion_to_roles_cache.get(str(champion).strip(), set())

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
            "champ1",
            "role1",
            "type",
            "champ2",
            "role2",
            "win_rate",
            "sample_size",
            "win_rate_shrunk_bayes",
            "log_odds_bayes",
            "win_rate_shrunk_advi",
            "log_odds_advi",
            "win_rate_shrunk_hierarchical",
            "log_odds_hierarchical",
            "delta",
            "delta_shrunk_bayes",
            "log_odds",
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

        self._df["type"] = self._df["type"].replace(
            {
                "synergy": "Synergy",
                "counter": "Counter",
            }
        )

        for col in ["champ1", "champ2"]:
            self._df[col] = self._df[col].astype(str).str.strip()

    def _normalize_relation_type(self, relation_type: str) -> str:
        value = str(relation_type).strip().lower()

        aliases = {
            "synergy": "Synergy",
            "counter": "Counter",
        }

        if value not in aliases:
            raise ValueError(
                f"Invalid relation type '{relation_type}'. "
                f"Expected 'Synergy' or 'Counter'."
            )

        return aliases[value]

    def _invalidate_cache(self) -> None:
        self._idx_cache = None
        self._idx_key = None
        self._fast_lookup_cache.clear()
        self._champion_roles_cache = None
        self._champion_to_roles_cache = None