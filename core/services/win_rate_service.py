from dataclasses import dataclass
from typing import Mapping, Sequence

from core.repo import MatchupRepository, PriorsRepository


"""
Service for calculating the estimated win rate of both teams.
"""


@dataclass(frozen=True)
class TeamInput:
    ally_by_role: Mapping[str, str]
    enemy_list: Sequence[str]


@dataclass(frozen=True)
class WinRateEstimate:
    ally_win_rate: float
    enemy_win_rate: float


class WinRateService:
    def __init__(
        self,
        priors_repo: PriorsRepository,
        matchup_repo: MatchupRepository,
    ):
        self.priors_repo = priors_repo
        self.matchup_repo = matchup_repo

    def estimate(
        self,
        team: TeamInput,
        method: str = "Bayesian",
    ) -> WinRateEstimate:
        ally_win_rate, enemy_win_rate = self.matchup_repo.update_overall_win_rates(
            self.priors_repo,
            list(team.enemy_list),
            dict(team.ally_by_role),
            method,
        )

        return WinRateEstimate(
            ally_win_rate=ally_win_rate,
            enemy_win_rate=enemy_win_rate,
        )


class WinRatePresenter:
    @staticmethod
    def to_label_text(est: WinRateEstimate) -> str:
        return (
            f"Estimated Ally Team Win Rate: {est.ally_win_rate:.2%}\n"
            f"Estimated Enemy Team Win Rate: {est.enemy_win_rate:.2%}"
        )