from dataclasses import dataclass
from typing import Mapping, Sequence

from core.repo import MatchupRepository, PriorsRepository


@dataclass(frozen=True)
class TeamInput:
    ally_by_role: Mapping[str, str]
    enemy_list: Sequence[str]


@dataclass(frozen=True)
class DraftScoreEstimate:
    ally_score: float
    enemy_score: float


class DraftScoreService:
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
    ) -> DraftScoreEstimate:
        ally_score, enemy_score = self.matchup_repo.update_overall_scores(
            self.priors_repo,
            list(team.enemy_list),
            dict(team.ally_by_role),
            method,
        )

        return DraftScoreEstimate(
            ally_score=float(ally_score),
            enemy_score=float(enemy_score),
        )


class DraftScorePresenter:
    @staticmethod
    def to_label_text(est: DraftScoreEstimate) -> str:
        return (
            f"Ally Draft Score: {est.ally_score:.2%}\n"
            f"Enemy Draft Score: {est.enemy_score:.2%}"
        )