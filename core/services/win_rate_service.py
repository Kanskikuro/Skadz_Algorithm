from dataclasses import dataclass
from typing import Dict, List

from core.repo import MatchupRepository, PriorsRepository

@dataclass(frozen=True)
class TeamInput:
    ally_by_role: Dict[str, str]
    enemy_list: List[str]

@dataclass(frozen=True)
class WinRateEstimate:
    ally_pct: float
    enemy_pct: float

class WinRateService:
    def __init__(self, priors_repo: PriorsRepository, matchup_repo: MatchupRepository):
        self.priors_repo = priors_repo
        self.matchup_repo = matchup_repo
    
    def estimate(self, team: TeamInput, method: str = "Bayesian") -> WinRateEstimate:
        ally_pct, enemy_pct = self.matchup_repo.update_overall_win_rates(
            self.priors_repo,
            team.enemy_list,
            team.ally_by_role,
            method,
        )

        return WinRateEstimate(ally_pct, enemy_pct)
    
class WinRatePresenter:
    @staticmethod
    def to_label_text(est: WinRateEstimate):
        return (
            f"Estimated Ally Team Win Rate: {est.ally_pct:.2%}\n"
            f"Estimated Enemy Team Win Rate: {est.enemy_pct:.2%}"
        )