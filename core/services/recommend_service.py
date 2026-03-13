from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from core.enums import ROLES, Role
from core.repo import MatchupRepository, PriorsRepository
from core.role_guess import guess_enemy_roles
from core.recommend import get_champion_scores_for_role

@dataclass(frozen=True)
class TeamState:
    ally_team: Dict[Role, str]
    enemy_champs: List[str]
    metric: Literal["Delta", "WinRate"]
    pick_strategy: Literal["Maximize", "MinimaxAllRoles"]

@dataclass(frozen=True)
class RecommendResult:
    enemy_team_role_guess: Dict[str, str]
    ally_role_suggestions: Dict[Role, List[Tuple[str, float, float]]]

class RecommendService:
    def __init__(self, matchup_repo: MatchupRepository, priors_repo: PriorsRepository, champion_list: list[str]):
        self._matchup_repo = matchup_repo
        self._priors_repo = priors_repo
        self._champion_list = champion_list
    
    def update_adjustments(self, method: str = "Bayesian"):
        self._matchup_repo._create_column(method)

    def recommend(self, state: TeamState):

        # Guess what roles the enemy champions are
        enemy_team_role_guess = guess_enemy_roles(state.enemy_champs, self._priors_repo)


        # Recommend top 5 picks for each ally role
        ally_pick_suggestions: dict[Role, list[tuple[str, float, float]]] = {}
        for role in ROLES:
            scores = get_champion_scores_for_role(
                df_indexed=self._matchup_repo.indexed(),
                role_to_fill=role,
                ally_team=state.ally_team,
                enemy_team=enemy_team_role_guess,
                pick_strategy="Maximize",
                champion_pool=state.enemy_champs,
                )

                        # Sort by Delta or WinRate
            if state.metric == "Delta":
                scores.sort(key=lambda x: x[2], reverse=True)
            else:
                scores.sort(key=lambda x: x[1], reverse=True)

            top_n = scores[:5]  # show top 5

            ally_pick_suggestions[Role(role)] = top_n
        
        return RecommendResult(
            enemy_team_role_guess=enemy_team_role_guess,
            ally_role_suggestions=ally_pick_suggestions
        )
