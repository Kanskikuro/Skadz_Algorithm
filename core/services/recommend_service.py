from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Mapping, Sequence

from core.enums import ROLES, Role
from core.repo import MatchupRepository, PriorsRepository
from core.role_guess import guess_enemy_roles
from core.recommend import get_champion_scores_for_role


Metric = Literal["Delta", "WinRate"]
PickStrategy = Literal["Maximize", "MinimaxAllRoles", "Hybrid"]

DEFAULT_PICK_STRATEGY: PickStrategy = "Hybrid"


@dataclass(frozen=True)
class TeamState:
    ally_team: Mapping[Role | str, str]
    enemy_champs: Sequence[str]
    banned_champs: Sequence[str]
    metric: Metric
    pick_strategy: PickStrategy = DEFAULT_PICK_STRATEGY


@dataclass(frozen=True)
class RecommendResult:
    enemy_team_role_guess: Dict[str, str]
    ally_role_suggestions: Dict[Role, List[Tuple[str, float, float]]]


class RecommendService:
    def __init__(
        self,
        matchup_repo: MatchupRepository,
        priors_repo: PriorsRepository,
        champion_list: list[str],
    ):
        self._matchup_repo = matchup_repo
        self._priors_repo = priors_repo
        self._champion_list = list(champion_list)
        self._method = "Bayesian"

    def update_adjustments(self, method: str = "Bayesian") -> None:
        self._method = method
        self._matchup_repo.update_adjustments(method)

    def recommend(self, state: TeamState) -> RecommendResult:
        ally_team = self._normalize_ally_team(state.ally_team)
        enemy_champs = self._clean_champion_list(state.enemy_champs)
        banned_champs = self._clean_champion_list(state.banned_champs)

        pick_strategy = self._normalize_pick_strategy(state.pick_strategy)

        enemy_team_role_guess = guess_enemy_roles(
            enemy_champs,
            self._priors_repo,
        )

        excluded_champions = (
            set(ally_team.values())
            | set(enemy_champs)
            | set(banned_champs)
        )

        ally_pick_suggestions: dict[Role, list[tuple[str, float, float]]] = {}

        for role in ROLES:
            role_value = role.value if isinstance(role, Role) else str(role).lower()
            role_key = role if isinstance(role, Role) else Role(role_value)

            scores = get_champion_scores_for_role(
                matchup_repo=self._matchup_repo,
                method=self._method,
                role_to_fill=role_value,
                ally_team=ally_team,
                enemy_team=enemy_team_role_guess,
                pick_strategy=pick_strategy,
                champion_pool=self._champion_list,
                enemy_candidate_pool=self._champion_list,
                excluded_champions=excluded_champions,
            )

            metric_index = 2 if state.metric == "Delta" else 1
            scores.sort(key=lambda x: x[metric_index], reverse=True)

            ally_pick_suggestions[role_key] = scores[:5]

        return RecommendResult(
            enemy_team_role_guess=enemy_team_role_guess,
            ally_role_suggestions=ally_pick_suggestions,
        )

    @staticmethod
    def _normalize_pick_strategy(pick_strategy: str | None) -> PickStrategy:
        if pick_strategy == "Maximize":
            return "Maximize"

        if pick_strategy == "MinimaxAllRoles":
            return "MinimaxAllRoles"
        
        if pick_strategy == "Hybrid":
            return "Hybrid"

        return "Hybrid"

    @staticmethod
    def _normalize_ally_team(ally_team: Mapping[Role | str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}

        for role, champ in ally_team.items():
            if not champ:
                continue

            role_value = role.value if isinstance(role, Role) else str(role).lower()
            normalized[role_value] = str(champ).strip()

        return normalized

    @staticmethod
    def _clean_champion_list(champions: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for champ in champions:
            champ = str(champ).strip()

            if not champ:
                continue

            key = champ.lower()

            if key in seen:
                continue

            seen.add(key)
            cleaned.append(champ)

        return cleaned