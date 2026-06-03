from core.services import TeamState, Metric, PickStrategy


class RecommendView:
    def __init__(
        self,
        ally_champs: dict,
        enemy_champ_boxes: list,
        enemy_guess_label,
        metric_var=None,
        pick_strategy_var=None,
        champion_resolver=None,
        banned_champs_provider=None,
    ):
        self._ally_champs = ally_champs
        self._enemy_champ_boxes = enemy_champ_boxes
        self._enemy_guess_label = enemy_guess_label
        self._metric_var = metric_var
        self._pick_strategy_var = pick_strategy_var
        self._champion_resolver = champion_resolver
        self._banned_champs_provider = banned_champs_provider

    def get_team_state(self) -> TeamState:
        ally_team = {}

        for role, entry in self._ally_champs.items():
            champ_name = self._resolve_champ_name(entry.get_text().strip())

            if champ_name:
                ally_team[role] = champ_name

        enemy_champs = []

        for entry in self._enemy_champ_boxes:
            champ_name = self._resolve_champ_name(entry.get_text().strip())

            if champ_name:
                enemy_champs.append(champ_name)

        return TeamState(
            ally_team=ally_team,
            enemy_champs=enemy_champs,
            banned_champs=self._get_banned_champs(),
            metric=self._get_metric(),
            pick_strategy=self._get_pick_strategy(),
        )

    def update_enemy_guess_label(self, enemy_team_role_guess: dict[str, str]) -> None:
        if not enemy_team_role_guess:
            self._enemy_guess_label.config(text="")
            return

        guessed_text = "\n".join(
            f"{champ} → {role}"
            for role, champ in enemy_team_role_guess.items()
        )

        self._enemy_guess_label.config(text=guessed_text)

    def _get_banned_champs(self) -> list[str]:
        if self._banned_champs_provider is None:
            return []

        banned_champs = []

        for champ in self._banned_champs_provider():
            resolved = self._resolve_champ_name(str(champ).strip())

            if resolved and resolved not in banned_champs:
                banned_champs.append(resolved)

        return banned_champs

    def _resolve_champ_name(self, name: str) -> str:
        if not name:
            return ""

        if self._champion_resolver is None:
            return name

        resolved = self._champion_resolver.resolve_name(name)
        return resolved or name

    def _get_metric(self) -> Metric:
        if self._metric_var is None:
            return "Delta"

        value = self._metric_var.get()

        if value == "Delta":
            return "Delta"

        if value == "WinRate":
            return "WinRate"

        return "Delta"

    def _get_pick_strategy(self) -> PickStrategy:
        if self._pick_strategy_var is None:
            return "Hybrid"

        value = self._pick_strategy_var.get()

        if value == "Hybrid":
            return "Hybrid"

        if value == "Maximize":
            return "Maximize"

        if value == "MinimaxAllRoles":
            return "MinimaxAllRoles"

        return "Hybrid"