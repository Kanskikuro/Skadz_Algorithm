from core.services import TeamState, Metric, PickStrategy



class RecommendView:
    def __init__(
        self,
        ally_champs: dict,
        enemy_champ_boxes: list,
        enemy_guess_label,
        metric_var=None,
        pick_strategy_var=None,
    ):
        self._ally_champs = ally_champs
        self._enemy_champ_boxes = enemy_champ_boxes
        self._enemy_guess_label = enemy_guess_label
        self._metric_var = metric_var
        self._pick_strategy_var = pick_strategy_var

    def get_team_state(self) -> TeamState:
        ally_team = {}

        for role, entry in self._ally_champs.items():
            champ_name = entry.get_text().strip()

            if champ_name:
                ally_team[role] = champ_name

        enemy_champs = []

        for entry in self._enemy_champ_boxes:
            champ_name = entry.get_text().strip()

            if champ_name:
                enemy_champs.append(champ_name)

        return TeamState(
            ally_team=ally_team,
            enemy_champs=enemy_champs,
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
            return "Maximize"

        value = self._pick_strategy_var.get()

        if value == "Maximize":
            return "Maximize"

        if value == "MinimaxAllRoles":
            return "MinimaxAllRoles"

        return "Maximize"