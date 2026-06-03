from core.services import TeamState, Metric, PickStrategy


class RecommendView:
    def __init__(
        self,
        ally_champs: dict,
        enemy_champ_boxes,
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

        enemy_team = {}
        enemy_champs = []

        if isinstance(self._enemy_champ_boxes, dict):
            for role, entry in self._enemy_champ_boxes.items():
                champ_name = self._resolve_champ_name(entry.get_text().strip())

                if champ_name:
                    enemy_team[role] = champ_name
                    enemy_champs.append(champ_name)
        else:
            for entry in self._enemy_champ_boxes:
                champ_name = self._resolve_champ_name(entry.get_text().strip())

                if champ_name:
                    enemy_champs.append(champ_name)

        return TeamState(
            ally_team=ally_team,
            enemy_champs=enemy_champs,
            enemy_team=enemy_team,
            banned_champs=self._get_banned_champs(),
            metric=self._get_metric(),
            pick_strategy=self._get_pick_strategy(),
        )

    def update_enemy_guess_label(
        self,
        enemy_team_role_guess: dict[str, str],
        enemy_role_probabilities: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        if isinstance(self._enemy_guess_label, dict):
            self._update_enemy_guess_labels_by_role(
                enemy_team_role_guess,
                enemy_role_probabilities,
            )
            return

        self._update_single_enemy_guess_label(
            enemy_team_role_guess,
            enemy_role_probabilities,
        )

    def _update_enemy_guess_labels_by_role(
        self,
        enemy_team_role_guess: dict[str, str],
        enemy_role_probabilities: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        for label in self._enemy_guess_label.values():
            label.config(text="")

        if not enemy_role_probabilities:
            for role, champ in enemy_team_role_guess.items():
                if role in self._enemy_guess_label:
                    self._enemy_guess_label[role].config(text=champ)

            return

        role_to_lines: dict[str, list[tuple[str, float]]] = {
            role: []
            for role in self._enemy_guess_label.keys()
        }

        for champ, probabilities in enemy_role_probabilities.items():
            for role, probability in probabilities:
                if role not in role_to_lines:
                    continue

                if probability <= 0:
                    continue

                role_to_lines[role].append((champ, probability))

        for role, champ_probs in role_to_lines.items():
            champ_probs.sort(key=lambda item: item[1], reverse=True)

            lines = [
                f"{champ} {probability:.0%}"
                for champ, probability in champ_probs[:3]
            ]

            self._enemy_guess_label[role].config(text="\n".join(lines))

    def _update_single_enemy_guess_label(
        self,
        enemy_team_role_guess: dict[str, str],
        enemy_role_probabilities: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        if not enemy_team_role_guess and not enemy_role_probabilities:
            self._enemy_guess_label.config(text="")
            return

        lines: list[str] = []

        if enemy_role_probabilities:
            for champ, probabilities in enemy_role_probabilities.items():
                if not probabilities:
                    continue

                prob_text = ", ".join(
                    f"{role} {prob:.0%}"
                    for role, prob in probabilities[:3]
                )

                lines.append(f"{champ}: {prob_text}")

        if not lines and enemy_team_role_guess:
            lines = [
                f"{champ} → {role}"
                for role, champ in enemy_team_role_guess.items()
            ]

        self._enemy_guess_label.config(text="\n".join(lines))

    def clear_enemy_guess_label(self) -> None:
        if isinstance(self._enemy_guess_label, dict):
            for label in self._enemy_guess_label.values():
                label.config(text="")
            return

        self._enemy_guess_label.config(text="")

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