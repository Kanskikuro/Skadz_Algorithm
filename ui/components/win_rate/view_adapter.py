from core.services import TeamInput


class TkWinRateViewAdapter:
    def __init__(self, ally_champs: dict, enemy_widgets: list, label_widget):
        self._ally_champs = ally_champs
        self._enemy_widgets = enemy_widgets
        self._label = label_widget

    def read_team_input(self) -> TeamInput:
        ally_by_role: dict[str, str] = {}

        for role, entry in self._ally_champs.items():
            champ_name = entry.get_text().strip()

            if champ_name:
                role_value = role.value if hasattr(role, "value") else str(role).lower()
                ally_by_role[role_value] = champ_name

        enemy_list: list[str] = []

        for entry in self._enemy_widgets:
            champ_name = entry.get_text().strip()

            if champ_name:
                enemy_list.append(champ_name)

        return TeamInput(
            ally_by_role=ally_by_role,
            enemy_list=enemy_list,
        )

    def render_win_rates(self, text: str) -> None:
        self._label.config(text=text)