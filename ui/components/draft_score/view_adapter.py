from core.services import TeamInput


class TkDraftScoreViewAdapter:
    def __init__(self, ally_champs: dict, enemy_widgets, label_widget):
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

        if isinstance(self._enemy_widgets, dict):
            widgets = self._enemy_widgets.values()
        else:
            widgets = self._enemy_widgets

        for entry in widgets:
            champ_name = entry.get_text().strip()

            if champ_name:
                enemy_list.append(champ_name)

        return TeamInput(
            ally_by_role=ally_by_role,
            enemy_list=enemy_list,
        )

    def render_draft_scores(self, text: str) -> None:
        """
        Supports either:

            label_widget = ttk.Label(...)

        or:

            label_widget = {
                "ally": ttk.Label(...),
                "enemy": ttk.Label(...),
            }

        This keeps backward compatibility while allowing separate ally/enemy
        draft-score labels above each team frame.
        """
        if not isinstance(self._label, dict):
            self._label.config(text=text)
            return

        ally_text, enemy_text = self._split_draft_score_text(text)

        ally_label = self._label.get("ally")
        enemy_label = self._label.get("enemy")

        if ally_label is not None:
            ally_label.config(text=ally_text)

        if enemy_label is not None:
            enemy_label.config(text=enemy_text)

    @staticmethod
    def _split_draft_score_text(text: str) -> tuple[str, str]:
        """
        Tries to split presenter output into ally/enemy text.

        Expected examples:
            Ally: 52%
            Enemy: 48%

        or:
            Ally score: 52%
            Enemy score: 48%

        If format is unknown, the full text stays on the ally label.
        """
        if not text:
            return "", ""

        ally_lines: list[str] = []
        enemy_lines: list[str] = []

        for raw_line in str(text).splitlines():
            line = raw_line.strip()

            if not line:
                continue

            lower = line.lower()

            if lower.startswith("ally"):
                ally_lines.append(line)
            elif lower.startswith("enemy"):
                enemy_lines.append(line)
            else:
                ally_lines.append(line)

        ally_text = "\n".join(ally_lines)
        enemy_text = "\n".join(enemy_lines)

        return ally_text, enemy_text