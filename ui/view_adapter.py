from core.services.win_rate_service import TeamInput

class TkWinRateViewAdapter:
    def __init__(self, ally_champs: dict, enemy_widgets: list, list_widget):
        self._ally_champs = ally_champs
        self._enemy_widgets = enemy_widgets
        self._label = list_widget
    
    def read_team_input(self) -> TeamInput:
        ally = {
            r: e.get_text().strip()
            for r, e in self._ally_champs.items()
            if e.get_text().strip()
        }
        enemy = [e.get_text().strip() for e in self._enemy_widgets if e.get_text().strip()]
        return TeamInput(ally_by_role=ally, enemy_list=enemy)

    def render_win_rates(self, text: str) -> None:
        self._label.config(text=text)