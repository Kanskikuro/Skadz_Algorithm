from core.services import TeamState

class RecommendView:
    def __init__(self, ally_champs: dict, enemy_champ_boxes: list):
        self._ally_champs = ally_champs
        self._enemy_champ_boxes = enemy_champ_boxes

    def get_team_state(self) -> TeamState:
        ally_team = {}
        for role, entry in self._ally_champs.items():
            nm = entry.get_text().strip()
            if nm:
                ally_team[role] = nm

        enemy_champs = [e.get_text().strip() for e in self._enemy_champ_boxes if e.get_text().strip()]
        
        return TeamState(
            ally_team=ally_team, 
            enemy_champs=enemy_champs,
            metric="Delta", ## Hard for now
            pick_strategy="Maximize",
        )