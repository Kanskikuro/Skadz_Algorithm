from core.services import TeamState

class RecommendView:
    def __init__(self, ally_champs: dict, enemy_champ_boxes: list, enemy_guess_label):
        self._ally_champs = ally_champs
        self._enemy_champ_boxes = enemy_champ_boxes
        self._enemy_guess_label = enemy_guess_label

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

    def update_enemy_guess_label(self, enemy_team_role_guess: dict[str, str]):
        # Display “Akshan → middle” etc.
        guessed_text = ""
        for role, champ in enemy_team_role_guess.items():
            guessed_text += f"{champ} → {role}\n"
        self._enemy_guess_label.config(text=guessed_text)
