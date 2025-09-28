
from core.services import RecommendService, TeamState
from .recommend_view import RecommendView


class RecommendController:
    def __init__(self, service: RecommendService, view: RecommendView):
        self.service = service
        self.view = view

    def on_recommend(self):
        self.service.update_adjustments()
        recommend_result = self.service.recommend(self.view.get_team_state())
        self.view.update_enemy_guess_label(recommend_result.enemy_team_role_guess)

        return recommend_result
