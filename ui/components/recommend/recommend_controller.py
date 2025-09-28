
from core.services import RecommendService, TeamState
from .recommend_view import RecommendView


class RecommendController:
    def __init__(self, service: RecommendService, view: RecommendView):
        self.service = service
        self.view = view

    def on_recommend(self):
        self.service.update_adjustments()
        return self.service.recommend(self.view.get_team_state())
