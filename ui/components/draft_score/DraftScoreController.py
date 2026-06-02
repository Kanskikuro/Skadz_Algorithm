# ui/components/draft_score/draft_score_controller.py

from .view_adapter import TkDraftScoreViewAdapter
from core.services import DraftScoreService, DraftScorePresenter


class DraftScoreController:
    def __init__(
        self,
        view: TkDraftScoreViewAdapter,
        service: DraftScoreService,
        presenter: DraftScorePresenter,
    ):
        self.view = view
        self.service = service
        self.presenter = presenter

    def on_update(self) -> None:
        team = self.view.read_team_input()
        estimate = self.service.estimate(team)
        text = self.presenter.to_label_text(estimate)
        self.view.render_draft_scores(text)