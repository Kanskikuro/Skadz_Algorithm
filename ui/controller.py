from ui.view_adapter import TkWinRateViewAdapter
from core.services.win_rate_service import WinRateService, WinRatePresenter


class WinRateController:
    def __init__(self, view: TkWinRateViewAdapter, service: WinRateService, presenter: WinRatePresenter):
        self.view = view
        self.service = service
        self.presenter = presenter

    def on_update(self) -> None:
        team = self.view.read_team_input()
        estimate = self.service.estimate(team)
        text = self.presenter.to_label_text(estimate)
        self.view.render_win_rates(text)

        pass