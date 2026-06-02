import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk

from core.enums import ROLES, Role
from core.repo import PriorsRepository, MatchupRepository
from core.score import log_odds_to_probability

from ui.autocompleteEntryPopup import AutocompleteEntryPopup
from ui.components.win_rate import WinRateController, TkWinRateViewAdapter
from ui.components.recommend import RecommendController, RecommendView

from core.services import WinRateService, WinRatePresenter
from core.services import RecommendService


###############################################################################
# The main GUI (ChampionPickerGUI)
###############################################################################

class ChampionPickerGUI(tk.Tk):
    ICON_PATH_FORMAT = os.path.join("data", "champion_icons", "{}.png")
    ICON_SIZE = (64, 64)

    def __init__(
        self,
        matchup_repo: MatchupRepository,
        priors_repo: PriorsRepository,
    ):
        super().__init__()

        self.geometry("1280x920")
        self.minsize(700, 500)
        self.title("League Champion Picker")

        self.roles_ally = ROLES

        self.matchup_repo = matchup_repo
        self.priors_repo = priors_repo

        # Champion list for autocomplete, recommendations, and icons
        self.champion_list: list[str] = self.priors_repo.champions()

        # Pre-load champion icons
        self.champion_icons: dict[str, ImageTk.PhotoImage | None] = {}
        self._load_champion_icons()

        # Settings variables
        self.adjustment_method = tk.StringVar(value="Bayesian")
        self.display_metric_var = tk.StringVar(value="Delta")
        self.pick_strategy_var = tk.StringVar(value="Maximize")
        self.auto_hide = tk.BooleanVar(value=True)

        # Build UI
        self._build_ui()

        # Controllers
        self.win_rate_controller = WinRateController(
            TkWinRateViewAdapter(
                self.ally_champs,
                self.enemy_champ_boxes,
                self.overall_win_rate_label,
            ),
            WinRateService(
                self.priors_repo,
                self.matchup_repo,
            ),
            WinRatePresenter(),
        )

        self.recommend_controller = RecommendController(
            RecommendService(
                self.matchup_repo,
                self.priors_repo,
                self.champion_list,
            ),
            RecommendView(
                self.ally_champs,
                self.enemy_champ_boxes,
                self.enemy_guess_label,
                metric_var=self.display_metric_var,
                pick_strategy_var=self.pick_strategy_var,
            ),
        )

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------

    def _load_champion_icons(self) -> None:
        for champ in self.champion_list:
            path = ChampionPickerGUI.ICON_PATH_FORMAT.format(champ)

            if os.path.exists(path):
                pil_img = Image.open(path).resize(
                    ChampionPickerGUI.ICON_SIZE,
                    resample=Image.LANCZOS,  # type: ignore
                )
                self.champion_icons[champ] = ImageTk.PhotoImage(pil_img)
            else:
                print(f"Icon not found for '{champ}' at {path}")
                self.champion_icons[champ] = None

    def _build_ui(self) -> None:
        bigger_font = ("Helvetica", 14)

        # ---------------------------------------------------------------------
        # Top settings row
        # ---------------------------------------------------------------------

        settings_frame = ttk.Frame(self)
        settings_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="ew")

        auto_hide_chk = ttk.Checkbutton(
            settings_frame,
            text="Auto-hide suggestion when champ picked",
            variable=self.auto_hide,
            command=self.check_filled_roles,
        )
        auto_hide_chk.grid(row=0, column=0, padx=(0, 20), sticky="w")

        ttk.Label(settings_frame, text="Metric:").grid(row=0, column=1, padx=(0, 5), sticky="w")
        metric_box = ttk.Combobox(
            settings_frame,
            textvariable=self.display_metric_var,
            values=["Delta", "WinRate"],
            width=10,
            state="readonly",
        )
        metric_box.grid(row=0, column=2, padx=(0, 20), sticky="w")
        metric_box.bind("<<ComboboxSelected>>", lambda _event: self.combined_callback())

        ttk.Label(settings_frame, text="Strategy:").grid(row=0, column=3, padx=(0, 5), sticky="w")
        strategy_box = ttk.Combobox(
            settings_frame,
            textvariable=self.pick_strategy_var,
            values=["Maximize", "MinimaxAllRoles"],
            width=16,
            state="readonly",
        )
        strategy_box.grid(row=0, column=4, padx=(0, 20), sticky="w")
        strategy_box.bind("<<ComboboxSelected>>", lambda _event: self.combined_callback())

        ttk.Label(settings_frame, text="Adjustment:").grid(row=0, column=5, padx=(0, 5), sticky="w")
        adjustment_box = ttk.Combobox(
            settings_frame,
            textvariable=self.adjustment_method,
            values=["Bayesian", "ADVI", "Hierarchical"],
            width=14,
            state="readonly",
        )
        adjustment_box.grid(row=0, column=6, padx=(0, 20), sticky="w")
        adjustment_box.bind("<<ComboboxSelected>>", lambda _event: self.combined_callback())

        # ---------------------------------------------------------------------
        # Overall win rate label
        # ---------------------------------------------------------------------

        self.overall_win_rate_label = ttk.Label(
            self,
            text="",
            justify="left",
            font=bigger_font,
        )
        self.overall_win_rate_label.grid(
            row=1,
            column=1,
            padx=10,
            pady=5,
            sticky="nw",
        )

        # ---------------------------------------------------------------------
        # Teams frame
        # ---------------------------------------------------------------------

        teams_frame = ttk.Frame(self)
        teams_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="nw",
            padx=10,
            pady=10,
        )

        # ---------------------------------------------------------------------
        # Ally picks frame
        # ---------------------------------------------------------------------

        ally_frame = ttk.LabelFrame(teams_frame, text="Ally Team (roles known)")
        ally_frame.grid(row=0, column=0, padx=10, sticky="nw")

        self.ally_champs = {}

        for i, role in enumerate(self.roles_ally):
            lbl = ttk.Label(
                ally_frame,
                text=f"{role.capitalize()}:",
                font=bigger_font,
            )
            lbl.grid(row=i, column=0, sticky="w")

            entry = AutocompleteEntryPopup(
                ally_frame,
                suggestion_list=self.champion_list,
                width=12,
                font=bigger_font,
                callback=self.combined_callback,
            )
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")

            self.ally_champs[role] = entry

        # ---------------------------------------------------------------------
        # Enemy picks frame
        # ---------------------------------------------------------------------

        enemy_frame = ttk.LabelFrame(teams_frame, text="Enemy Team (champions only)")
        enemy_frame.grid(row=0, column=1, padx=10, sticky="nw")

        self.enemy_champ_boxes = []

        for i in range(5):
            lbl = ttk.Label(
                enemy_frame,
                text=f"Enemy #{i + 1}:",
                font=bigger_font,
            )
            lbl.grid(row=i, column=0, sticky="w")

            entry = AutocompleteEntryPopup(
                enemy_frame,
                suggestion_list=self.champion_list,
                width=12,
                font=bigger_font,
                callback=self.combined_callback,
            )
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")

            self.enemy_champ_boxes.append(entry)

        self.enemy_guess_label = ttk.Label(
            enemy_frame,
            text="",
            justify="left",
            font=bigger_font,
            foreground="blue",
        )
        self.enemy_guess_label.grid(
            row=0,
            column=2,
            rowspan=len(self.enemy_champ_boxes),
            sticky="nw",
            padx=(10, 0),
        )

        # ---------------------------------------------------------------------
        # Suggested picks frame
        # ---------------------------------------------------------------------

        pick_frame = ttk.LabelFrame(self, text="Suggested Picks")
        pick_frame.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=10,
            pady=10,
            sticky="ew",
        )

        btn_reset = ttk.Button(
            pick_frame,
            text="Reset",
            command=self.reset_all,
        )
        btn_reset.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        self.results_frame = ttk.Frame(pick_frame)
        self.results_frame.grid(row=2, column=0, columnspan=3, sticky="ew")

        self.icon_frames = {}

        for i, role in enumerate(self.roles_ally):
            container = ttk.LabelFrame(
                self.results_frame,
                text=role.capitalize(),
            )
            container.grid(row=0, column=i, padx=10, pady=5, sticky="nw")

            self.icon_frames[role] = {
                "container": container,
                "icons": [],
                "should_be_visible": True,
            }

            copy_btn = ttk.Button(
                container,
                text="Copy",
                command=lambda r=role: self.copy_role_list(r),
            )
            copy_btn.grid(row=6, column=0, columnspan=2, pady=(5, 0), sticky="ew")

    # -------------------------------------------------------------------------
    # Main callbacks
    # -------------------------------------------------------------------------

    def combined_callback(self) -> None:
        """
        Called whenever an AutocompleteEntryPopup changes.
        Recomputes hiding logic, overall win rates, and recommendations.
        """
        self.check_filled_roles()
        self.update_overall_win_rates()
        self.on_recommend()

    def update_overall_win_rates(self) -> None:
        self.win_rate_controller.on_update()

    def on_recommend(self) -> None:
        """
        Updates recommendations and redraws the suggested-picks area.
        """
        # Make sure the selected adjustment method is applied.
        self.recommend_controller.service.update_adjustments(
            self.adjustment_method.get()
        )

        recommend_result = self.recommend_controller.on_recommend()

        self.results_frame.grid()
        self.rearrange_result_icons()

        for role in self.roles_ally:
            # Clear previous widgets under this role
            for widget in self.icon_frames[role]["icons"]:
                widget.destroy()

            self.icon_frames[role]["icons"].clear()

            suggestions = recommend_result.ally_role_suggestions.get(Role(role), [])

            for idx, (champ, total_log_odds, total_delta) in enumerate(suggestions):
                self._add_recommendation_row(
                    role=role,
                    row_index=idx,
                    champ=champ,
                    total_log_odds=total_log_odds,
                    total_delta=total_delta,
                )

        self.check_filled_roles()

    def reset_all(self) -> None:
        """
        Clear all entry fields and destroy existing icon subframes.
        """
        for entry in self.ally_champs.values():
            entry.clear()

        for entry in self.enemy_champ_boxes:
            entry.clear()

        for role in self.roles_ally:
            for widget in self.icon_frames[role]["icons"]:
                widget.destroy()

            self.icon_frames[role]["icons"].clear()

        self.enemy_guess_label.config(text="")
        self.overall_win_rate_label.config(text="")

        self.check_filled_roles()

    # -------------------------------------------------------------------------
    # Suggested-pick rendering
    # -------------------------------------------------------------------------

    def _add_recommendation_row(
        self,
        role: str,
        row_index: int,
        champ: str,
        total_log_odds: float,
        total_delta: float,
    ) -> None:
        photo = self.champion_icons.get(champ)

        subframe = ttk.Frame(self.icon_frames[role]["container"])
        subframe.grid(row=row_index, column=0, padx=2, pady=2, sticky="w")

        subframe.champ_name = champ  # type: ignore

        if photo is not None:
            icon_lbl = ttk.Label(subframe, image=photo)
            icon_lbl.image = photo  # type: ignore
            icon_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")
        else:
            text_lbl = ttk.Label(
                subframe,
                text=champ,
                font=("Helvetica", 10),
            )
            text_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")

        text_frame = ttk.Frame(subframe)
        text_frame.grid(row=0, column=1, sticky="nw")

        # A candidate score is not a full team win-rate.
        # If display metric is WinRate, convert log-odds to probability for readability.
        if self.display_metric_var.get() == "WinRate":
            win_rate = log_odds_to_probability(total_log_odds)

            primary_text = f"W: {win_rate:.0%}"
        else:
            primary_text = f"Score: {total_log_odds:.3f}"

        primary_lbl = ttk.Label(
            text_frame,
            text=primary_text,
            font=("Helvetica", 10),
        )
        primary_lbl.pack(anchor="w")

        delta_lbl = ttk.Label(
            text_frame,
            text=f"Δ: {total_delta:.3f}",
            font=("Helvetica", 10),
        )
        delta_lbl.pack(anchor="sw")

        self.icon_frames[role]["icons"].append(subframe)

    def copy_role_list(self, role: str) -> None:
        """
        Copy the champion list for the given role to clipboard.
        """
        names = [
            subframe.champ_name
            for subframe in self.icon_frames[role]["icons"]
        ]

        text = "\n".join(names)

        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()

    # -------------------------------------------------------------------------
    # Auto-hide / layout
    # -------------------------------------------------------------------------

    def check_filled_roles(self) -> None:
        """
        Hides suggestion columns for roles that already have valid ally picks,
        if auto-hide is enabled.
        """
        for role, entry_widget in self.ally_champs.items():
            champ_name = entry_widget.get_text().strip()
            is_valid_champion = champ_name in self.champion_list

            should_show = not (is_valid_champion and self.auto_hide.get())

            self.icon_frames[role]["should_be_visible"] = should_show

        self.rearrange_result_icons()

    def rearrange_result_icons(self) -> None:
        """
        Re-grid visible role containers from left to right.
        """
        for _role, info in self.icon_frames.items():
            info["container"].grid_forget()

        visible_roles = [
            role
            for role in self.roles_ally
            if self.icon_frames[role].get("should_be_visible", True)
        ]

        for idx, role in enumerate(visible_roles):
            self.icon_frames[role]["container"].grid(
                row=0,
                column=idx,
                padx=10,
                pady=5,
                sticky="nw",
            )