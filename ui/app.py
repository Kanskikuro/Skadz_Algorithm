import tkinter as tk
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk 

from core.recommend import get_champion_scores_for_role
from core.role_guess import guess_enemy_roles
from core.enums import ROLES, STRATEGIES, Role
from core.repo import PriorsRepository, MatchupRepository

from ui.autocompleteEntryPopup import AutocompleteEntryPopup
from ui.components.win_rate import WinRateController, TkWinRateViewAdapter
from core.services import WinRateService, WinRatePresenter
from core.services import RecommendService, TeamState


###############################################################################
# The main GUI (ChampionPickerGUI)
###############################################################################
class ChampionPickerGUI(tk.Tk):
    ICON_PATH_FORMAT = os.path.join("data", "champion_icons", "{}.png")
    ICON_SIZE = (64, 64)

    def __init__(self, matchup_repo: MatchupRepository, priors_repo: PriorsRepository):
        super().__init__()
        self.geometry("1280x920")
        self.minsize(700, 500)
        self.title("League Champion Picker")

        # Roles on the ally side
        self.roles_ally = ROLES

        # Store and index matchups DataFrame
        self.matchup_repo = matchup_repo
        self.priors_repo = priors_repo

        # Champion list (for autocomplete + icons)
        self.champion_list: list[str] = self.priors_repo.champions()

        # Pre-load champion icons
        self.champion_icons: dict[str, ImageTk.PhotoImage | None] = {}
        for champ in self.champion_list:
            path = ChampionPickerGUI.ICON_PATH_FORMAT.format(champ)
            if os.path.exists(path):
                pil_img = Image.open(path).resize(
                    ChampionPickerGUI.ICON_SIZE,
                    resample=Image.LANCZOS # type: ignore
                )
                self.champion_icons[champ] = ImageTk.PhotoImage(pil_img)
            else:
                print(f"⚠️ Icon not found for '{champ}' at {path}")
                self.champion_icons[champ] = None

        # StringVars / BooleanVars for settings
        self.adjustment_method = tk.StringVar(value="Bayesian")
        self.display_metric_var = tk.StringVar(value="Delta")
        self.pick_strategy_var = tk.StringVar(value="Maximize")
        self.auto_hide = tk.BooleanVar(value=True)

        # Build the UI
        self._build_ui()

        # Controllers
        self.win_rate_controller = WinRateController(
            TkWinRateViewAdapter(self.ally_champs, self.enemy_champ_boxes, self.overall_win_rate_label),
            WinRateService(self.priors_repo, self.matchup_repo),
            WinRatePresenter()
        )

    def combined_callback(self):
        """
        Called whenever any AutocompleteEntryPopup changes.
        Recomputes hiding logic, overall win rates, and recommendations.
        """
        self.check_filled_roles()
        self.update_overall_win_rates()
        self.on_recommend()
        
    def check_filled_roles(self):
        """
        Called any time ally-champ text changes OR the auto-hide checkbox toggles.
        We compute “should this role be visible?” for every role, then let
        rearrange_result_icons() do all the actual grid/grid_forget calls.
        """
        # 1) For each role, decide visible=True/False and store it somewhere.
        #    Visible = “either auto_hide is OFF, or champ name is blank/invalid”.
        for role, entry_widget in self.ally_champs.items():
            champ_name = entry_widget.get_text().strip()
            is_valid_champion = (champ_name in self.champion_list)

            # If auto_hide is True and the user has typed a VALID champ, hide it.
            # Otherwise (auto_hide=False, or name blank/invalid), show it.
            should_show = (not (is_valid_champion and self.auto_hide.get()))

            # Store this boolean in a simple dict; rearrange_result_icons() will read it.
            self.icon_frames[role]['should_be_visible'] = should_show

        # 2) Now reposition ALL containers in one sweep.
        self.rearrange_result_icons()


    def toggle_role_visibility(self, role, visible):
        """
        If auto_hide is True and visible is False, remove that role's container
        (grid_forget). Otherwise, leave it unmapped for now and let
        rearrange_result_icons() do the actual grid() call.
        """
        container = self.icon_frames[role]['container']
        if not visible and self.auto_hide.get():
            container.grid_forget()
        # If visible is True, do NOT call container.grid() here.
        # We'll let rearrange_result_icons() place it in the right spot.   


    def rearrange_result_icons(self):
        """
        Forget (i.e. remove) all frames, then re-grid only those where
        'should_be_visible' is True, in the order of roles_ally.
        This guarantees that anything we show always lands in row=0, col=0…N.
        """
        # First, un‐grid everything.
        for role, info in self.icon_frames.items():
            info['container'].grid_forget()

        # Build an ordered list of roles that should be visible.
        # (Assume self.roles_ally is a list in the order you want.)
        visible_roles = [
            role for role in self.roles_ally
            if (info := self.icon_frames.get(role))
            and info.get('should_be_visible', True)
        ]

        # Now grid them left→right at row=0, col=0…len(visible_roles)-1.
        for idx, role in enumerate(visible_roles):
            self.icon_frames[role]['container'].grid(
                row=0,
                column=idx,
                padx=10,
                pady=5,
                sticky="nw"
            )
    def toggle_advanced_settings(self):
        """
        Show or hide the advanced settings frame based on self.advanced_visible.
        """
        if self.advanced_visible.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def recalculate_matchups(self):
        try:
            self.matchup_repo.recalculate_matchups(self.m_var.get())
            # Refresh UI
            self.update_overall_win_rates()
            self.on_recommend()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to recalculate matchups: {e}")

    def copy_role_list(self, role):
        """
        Copy the champion list (in order) for the given role to clipboard.
        """
        names = [subframe.champ_name for subframe in self.icon_frames[role]['icons']]
        text = "\n".join(names)
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()

    def on_recommend(self):
        """
        1) Update log_odds column based on the selected adjustment method.
        2) Build ally_team dict and guess enemy_team.
        3) For each role, compute top-5 picks, then place icon + W/Δ in a vertical list under that role’s frame.
        """
        recommender = RecommendService(
            self.matchup_repo, 
            self.priors_repo,
            self.champion_list
            )

        # ── Update log_odds in df_matchups ───────────────────────────────────
        method = self.adjustment_method.get().lower()
        
        # self.matchup_repo._create_column(method)
        recommender.update_adjustments(method)

        # ── Build ally_team dict ──────────────────────────────────────────────
        ally_team = {}
        for role, entry in self.ally_champs.items():
            nm = entry.get_text().strip()
            if nm:
                ally_team[role] = nm

        # ── Gather enemy champions and guess roles ────────────────────────────
        enemy_champs = [e.get_text().strip() for e in self.enemy_champ_boxes if e.get_text().strip()]
        
        recommend_result = recommender.recommend(
            state=TeamState(
                ally_team=ally_team,
                enemy_champs=enemy_champs,
                metric="Delta", ## Hard for now
                pick_strategy="Maximize",
            ))

        # Display “Akshan → middle” etc.
        guessed_text = ""
        for role, champ in recommend_result.enemy_team_role_guess.items():
            guessed_text += f"{champ} → {role}\n"
        self.enemy_guess_label.config(text=guessed_text)


        # Ensure the Suggested Picks area is visible before drawing
        self.results_frame.grid()
        self.rearrange_result_icons()

        # ── For each role, compute top-5 and build a vertical list ────────────
        for role in self.roles_ally:
            # Clear out previous widgets under this role
            for widget in self.icon_frames[role]['icons']:
                widget.destroy()
            self.icon_frames[role]['icons'].clear()

            # Place each champion’s icon + W/Δ in a vertical stack
            for idx, (champ, total_log_odds, total_delta) in enumerate(recommend_result.ally_role_suggestions[Role(role)]):
                photo = self.champion_icons.get(champ)

                # … inside on_recommend(), for each (champ, total_log_odds, total_delta) …
                subframe = ttk.Frame(self.icon_frames[role]['container'])
                subframe.grid(row=idx, column=0, padx=2, pady=2, sticky="w")

                subframe.champ_name = champ # type: ignore

                # Icon goes in row=0, column=0
                if photo is not None:
                    icon_lbl = ttk.Label(subframe, image=photo)
                    icon_lbl.image = photo # type: ignore
                    icon_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")
                else:
                    text_lbl = ttk.Label(subframe, text=champ, font=("Helvetica", 10))
                    text_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")

                win_pct = 100 * total_log_odds
                delta_pct = 10 * total_delta

                # Create a small “text_frame” in row=0, column=1 that will stack two labels:
                text_frame = ttk.Frame(subframe)
                text_frame.grid(row=0, column=1, sticky="nw")

                win_lbl = ttk.Label(text_frame, text=f"W: {win_pct:.0f}%", font=("Helvetica", 10))
                win_lbl.pack(anchor="w")
                delta_lbl = ttk.Label(text_frame, text=f"Δ: {delta_pct:.1f}", font=("Helvetica", 10))
                delta_lbl.pack(anchor="sw")

                self.icon_frames[role]['icons'].append(subframe)

        # Re-apply auto-hide logic
        self.check_filled_roles()

    def update_overall_win_rates(self):
        self.win_rate_controller.on_update()

    def reset_all(self):
        """
        Clear all entry fields and destroy any existing icon subframes.
        """
        for e in self.ally_champs.values():
            e.entry_var.set("")
        for e in self.enemy_champ_boxes:
            e.entry_var.set("")

        for role in self.roles_ally:
            for widget in self.icon_frames[role]['icons']:
                widget.destroy()
            self.icon_frames[role]['icons'].clear()

        self.enemy_guess_label.config(text="")
        self.overall_win_rate_label.config(text="")

        self.check_filled_roles()
        self.update_overall_win_rates()

    def _build_ui(self):
        bigger_font = ("Helvetica", 14)

        # ── AUTO-HIDE CHECKBUTTON ─────────────────────────────────────────────────────
        auto_hide_chk = ttk.Checkbutton(
            self,
            text="Auto-hide suggestion when champ picked",
            variable=self.auto_hide,
            command=self.check_filled_roles
        )
        auto_hide_chk.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        # ── ADVANCED SETTINGS TOGGLE ────────────────────────────────────────────────────
        self.advanced_visible = tk.BooleanVar(value=False)
        toggle_btn = ttk.Checkbutton(
            self,
            text="Show Advanced Settings",
            variable=self.advanced_visible,
            command=self.toggle_advanced_settings,
            onvalue=True,
            offvalue=False
        )
        toggle_btn.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="w")

        # Advanced settings frame (initially hidden)
        self.advanced_frame = ttk.LabelFrame(self, text="Advanced Settings")

        # ── Pick Strategy ───────────────────────────────────────────────────────────
        strategy_frame = ttk.LabelFrame(self.advanced_frame, text="Pick Strategy")
        strategy_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        strategy_label = ttk.Label(strategy_frame, text="Strategy:", font=bigger_font)
        strategy_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        strategy_dropdown = ttk.Combobox(
            strategy_frame,
            textvariable=self.pick_strategy_var,
            values=STRATEGIES,
            font=bigger_font,
            state="readonly",
            width=10
        )
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ── Display Metric ──────────────────────────────────────────────────────────
        display_metric_frame = ttk.LabelFrame(self.advanced_frame, text="Display Metric")
        display_metric_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        display_metric_label = ttk.Label(display_metric_frame, text="Sort By:", font=bigger_font)
        display_metric_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        display_metric_dropdown = ttk.Combobox(
            display_metric_frame,
            textvariable=self.display_metric_var,
            values=["Win Rate", "Delta"],
            font=bigger_font,
            state="readonly"
        )
        display_metric_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ── Adjustment Method ───────────────────────────────────────────────────────
        adjustment_frame = ttk.LabelFrame(self.advanced_frame, text="Adjustment Method")
        adjustment_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")
        adjustment_label = ttk.Label(adjustment_frame, text="Adjustment:", font=bigger_font)
        adjustment_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        adjustment_dropdown = ttk.Combobox(
            adjustment_frame,
            textvariable=self.adjustment_method,
            values=["Bayesian", "ADVI", "Hierarchical"],
            font=bigger_font,
            state="readonly"
        )
        adjustment_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ── Bayesian “m” Value ───────────────────────────────────────────────────────
        m_frame = ttk.LabelFrame(self.advanced_frame, text="Bayesian Adjustment (m)")
        m_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nw")
        m_label = ttk.Label(m_frame, text="Set m:", font=bigger_font)
        m_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.m_var = tk.IntVar(value=0)
        m_entry = ttk.Entry(m_frame, textvariable=self.m_var, width=10, font=bigger_font)
        m_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        m_button = ttk.Button(m_frame, text="Recalculate", command=self.recalculate_matchups)
        m_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # ── TEAMS FRAME ───────────────────────────────────────────────────────────────
        teams_frame = ttk.Frame(self)
        teams_frame.grid(row=2, column=0, columnspan=2, sticky="nw", padx=10, pady=10)

        # ── ALLY PICKS FRAME ───────────────────────────────────────────────────────────
        ally_frame = ttk.LabelFrame(teams_frame, text="Ally Team (roles known)")
        ally_frame.grid(row=0, column=0, padx=10, sticky="nw")

        self.ally_champs = {}
        for i, role in enumerate(self.roles_ally):
            lbl = ttk.Label(ally_frame, text=f"{role.capitalize()}:", font=bigger_font)
            lbl.grid(row=i, column=0, sticky="w")

            entry = AutocompleteEntryPopup(
                ally_frame,
                suggestion_list=self.champion_list,
                width=12,
                font=bigger_font,
                callback=self.combined_callback
            )
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.ally_champs[role] = entry

        # ── ENEMY PICKS FRAME ───────────────────────────────────────────────────────────
        enemy_frame = ttk.LabelFrame(teams_frame, text="Enemy Team (champions only)")
        enemy_frame.grid(row=0, column=1, padx=10, sticky="nw")

        self.enemy_champ_boxes = []
        self.enemy_role_labels = []
        for i in range(5):
            lbl = ttk.Label(enemy_frame, text=f"Enemy #{i+1}:", font=bigger_font)
            lbl.grid(row=i, column=0, sticky="w")

            entry = AutocompleteEntryPopup(
                enemy_frame,
                suggestion_list=self.champion_list,
                width=12,
                font=bigger_font,
                callback=self.combined_callback
            )
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.enemy_champ_boxes.append(entry)

            role_lbl = ttk.Label(enemy_frame, text="", font=bigger_font, foreground="blue")
            role_lbl.grid(row=i, column=2, padx=(5, 10), sticky="w")
            self.enemy_role_labels.append(role_lbl)

        self.enemy_guess_label = ttk.Label(
            enemy_frame,
            text="",
            justify="left",
            font=bigger_font
        )
        self.enemy_guess_label.grid(
            row=0, column=2, rowspan=len(self.enemy_champ_boxes),
            sticky="nw", padx=(10, 0)
        )

        # ── SUGGESTED PICKS (“Results”) ───────────────────────────────────────────────
        pick_frame = ttk.LabelFrame(self, text="Suggested Picks")
        pick_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        btn_reset = ttk.Button(pick_frame, text="Reset", command=self.reset_all)
        btn_reset.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        # The results_frame will contain five LabelFrames (one per role), arranged left→right
        self.results_frame = ttk.Frame(pick_frame)
        self.results_frame.grid(row=2, column=0, columnspan=3, sticky="ew")

        self.icon_frames = {}
        for i, role in enumerate(self.roles_ally):
            container = ttk.LabelFrame(self.results_frame, text=role.capitalize())
            container.grid(row=0, column=i, padx=10, pady=5, sticky="nw")
            self.icon_frames[role] = {'container': container, 'icons': []}

            # Add a Copy button under each role's LabelFrame
            copy_btn = ttk.Button(
                container,
                text="Copy",
                command=lambda r=role: self.copy_role_list(r)
            )
            copy_btn.grid(row=6, column=0, columnspan=2, pady=(5, 0), sticky="ew")

        # ── OVERALL WIN RATE LABEL ────────────────────────────────────────────────────
        self.overall_win_rate_label = ttk.Label(self, text="", justify="left", font=bigger_font)
        self.overall_win_rate_label.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nw")
