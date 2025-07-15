import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from math import log, exp
from scipy.optimize import linear_sum_assignment
import pyperclip
import os
from PIL import Image, ImageTk
import time
import base64
import requests
import urllib3
import json
import threading

urllib3.disable_warnings()


###############################################################################
# 1) Load synergy/counter data
###############################################################################
def load_matchup_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

###############################################################################
# 2) Load champion priors (and full champion list) for Bayesian role-guessing
###############################################################################


def load_champion_priors(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

###############################################################################
# 3) Hungarian-based guess of roles for enemy champions
###############################################################################


def guess_roles(champ_list, priors_df):
    """
    champ_list: list of championName (strings), e.g. ['Gragas', 'Yasuo', ...].
    priors_df:  DataFrame with columns [champion_name, top, jungle, middle, bottom, support].

    Returns: dict { "top": champName, "jungle": champName, ... }
             by maximizing prior probabilities.
    """
    known = {}
    unknown = champ_list
    if not unknown:
        return known

    roles = ["top", "jungle", "middle", "bottom", "support"]
    n, m = len(unknown), len(roles)
    cost_matrix = np.zeros((n, m))

    for i, champ in enumerate(unknown):
        row = priors_df[priors_df['champion_name'].str.lower() == champ.lower()]
        probs = [1.0 / m] * m if row.empty else [float(row[role].iloc[0]) for role in roles]

        for j in range(m):
            p = probs[j]
            cost_matrix[i, j] = 9999.0 if p <= 0 else -log(p)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for i in range(len(row_ind)):
        champ = unknown[row_ind[i]]
        role = roles[col_ind[i]]
        known[role] = champ

    return known


###############################################################################
# 4) Synergy/Delta scoring
###############################################################################


def prepare_multiindex(df):
    # Create a multi-index based on columns commonly used in lookups.
    df_indexed = df.set_index(['champ1', 'role1', 'type', 'champ2', 'role2'])
    return df_indexed.sort_index()  # Sort the index for optimal performance


def log_odds_to_probability(log_odds):
    return 1 / (1 + exp(-log_odds))


def win_rate_to_log_odds(win_rate):
    # Convert percentage to probability
    p = win_rate / 100.0
    # Ensure p is not 0 or 1 to avoid infinite log values.
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return log(p / (1 - p))


def calculate_overall_win_rates(df_indexed, ally_team, enemy_team):
    # Sum log_odds for one team
    ally_log_odds = calculate_team_log_odds(df_indexed, ally_team, enemy_team)

    # Ally win probability, enemy win probability is the complement
    ally_win_rate = log_odds_to_probability(ally_log_odds)
    enemy_win_rate = 1 - ally_win_rate
    return ally_win_rate, enemy_win_rate


def calculate_team_log_odds(df_indexed, team, opponent_team):
    total_log_odds = 0.0

    # 1. Ally Synergy using unique pairs with reverse lookup
    ally_roles = list(team.keys())
    for i in range(len(ally_roles)):
        for j in range(i + 1, len(ally_roles)):
            role_i = ally_roles[i]
            role_j = ally_roles[j]
            champ_i = team[role_i]
            champ_j = team[role_j]

            synergy_value = None
            # Attempt forward lookup
            try:
                synergy_value = df_indexed.loc[
                    (champ_i, role_i, 'Synergy', champ_j, role_j),
                    'log_odds'
                ]
            except KeyError:
                # Attempt reverse lookup if forward fails
                try:
                    synergy_value = df_indexed.loc[
                        (champ_j, role_j, 'Synergy', champ_i, role_i),
                        'log_odds'
                    ]
                except KeyError:
                    synergy_value = None

            if synergy_value is not None:
                added_value = synergy_value.sum() if hasattr(
                    synergy_value, 'sum') else synergy_value
                total_log_odds += added_value

    # 2. Counters against enemy with reverse lookup and complement
    for role in team.keys():
        champ = team[role]
        for enemy_role, enemy_champ in opponent_team.items():
            counter_value = None
            # Attempt forward lookup for counter (champion A counters enemy)
            try:
                counter_value = df_indexed.loc[
                    (champ, role, 'Counter', enemy_champ, enemy_role),
                    'log_odds'
                ]
                total_log_odds += counter_value.sum() if hasattr(counter_value,
                                                                 'sum') else counter_value
            except KeyError:
                # If forward lookup fails, attempt reverse lookup: enemy counters champion
                try:
                    counter_value = df_indexed.loc[
                        (enemy_champ, enemy_role, 'Counter', champ, role),
                        'log_odds'
                    ]
                    # Subtract the value since this is disadvantageous (the complement)
                    total_log_odds -= counter_value.sum() if hasattr(counter_value,
                                                                     'sum') else counter_value
                except KeyError:
                    pass

    # 3. Subtract Enemy Synergy Effects with reverse lookup
    enemy_roles = list(opponent_team.keys())
    if enemy_roles:
        for i in range(len(enemy_roles)):
            for j in range(i + 1, len(enemy_roles)):
                role_i = enemy_roles[i]
                role_j = enemy_roles[j]
                champ_i = opponent_team[role_i]
                champ_j = opponent_team[role_j]

                enemy_synergy = None
                # Attempt forward lookup for enemy synergy
                try:
                    enemy_synergy = df_indexed.loc[
                        (champ_i, role_i, 'Synergy', champ_j, role_j),
                        'log_odds'
                    ]
                except KeyError:
                    # Attempt reverse lookup if forward fails
                    try:
                        enemy_synergy = df_indexed.loc[
                            (champ_j, role_j, 'Synergy', champ_i, role_i),
                            'log_odds'
                        ]
                    except KeyError:
                        enemy_synergy = None

                if enemy_synergy is not None:
                    subtracted_value = enemy_synergy.sum() if hasattr(
                        enemy_synergy, 'sum') else enemy_synergy
                    total_log_odds -= subtracted_value

    return total_log_odds

###############################################################################
# 5) get_champion_scores_for_role, with "excluded" filter
###############################################################################


def get_champion_scores_for_role(
    df_indexed,
    role_to_fill,
    ally_team,
    enemy_team,
    pick_strategy="Maximize",
    champion_pool=None,
    excluded_champions=None
):
    """
    Returns a list of (championName, sum_log_odds, sum_delta).

    Now includes 'excluded_champions' to filter out any picks that are already taken or banned.
    """

    if champion_pool is None:
        champion_pool = []
    if excluded_champions is None:
        excluded_champions = set()

    # Figure out which roles the enemy hasn't filled
    all_roles = ["top", "jungle", "middle", "bottom", "support"]
    enemy_filled_roles = set(enemy_team.keys())
    unfilled_enemy_roles = [
        r for r in all_roles if r not in enemy_filled_roles]

    ###########################################################################
    # 1) Find all candidate champions that can plausibly fill 'role_to_fill'
    ###########################################################################
    try:
        # Filter the indexed DataFrame where:
        #   champ1 is anything
        #   role1 == role_to_fill
        #   type can be Synergy or Counter
        #   champ2 is anything
        #   role2 is anything
        subset = df_indexed.loc[(slice(None), role_to_fill, slice(
            None), slice(None), slice(None)), :]
    except KeyError:
        subset = pd.DataFrame()

    if subset.empty:
        # No known synergy/counter data for this role
        return []

    all_role_candidates = subset.reset_index()["champ1"].unique().tolist()
    # Also ensure we include anything from champion_pool if needed:
    all_role_candidates = sorted(
        list(set(all_role_candidates + champion_pool)))

    # Filter out champions that are excluded (already picked or banned)
    candidates = [
        champ for champ in all_role_candidates
        if champ not in excluded_champions
    ]

    ###########################################################################
    # 2) Helper function for synergy if "candidate_champ" is in role_to_fill
    ###########################################################################
    def synergy_for_candidate(candidate_champ, ally_team, enemy_team):
        """
        Original synergy/counter logic from your snippet, returning
        (total_log_odds, total_delta) for candidate_champ given known ally and enemy picks.
        """
        total_log_odds = 0.0
        total_delta = 0.0

        # 1) Synergy with allies
        for a_role, a_champ in ally_team.items():
            synergy_row = None
            # forward synergy
            try:
                synergy_row = df_indexed.loc[
                    (candidate_champ, role_to_fill, 'Synergy', a_champ, a_role)
                ]
            except KeyError:
                synergy_row = None
            # reverse synergy if forward missing
            if (synergy_row is None) or synergy_row.empty:
                try:
                    synergy_row = df_indexed.loc[
                        (a_champ, a_role, 'Synergy', candidate_champ, role_to_fill)
                    ]
                except KeyError:
                    synergy_row = None

            if synergy_row is not None and not synergy_row.empty:
                synergy_value = synergy_row['log_odds']
                delta_value = synergy_row.get('delta_shrunk_bayes', 0.0)
                total_log_odds += synergy_value.sum()
                total_delta += delta_value.sum()

        # 2) Counters vs enemy
        for e_role, e_champ in enemy_team.items():
            # forward (candidate_champ counters e_champ)
            try:
                counter_row = df_indexed.loc[
                    (candidate_champ, role_to_fill, 'Counter', e_champ, e_role)
                ]
                if counter_row is not None and not counter_row.empty:
                    total_log_odds += counter_row['log_odds'].sum()
                    total_delta += counter_row.get(
                        'delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

            # reverse (e_champ counters candidate_champ) => subtract
            try:
                reverse_row = df_indexed.loc[
                    (e_champ, e_role, 'Counter', candidate_champ, role_to_fill)
                ]
                if reverse_row is not None and not reverse_row.empty:
                    total_log_odds -= reverse_row['log_odds'].sum()
                    total_delta -= reverse_row.get(
                        'delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

        return total_log_odds, total_delta

    ###########################################################################
    # 3) Helper function for synergy if "enemy_candidate" is placed
    #    in some unfilled role on the enemy team
    ###########################################################################
    def synergy_for_enemy_candidate(enemy_champ, enemy_role, enemy_team, ally_team):
        total_log_odds = 0.0
        total_delta = 0.0

        # 1) Synergy with existing enemy picks
        for r_exist, ch_exist in enemy_team.items():
            if r_exist == enemy_role:
                # skip the role we are about to fill, so we don't double-count
                continue

            synergy_row = None
            # forward synergy
            try:
                synergy_row = df_indexed.loc[
                    (enemy_champ, enemy_role, 'Synergy', ch_exist, r_exist)
                ]
            except KeyError:
                synergy_row = None
            # reverse synergy if forward not found
            if (synergy_row is None) or synergy_row.empty:
                try:
                    synergy_row = df_indexed.loc[
                        (ch_exist, r_exist, 'Synergy', enemy_champ, enemy_role)
                    ]
                except KeyError:
                    synergy_row = None

            if synergy_row is not None and not synergy_row.empty:
                synergy_value = synergy_row['log_odds']
                delta_value = synergy_row.get('delta_shrunk_bayes', 0.0)
                total_log_odds += synergy_value.sum()
                total_delta += delta_value.sum()

        # 2) Counter vs ally
        for a_role, a_champ in ally_team.items():
            # forward (enemy_candidate counters ally champ)
            try:
                counter_row = df_indexed.loc[
                    (enemy_champ, enemy_role, 'Counter', a_champ, a_role)
                ]
                if counter_row is not None and not counter_row.empty:
                    total_log_odds += counter_row['log_odds'].sum()
                    total_delta += counter_row.get(
                        'delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

            # reverse (ally champ counters this new enemy pick) => subtract
            try:
                reverse_row = df_indexed.loc[
                    (a_champ, a_role, 'Counter', enemy_champ, enemy_role)
                ]
                if reverse_row is not None and not reverse_row.empty:
                    total_log_odds -= reverse_row['log_odds'].sum()
                    total_delta -= reverse_row.get(
                        'delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

        return total_log_odds, total_delta

    ###########################################################################
    # 4) Main loop over each candidate champion
    ###########################################################################
    champion_scores = {}

    for candidate_champ in candidates:
        # **Simulate picking this candidate champion for the role**
        old_ally_pick = ally_team.get(role_to_fill, None)
        ally_team[role_to_fill] = candidate_champ

        # A) Compute synergy of "candidate_champ" with ally_team vs enemy_team
        sum_log_odds, sum_delta = synergy_for_candidate(
            candidate_champ, ally_team, enemy_team)

        if pick_strategy == "Maximize":
            final_log_odds = sum_log_odds
            final_delta = sum_delta

        elif pick_strategy == "MinimaxAllRoles":
            worst_log_odds = float("-inf")
            worst_delta = float("-inf")

            if len(unfilled_enemy_roles) == 0:
                worst_log_odds = 0.0
                worst_delta = 0.0
            else:
                # For each unfilled enemy role:
                for e_role in unfilled_enemy_roles:
                    old_enemy_pick = enemy_team.get(e_role, None)

                    for e_candidate in champion_pool:
                        # Skip any e_candidate that is also excluded for the enemy
                        # (But typically the enemy can pick it. So there's no "excluded" for enemy, unless you want it.)
                        enemy_team[e_role] = e_candidate

                        # Evaluate synergy from the enemy perspective
                        e_log_odds, e_delta = synergy_for_enemy_candidate(
                            e_candidate, e_role, enemy_team, ally_team)

                        if e_log_odds > worst_log_odds:
                            worst_log_odds = e_log_odds
                        if e_delta > worst_delta:
                            worst_delta = e_delta

                    # revert enemy pick after evaluating all candidates for this role
                    if old_enemy_pick is not None:
                        enemy_team[e_role] = old_enemy_pick
                    else:
                        enemy_team.pop(e_role, None)

            final_log_odds = sum_log_odds - worst_log_odds
            final_delta = sum_delta - worst_delta

        else:
            final_log_odds = sum_log_odds
            final_delta = sum_delta

        # Revert the ally_team to its previous state after evaluating this candidate
        if old_ally_pick is not None:
            ally_team[role_to_fill] = old_ally_pick
        else:
            ally_team.pop(role_to_fill, None)

        champion_scores[candidate_champ] = (final_log_odds, final_delta)

    ###########################################################################
    # 5) Build the final list of three-tuples
    ###########################################################################
    result = [
        (champ, vals[0], vals[1])
        for champ, vals in champion_scores.items()
    ]
    return result


###############################################################################
# 6) Custom Entry + Popup Listbox Autocomplete
###############################################################################
class AutocompleteEntryPopup(tk.Frame):
    """
    A custom widget with:
      - A tk.Entry for user input
      - A popup tk.Toplevel with a tk.Listbox of suggestions
    """

    def __init__(self, master, suggestion_list=None, width=30, font=None, callback=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.callback = callback
        self.suggestion_list = suggestion_list or []
        self.current_suggestions = []
        self.current_index = -1

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            self, textvariable=self.entry_var, width=width, font=font)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

        # Bind entry events
        self.entry.bind("<KeyRelease>", self._on_keyrelease)
        self.entry.bind("<Down>", self._on_down_arrow)
        self.entry.bind("<Up>", self._on_up_arrow)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Tab>", self._on_tab_press)
        self.entry.bind("<FocusOut>", self._on_focus_out)

        self.popup = None

    def _update_background(self):
        text = self.entry_var.get().strip().lower()
        if text in getattr(self, "normalized_suggestions", set()):
            self.entry.configure(bg="white")
        else:
            self.entry.configure(bg="#ffd6d6")

    def _on_keyrelease(self, event):
        if event.keysym in ("Up", "Down", "Left", "Right", "Return", "Tab", "Escape"):
            return

        text = self.entry_var.get().strip().lower()

        if not text:
            matches = list(self.suggestion_list)
            if matches:
                self._show_popup(matches)
            else:
                self._hide_popup()
        else:
            matches = self._filter_suggestions(text)
            if matches:
                self._show_popup(matches)
            else:
                self._hide_popup()

        self._update_background()

        # Notify UI about text change so it can refresh roles, etc.
        if self.callback:
            self.callback()

    def _show_popup(self, suggestions):
        self.current_suggestions = suggestions
        self.current_index = 0

        if not self.popup or not tk.Toplevel.winfo_exists(self.popup):
            self.popup = tk.Toplevel(self)
            self.popup.wm_overrideredirect(True)

            frame = tk.Frame(self.popup)
            frame.pack(fill="both", expand=True)

            scrollbar = tk.Scrollbar(frame, orient="vertical")
            scrollbar.pack(side="right", fill="y")

            self.listbox = tk.Listbox(
                frame, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set)
            self.listbox.pack(side="left", fill="both", expand=True)

            scrollbar.config(command=self.listbox.yview)

            self.listbox.bind("<Button-1>", self._on_listbox_click)
            self.listbox.bind("<Return>", self._on_return)
            self.listbox.bind("<Down>", self._on_down_arrow)
            self.listbox.bind("<Up>", self._on_up_arrow)

        # Update popup position every time
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()

        # Limit visible suggestions height and listbox width
        MAX_VISIBLE_ITEMS = 8
        MAX_WIDTH_CHARS = 30

        height = min(len(suggestions), MAX_VISIBLE_ITEMS)
        self.listbox.config(height=height)

        max_suggestion_len = max((len(s) for s in suggestions), default=0)
        width = min(max_suggestion_len, MAX_WIDTH_CHARS)
        self.listbox.config(width=width)

        self.popup.geometry(f"+{x}+{y}")

        # Update listbox contents efficiently
        self.listbox.delete(0, tk.END)
        for item in suggestions:
            self.listbox.insert(tk.END, item)

        # Update selection
        self._wrap_index()
        self._update_listbox_selection()

    def _hide_popup(self):
        if self.popup and tk.Toplevel.winfo_exists(self.popup):
            self.popup.destroy()
        self.popup = None
        self.current_suggestions = []
        self.current_index = -1

    def _on_listbox_click(self, event):
        idx = self.listbox.curselection()
        if idx:
            self.current_index = idx[0]
            self._select_current()
        self._hide_popup()
        self.entry.focus_set()
        return "break"

    def _on_down_arrow(self, event):
        if not self.popup:
            return
        self.current_index += 1
        self._wrap_index()
        self._update_listbox_selection()
        return "break"

    def _on_up_arrow(self, event):
        if not self.popup:
            return
        self.current_index -= 1
        self._wrap_index()
        self._update_listbox_selection()
        return "break"

    def _on_return(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            self.entry.focus_set()  # Focus back on the entry widget
            self._move_to_next_widget()
            return "break"
        else:
            matches = self._filter_suggestions(self.entry_var.get().strip())
            if len(matches) == 1:
                self._set_text(matches[0])
                self._move_to_next_widget()

    def _on_tab_press(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            self._move_to_next_widget()
            return "break"

    def _move_to_next_widget(self):
        next_widget = self.entry.tk_focusNext()
        if next_widget:
            next_widget.focus_set()

    def _on_focus_out(self, event):
        if self.popup:
            widget = self.winfo_containing(
                self.winfo_pointerx(), self.winfo_pointery())
            if not (widget and str(widget).startswith(str(self.popup))):
                self._hide_popup()

    def _select_current(self):
        self._wrap_index()
        if 0 <= self.current_index < len(self.current_suggestions):
            self._set_text(self.current_suggestions[self.current_index])

    def _filter_suggestions(self, typed):
        low = typed.lower()
        return [s for s in self.suggestion_list if low in s.lower()]

    def _set_text(self, text):
        self.entry_var.set(text)
        if self.callback:
            self.callback()

    def _wrap_index(self):
        """
        Wrap current_index around if it goes out of bounds
        """
        count = len(self.current_suggestions)
        if count == 0:
            self.current_index = -1
        else:
            self.current_index %= count

    def _update_listbox_selection(self):
        """
        Reflect current_index in the listbox UI
        """
        self.listbox.select_clear(0, tk.END)
        if 0 <= self.current_index < len(self.current_suggestions):
            self.listbox.select_set(self.current_index)
            self.listbox.activate(self.current_index)
            # Scroll to make the selection visible
            self.listbox.see(self.current_index)

    def update_suggestions(self, new_suggestion_list):
        valid_pick = self.get_valid_text().strip()
        if valid_pick and valid_pick not in new_suggestion_list:
            new_suggestion_list = list(new_suggestion_list) + [valid_pick]

        # Dedupe & sort
        self.suggestion_list = sorted(set(new_suggestion_list))

        # Cache normalized suggestions (lowercase stripped)
        self.normalized_suggestions = set(
            s.lower().strip() for s in self.suggestion_list)

        # Refresh popup
        if self.popup:
            text = self.entry_var.get().strip()
            matches = self._filter_suggestions(
                text) if text else list(self.suggestion_list)
            if matches:
                self._show_popup(matches)
            else:
                self._hide_popup()

    def get_valid_text(self):
        text = self.entry_var.get().strip()
        return text if text in self.suggestion_list else ""

    def get_text(self):
        return self.entry_var.get()

    def _set_text(self, text):
        self.entry_var.set(text)
        if self.callback:
            self.callback()
        self._update_background()

    def set_text(self, text):
        """Public method to set text from outside the widget."""
        self._set_text(text)

###############################################################################
# 7) The main GUI (ChampionPickerGUI)
###############################################################################


class ChampionPickerGUI(tk.Tk):
    ICON_PATH_FORMAT = os.path.join("data", "champion_icons", "{}.png")
    ICON_SIZE = (64, 64)

    def __init__(self, df_matchups, df_priors):
        super().__init__()
        self.geometry("1280x920")
        self.minsize(700, 500)

        # Roles on the ally side
        self.roles_ally = ["top", "jungle", "middle", "bottom", "support"]

        self.title("League Champion Picker")

        # Store and index matchups DataFrame
        self.df_matchups = df_matchups
        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)
        self.df_priors = df_priors

        # Champion list (for autocomplete + icons)
        self.champion_list = list(self.df_priors['champion_name'].unique())

        # banned champ list
        self.banned_champions_names = list()
        self.ban_entries = []

        # Pre-load champion icons (regular size)
        self.champion_icons = {}
        # New: Pre-load ban‐size icons
        self.banned_champion_icons = {}

        for champ in self.champion_list:
            path = ChampionPickerGUI.ICON_PATH_FORMAT.format(champ)
            if os.path.exists(path):
                pil = Image.open(path)
                # regular
                pil_reg = pil.resize(
                    ChampionPickerGUI.ICON_SIZE, Image.LANCZOS)
                self.champion_icons[champ] = ImageTk.PhotoImage(pil_reg)
                # small for bans (half size, for example)
                small_size = (
                    ChampionPickerGUI.ICON_SIZE[0]//2, ChampionPickerGUI.ICON_SIZE[1]//2)
                pil_small = pil.resize(small_size, Image.LANCZOS)
                self.banned_champion_icons[champ] = ImageTk.PhotoImage(
                    pil_small)
            else:
                print(f"⚠️ Icon not found for '{champ}' at {path}")
                self.champion_icons[champ] = None
                self.banned_champion_icons[champ] = None

        # StringVars / BooleanVars for settings
        self.adjustment_method = tk.StringVar(value="Bayesian")
        self.display_metric_var = tk.StringVar(value="Delta")
        self.pick_strategy_var = tk.StringVar(value="Maximize")
        self.auto_hide = tk.BooleanVar(value=True)

        # Build the UI
        self._build_ui()

    def combined_callback(self):
        """
        Called whenever any AutocompleteEntryPopup changes.
        Recomputes hiding logic, overall win rates, and recommendations.
        """
        self.check_filled_roles()
        self.update_overall_win_rates()
        self.on_recommend()
        self.update_autocomplete_suggestions()

    def toggle_advanced_settings(self):
        """
        Show or hide the advanced settings frame based on self.advanced_visible.
        """
        if self.advanced_visible.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def recalculate_matchups(self):
        """
        Recompute Bayesian-shrunk win rates (and optionally Δ), re-index DataFrame,
        save to CSV, then update overall win rates and recommendations.
        """
        try:
            m_value = self.m_var.get()
            # Shrink win_rate toward 0.5*100% by m_value "pseudo-samples"
            self.df_matchups['win_rate_shrunk_bayes'] = (
                (self.df_matchups['win_rate'] * self.df_matchups['sample_size'] +
                 50.0 * m_value) / (self.df_matchups['sample_size'] + m_value)
            )
            self.df_matchups['log_odds_bayes'] = self.df_matchups['win_rate_shrunk_bayes'].apply(
                win_rate_to_log_odds)

            if 'delta' in self.df_matchups.columns:
                self.df_matchups['delta_shrunk_bayes'] = (
                    (self.df_matchups['delta'] * self.df_matchups['sample_size'] +
                     0.0 * m_value) /
                    (self.df_matchups['sample_size'] + m_value)
                )

            # Re-index
            self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

            # Save relevant columns
            desired_columns = [
                "champ1", "role1", "type", "champ2", "role2",
                "win_rate", "sample_size",
                "win_rate_shrunk_bayes", "log_odds_bayes",
                "win_rate_shrunk_advi", "log_odds_advi",
                "win_rate_shrunk_hierarchical", "log_odds_hierarchical",
            ]
            if 'delta_shrunk_bayes' in self.df_matchups.columns:
                desired_columns.append('delta_shrunk_bayes')
            if 'delta' in self.df_matchups.columns:
                desired_columns.append('delta')

            columns_to_save = [
                c for c in desired_columns if c in self.df_matchups.columns]
            self.df_matchups.to_csv(
                "data/matchups_shrunk.csv", columns=columns_to_save, index=False)

            # Refresh UI
            self.update_overall_win_rates()
            self.on_recommend()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to recalculate matchups: {e}")

    def update_overall_win_rates(self):
        """
        Recompute ally vs. enemy team win rates and update the label.
        """
        method = self.adjustment_method.get().lower()
        log_col = f'log_odds_{method}'
        if log_col in self.df_matchups.columns:
            self.df_matchups['log_odds'] = self.df_matchups[log_col]
        else:
            self.df_matchups['log_odds'] = self.df_matchups['log_odds_bayes']

        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

        ally_team = {
            r: e.get_valid_text().strip()
            for r, e in self.ally_champs.items()
            if e.get_valid_text().strip()
        }
        enemy_list = [e.get_valid_text().strip()
                      for e in self.enemy_champ_boxes if e.get_valid_text().strip()]
        enemy_team = guess_roles(enemy_list, self.df_priors)

        ally_pct, enemy_pct = calculate_overall_win_rates(
            self.df_matchups_indexed, ally_team, enemy_team
        )
        text = (
            f"Estimated Ally Team Win Rate: {ally_pct:.2%}\n"
            f"Estimated Enemy Team Win Rate: {enemy_pct:.2%}"
        )
        self.overall_win_rate_label.config(text=text)

    def check_filled_roles(self):
        """
        Called any time ally‐champ text changes OR the auto‐hide checkbox toggles.
        We compute “should this role be visible?” for every role, then let
        rearrange_result_icons() do all the actual grid/grid_forget calls.
        """
        # 1) For each role, decide visible=True/False and store it somewhere.
        #    Visible = “either auto_hide is OFF, or champ name is blank/invalid”.
        for role, entry_widget in self.ally_champs.items():
            champ_name = entry_widget.get_valid_text().strip()
            is_valid_champion = (champ_name in self.champion_list)

            entry_widget.bind(
                "<KeyRelease>", lambda e: self.check_filled_roles())
            self.auto_hide.trace_add(
                'write', lambda *args: self.check_filled_roles())

            # If auto_hide is True and the user has typed a VALID champ, hide it.
            # Otherwise (auto_hide=False, or name blank/invalid), show it.
            should_show = (not (is_valid_champion and self.auto_hide.get()))

            # Store this boolean in a simple dict; rearrange_result_icons() will read it.
            self.icon_frames[role]['should_be_visible'] = should_show

        # 2) Now reposition ALL containers in one sweep.
        self.rearrange_result_icons()

    def toggle_role_visibility(self, role, visible):
        """
        If auto_hide is True and visible is False, remove that role’s container
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
        Forget (i.e. remove) all frames, then re‐grid only those where
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
        # ── Update log_odds in df_matchups ───────────────────────────────────
        method = self.adjustment_method.get().lower()
        log_col = f'log_odds_{method}'
        if log_col in self.df_matchups.columns:
            self.df_matchups['log_odds'] = self.df_matchups[log_col]
        else:
            self.df_matchups['log_odds'] = self.df_matchups['log_odds_bayes']

        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

        # ── Build ally_team dict using get_valid_text() ───────────────────────
        typed_ally = {}  # directly assigned by user
        typed_ally_champs = []

        for role, entry in self.ally_champs.items():
            nm = entry.get_valid_text()
            if nm:
                typed_ally[role] = nm
                typed_ally_champs.append(nm)

        # If not all 5 roles are filled, guess the missing ones
        if len(typed_ally) < 5:
            # Use only typed champions for guessing
            guessed_roles = guess_roles(typed_ally_champs, self.df_priors)

            # Merge guesses into typed_ally, without overwriting already typed roles
            for role, champ in guessed_roles.items():
                if role not in typed_ally:
                    typed_ally[role] = champ

        ally_team = typed_ally


        # ── Gather enemy champions and guess roles ────────────────────────────
        enemy_champs = [
            e.get_valid_text() for e in self.enemy_champ_boxes if e.get_valid_text()
        ]
        enemy_team = guess_roles(enemy_champs, self.df_priors)

        # Display “Akshan → middle” etc.
        guessed_text = ""
        for role, champ in enemy_team.items():
            guessed_text += f"{champ} → {role}\n"
        self.enemy_guess_label.config(text=guessed_text)

        chosen_metric = self.display_metric_var.get()
        excluded_base = (
            set(enemy_team.values())
            | set(self.banned_champions_names)
        )

        # Ensure the Suggested Picks area is visible before drawing
        self.results_frame.grid()
        self.rearrange_result_icons()

        # ── For each role, compute top-5 and build a vertical list ───────────
        for role in self.roles_ally:
            # Clear out previous widgets under this role
            for widget in self.icon_frames[role]['icons']:
                widget.destroy()
            self.icon_frames[role]['icons'].clear()

            # Compute scores for this role
            excluded_dynamic = set(excluded_base)
            scores = get_champion_scores_for_role(
                df_indexed=self.df_matchups_indexed,
                role_to_fill=role,
                ally_team=ally_team,
                enemy_team=enemy_team,
                pick_strategy=(
                    "MinimaxAllRoles"
                    if self.pick_strategy_var.get().startswith("Minimax")
                    else "Maximize"
                ),
                champion_pool=self.champion_list,
                excluded_champions=excluded_dynamic
            )

            # Sort by Delta or WinRate
            if chosen_metric == "Delta":
                scores.sort(key=lambda x: x[2], reverse=True)
            else:
                scores.sort(key=lambda x: x[1], reverse=True)

            top_n = scores[:5]  # show top 5

            # Place each champion’s icon + W/Δ in a vertical stack
            for idx, (champ, total_log_odds, total_delta) in enumerate(top_n):
                photo = self.champion_icons.get(champ)

                subframe = ttk.Frame(self.icon_frames[role]['container'])
                subframe.grid(row=idx, column=0, padx=2, pady=2, sticky="w")
                subframe.champ_name = champ

                # Icon
                if photo is not None:
                    icon_lbl = ttk.Label(subframe, image=photo)
                    icon_lbl.image = photo
                    icon_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")
                else:
                    text_lbl = ttk.Label(
                        subframe, text=champ, font=("Helvetica", 10))
                    text_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")

                win_pct = 100 * total_log_odds
                delta_pct = 10 * total_delta

                # Stats
                text_frame = ttk.Frame(subframe)
                text_frame.grid(row=0, column=1, sticky="nw")
                win_lbl = ttk.Label(
                    text_frame, text=f"W: {win_pct:.0f}%", font=("Helvetica", 10))
                win_lbl.pack(anchor="w")
                delta_lbl = ttk.Label(
                    text_frame, text=f"Δ: {delta_pct:.1f}", font=("Helvetica", 10))
                delta_lbl.pack(anchor="sw")

                self.icon_frames[role]['icons'].append(subframe)

                if idx == 0:
                    excluded_base.add(champ)

            # Re-apply auto-hide logic
            self.check_filled_roles()

            self.update_autocomplete_suggestions()

    def reset_all(self):
        """
        Clear all entry fields, clear bans, and reset UI.
        """
        # For ally champs (dict)
        for entry in self.ally_champs.values():
            if hasattr(entry, "set_text"):
                entry.set_text("")
            else:
                entry.entry_var.set("")

        # For enemy champs (list)
        for entry in self.enemy_champ_boxes:
            if hasattr(entry, "set_text"):
                entry.set_text("")
            else:
                entry.entry_var.set("")

        # 3) Clear ban entry widgets
        for entry in self.ban_entries:
            entry.entry_var.set("")

        # 4) Clear banned champion names
        self.banned_champions_names.clear()
        if hasattr(self, "banned_champion_ids"):
            self.banned_champion_ids.clear()

        # 5) Clear ban icon labels
        for lbl in self.banned_icon_labels:
            lbl.configure(image="")
            lbl.image = None

        # 6) Re-run autocomplete filtering
        self.update_autocomplete_for_bans()

        # 7) Destroy existing result icon subframes
        for role in self.roles_ally:
            for widget in self.icon_frames[role]['icons']:
                widget.destroy()
            self.icon_frames[role]['icons'].clear()

        # 8) Clear derived labels
        self.enemy_guess_label.config(text="")
        self.overall_win_rate_label.config(text="")

        # 9) Refresh recommendation state
        self.check_filled_roles()
        self.update_overall_win_rates()

    def get_banned_champions(self):
        return [e.get_valid_text() for e in self.ban_entries if e.get_valid_text()]

    def update_autocomplete_for_bans(self):
        filtered_list = [
            champ for champ in self.champion_list if champ not in self.banned_champions_names]

        for entry in self.ally_champs.values():
            entry.update_suggestions(filtered_list)

        for entry in self.enemy_champ_boxes:
            entry.update_suggestions(filtered_list)

    def update_gui_bans(self, new_banned_names):
        # 1) Add only the *new* bans that aren’t already in the list:
        for name in new_banned_names:
            if name not in self.banned_champions_names:
                self.banned_champions_names.append(name)
            # stop once we have 10 slots

        # 2) Now update the icons & autocomplete
        self.update_ban_icons()
        self.update_autocomplete_for_bans()
        self.on_recommend()

    def update_ban_icons(self):
        # Clear all icons
        for lbl in self.banned_icon_labels:
            lbl.configure(image="")
            lbl.image = None

        # Add icons for current banned champions
        for i, name in enumerate(self.banned_champions_names):
            if i >= len(self.banned_icon_labels):
                break
            icon = self.champion_icons.get(name)
            if icon:
                self.banned_icon_labels[i].configure(image=icon)
                self.banned_icon_labels[i].image = icon

    def update_autocomplete_suggestions(self):
        # Exclude all banned champions (names) tracked in the GUI
        excluded = set(self.banned_champions_names)

        # Exclude enemy picks (names)
        excluded.update(
            e.get_valid_text() for e in self.enemy_champ_boxes if e.get_valid_text()
        )

        for entry in list(self.ally_champs.values()) + self.enemy_champ_boxes:
            current_text = entry.get_text().strip()
            valid_text = entry.get_valid_text().strip()

            # Build base suggestions: non-excluded champs + your own valid pick
            suggestions = [
                champ for champ in self.champion_list
                if champ not in excluded or champ == valid_text
            ]

            # Remove exact typed text always
            if current_text:
                suggestions = [
                    champ for champ in suggestions
                    if champ.lower() != current_text.lower()
                ]

            # Substring filter
            filtered = [
                champ for champ in suggestions
                if not current_text or current_text.lower() in champ.lower()
            ]

            entry.update_suggestions(filtered)

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
        strategy_frame = ttk.LabelFrame(
            self.advanced_frame, text="Pick Strategy")
        strategy_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        strategy_label = ttk.Label(
            strategy_frame, text="Strategy:", font=bigger_font)
        strategy_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        strategy_dropdown = ttk.Combobox(
            strategy_frame,
            textvariable=self.pick_strategy_var,
            values=["Maximize", "Minimax"],
            font=bigger_font,
            state="readonly",
            width=10
        )
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ── Display Metric ──────────────────────────────────────────────────────────
        display_metric_frame = ttk.LabelFrame(
            self.advanced_frame, text="Display Metric")
        display_metric_frame.grid(
            row=1, column=0, padx=10, pady=10, sticky="nw")
        display_metric_label = ttk.Label(
            display_metric_frame, text="Sort By:", font=bigger_font)
        display_metric_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        display_metric_dropdown = ttk.Combobox(
            display_metric_frame,
            textvariable=self.display_metric_var,
            values=["Win Rate", "Delta"],
            font=bigger_font,
            state="readonly"
        )
        display_metric_dropdown.grid(
            row=0, column=1, padx=5, pady=5, sticky="w")

        # ── Adjustment Method ───────────────────────────────────────────────────────
        adjustment_frame = ttk.LabelFrame(
            self.advanced_frame, text="Adjustment Method")
        adjustment_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")
        adjustment_label = ttk.Label(
            adjustment_frame, text="Adjustment:", font=bigger_font)
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
        m_frame = ttk.LabelFrame(
            self.advanced_frame, text="Bayesian Adjustment (m)")
        m_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nw")
        m_label = ttk.Label(m_frame, text="Set m:", font=bigger_font)
        m_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.m_var = tk.IntVar(value=0)
        m_entry = ttk.Entry(m_frame, textvariable=self.m_var,
                            width=10, font=bigger_font)
        m_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        m_button = ttk.Button(m_frame, text="Recalculate",
                              command=self.recalculate_matchups)
        m_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # ── BANNED CHAMPIONS ICONS FRAME ───────────────────────────────────────────────
        self.ban_icons_frame = ttk.LabelFrame(self, text="Banned Champions")
        self.ban_icons_frame.grid(
            row=2, column=0, columnspan=3, padx=10, pady=(10, 0), sticky="w")

        self.banned_icon_labels = []
        for i in range(10):
            lbl = tk.Label(self.ban_icons_frame, image=None)
            lbl.grid(row=0, column=i, padx=3, pady=3)
            self.banned_icon_labels.append(lbl)

        # ── TEAMS FRAME ───────────────────────────────────────────────────────────────
        teams_frame = ttk.Frame(self)
        teams_frame.grid(row=3, column=0, columnspan=2,
                         sticky="nw", padx=10, pady=10)

        # ── ALLY PICKS FRAME ───────────────────────────────────────────────────────────
        ally_frame = ttk.LabelFrame(
            teams_frame, text="Ally Team (roles known)")
        ally_frame.grid(row=0, column=0, padx=10, sticky="nw")

        self.ally_champs = {}
        for i, role in enumerate(self.roles_ally):
            lbl = ttk.Label(
                ally_frame, text=f"{role.capitalize()}:", font=bigger_font)
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
        enemy_frame = ttk.LabelFrame(
            teams_frame, text="Enemy Team (champions only)")
        enemy_frame.grid(row=0, column=1, padx=10, sticky="nw")

        self.enemy_champ_boxes = []
        self.enemy_role_labels = []
        for i in range(5):
            lbl = ttk.Label(
                enemy_frame, text=f"Enemy #{i+1}:", font=bigger_font)
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

            role_lbl = ttk.Label(enemy_frame, text="",
                                 font=bigger_font, foreground="blue")
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
        pick_frame.grid(row=4, column=0, columnspan=3,
                        padx=10, pady=10, sticky="ew")

        btn_reset = ttk.Button(pick_frame, text="Reset",
                               command=self.reset_all)
        btn_reset.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        # The results_frame will contain five LabelFrames (one per role), arranged left→right
        self.results_frame = ttk.Frame(pick_frame)
        self.results_frame.grid(row=2, column=0, columnspan=3, sticky="ew")

        self.icon_frames = {}
        for i, role in enumerate(self.roles_ally):
            container = ttk.LabelFrame(
                self.results_frame, text=role.capitalize())
            container.grid(row=0, column=i, padx=10, pady=5, sticky="nw")
            self.icon_frames[role] = {'container': container, 'icons': []}

            # Add a Copy button under each role's LabelFrame
            copy_btn = ttk.Button(
                container,
                text="Copy",
                command=lambda r=role: self.copy_role_list(r)
            )
            copy_btn.grid(row=6, column=0, columnspan=2,
                          pady=(5, 0), sticky="ew")

        # ── OVERALL WIN RATE LABEL ────────────────────────────────────────────────────
        self.overall_win_rate_label = ttk.Label(
            self, text="", justify="left", font=bigger_font)
        self.overall_win_rate_label.grid(
            row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nw")

##############################################################################
# 8) Draft Detection
###############################################################################


champion_map = {}


def get_lockfile_info(retries=5, delay=1):
    lockfile_path = r"C:\Riot Games\League of Legends\lockfile"
    for _ in range(retries):
        if os.path.exists(lockfile_path):
            with open(lockfile_path, 'r') as f:
                _, _, port, password, _ = f.read().split(':')
                return port, password
        time.sleep(delay)
    # After retries, still no file found
    raise FileNotFoundError(
        "LeagueClient lockfile not found at expected location.")


def get_headers(password):
    auth = base64.b64encode(f'riot:{password}'.encode()).decode()
    return {
        'Authorization': f'Basic {auth}',
        'Accept': 'application/json'
    }


def download_champion_map():
    try:
        version = requests.get(
            'https://ddragon.leagueoflegends.com/api/versions.json').json()[0]
        champ_data = requests.get(
            f'https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json').json()['data']
        for champ in champ_data.values():
            champion_map[int(champ['key'])] = champ['id']
    except Exception as e:
        print("Error fetching champion map:", e)


def get_self_selection(port, headers):
    url = f'https://127.0.0.1:{port}/lol-champ-select/v1/my-selection'
    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Champion select not active yet or already finished
            print("Not in champion select.")
        else:
            print(f"Selection request failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to League Client API.")
    except Exception as e:
        print(f"Request error: {e}")
    return None


was_champselect = True  # Persistent across calls


def get_current_champion_picks(port, headers):
    global was_champselect
    url = f'https://127.0.0.1:{port}/lol-champ-select/v1/session'
    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            if not was_champselect:
                print("Champion select: True")
            was_champselect = True
            return response.json()
        elif response.status_code == 404:
            if was_champselect:
                print("Champion select: False")
                was_champselect = False
            return None
        else:
            print(f"Session request failed: {response.status_code}")
    except Exception as e:
        print(f"Session request error: {e}")
    return None


CELL_ID_TO_ROLE = {
    0: "top",
    1: "jungle",
    2: "middle",
    3: "bottom",
    4: "utility"
}


def update_ally_pick_and_guess(app):
    """
    Rebuilds the full ally team layout based on all locked picks.
    Ensures no duplicate champions and consistent role assignments.
    """
    # Get all unique ally picks from locked_picks
    ally_champs = [
        name for cid, name in sorted(app.locked_picks.items())
        if cid in CELL_ID_TO_ROLE  # Only ally team cells (0-4)
    ]

    # Deduplicate (LCU sometimes sends repeated events)
    unique_champs = list(dict.fromkeys(ally_champs))

    # Clear all ally role boxes
    for entry in app.ally_champs.values():
        entry.set_text("")

    # Assign roles using guess_roles()
    guessed_roles = guess_roles(unique_champs, app.df_priors)

    filled_champs = set()
    for role, champ in guessed_roles.items():
        if champ not in filled_champs and role in app.ally_champs:
            app.ally_champs[role].set_text(champ)
            filled_champs.add(champ)

    app.on_recommend()


def update_enemy_picks(app):
    enemy_picks = [
        name for cid, name in sorted(app.locked_picks.items())
        if cid not in CELL_ID_TO_ROLE
    ]
    print("[DEBUG] enemy_picks:", enemy_picks)

    unique_champs = list(dict.fromkeys(enemy_picks))
    print("[DEBUG] unique enemy champs:", unique_champs)

    for i, entry in enumerate(app.enemy_champ_boxes):
        print(f"[DEBUG] Clearing enemy box {i}")
        entry.set_text("")

    for i, champ in enumerate(unique_champs):
        if i < len(app.enemy_champ_boxes):
            print(f"[DEBUG] Setting enemy box {i} to {champ}")
            app.enemy_champ_boxes[i].set_text(champ)

    app.on_recommend()

def draft_monitor(app):
    printed_waiting_message = False
    while True:
        try:
            port, password = get_lockfile_info()
            print("lockfile aquired")
            break
        except FileNotFoundError:
            if not printed_waiting_message:
                print("Waiting for League Client lockfile...")
                printed_waiting_message = True
            time.sleep(2)
    headers = get_headers(password)
    download_champion_map()

    seen_locks = set()
    if not hasattr(app, "banned_champion_ids"):
        app.banned_champion_ids = set()

    was_in_champ_select = False

    while True:
        session = get_current_champion_picks(port, headers)
        in_champ_select = session is not None and "actions" in session

        if not in_champ_select:
            if was_in_champ_select:
                # Only clear once when leaving champ select
                seen_locks.clear()
                app.banned_champion_ids.clear()

                if hasattr(app, 'reset_all'):
                    app.after(0, app.reset_all)
                elif hasattr(app, 'reset'):
                    app.after(0, app.reset)
                elif hasattr(app, 'reset_gui'):
                    app.after(0, app.reset_gui)

            was_in_champ_select = False
            time.sleep(1)
            continue

        # Only clear once when entering champ select
        if not was_in_champ_select:
            seen_locks.clear()
            app.banned_champion_ids.clear()

        was_in_champ_select = True

        # Now safe to proceed with processing session
        cell_id_to_team = {p["cellId"]: "Ally" for p in session.get("myTeam", [])}
        cell_id_to_team.update({p["cellId"]: "Enemy" for p in session.get("theirTeam", [])})
        app.cell_id_to_team = cell_id_to_team  # Save in app for later use

        # Handle bans
        bans = session.get("bans", {})
        team_bans = bans.get("myTeamBans", [])
        enemy_bans = bans.get("theirTeamBans", [])
        new_banned_names = []

        for champ_id in team_bans + enemy_bans:
            if champ_id and champ_id not in app.banned_champion_ids:
                name = champion_map.get(champ_id, f"Unknown ({champ_id})")
                new_banned_names.append(name)
                app.banned_champion_ids.add(champ_id)

        # Fallback for ban actions
        if not (team_bans or enemy_bans):
            for group in session.get("actions", []):
                for action in group:
                    if action.get("type") == "ban" and action.get("completed"):
                        cid = action.get("championId", 0)
                        if cid and cid not in app.banned_champion_ids:
                            name = champion_map.get(cid, f"Unknown ({cid})")
                            new_banned_names.append(name)
                            app.banned_champion_ids.add(cid)

        if new_banned_names:
            app.after(0, app.update_gui_bans, new_banned_names)

        # Track previous picks to avoid duplicate updates
        if not hasattr(app, "locked_picks"):
            app.locked_picks = {}
        for group in session.get("actions", []):
            for action in group:
                if action.get("type") == "pick" and action.get("completed"):
                    cid = action.get("championId", 0)
                    cell = action.get("actorCellId")
                    team = cell_id_to_team.get(cell, "Unknown")

                    if cid and (cell, cid) not in seen_locks:
                        name = champion_map.get(cid, f"Unknown ({cid})")
                        seen_locks.add((cell, cid))
                        print(team + " - " + name)
                        if team == "Ally":
                            app.locked_picks[cell] = name
                            app.after(0, update_ally_pick_and_guess, app)
                        elif team == "Enemy":
                            app.locked_picks[cell] = name
                            app.after(0, update_enemy_picks, app)
                            app.after(0, app.on_recommend)


        time.sleep(1)


##############################################################################
# 9) Run the GUI
###############################################################################
if __name__ == "__main__":
    # Load data
    df_matchups = load_matchup_data("data/matchups_shrunk.csv")
    df_priors = load_champion_priors("data/champion_priors.csv")

    # Run GUI (create app instance first)
    app = ChampionPickerGUI(df_matchups, df_priors)

    # Start draft detection in background thread, pass app to it
    thread = threading.Thread(target=draft_monitor, args=(app,), daemon=True)
    thread.start()

    # Run the GUI loop
    app.mainloop()
