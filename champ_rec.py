import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from math import log, exp
from scipy.optimize import linear_sum_assignment
import pyperclip
import os
from PIL import Image, ImageTk 



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
def guess_enemy_roles(enemy_champs, priors_df):
    """
    enemy_champs: list of championName (strings), e.g. ['Gragas', 'Yasuo', ...].
    priors_df:    DataFrame with columns [champion_name, top, jungle, middle, bottom, support].
    
    Returns: dict { "top": champName, "jungle": champName, ... }
             from those champions by maximizing prior probabilities,
             allowing any typed champion to take any of the 5 roles.
    """

    known = {}
    unknown = enemy_champs  # all typed champs we want to guess roles for

    if not unknown:
        return known

    # Always use all 5 roles:
    roles_for_guess = ["top", "jungle", "middle", "bottom", "support"]
    n = len(unknown)
    m = len(roles_for_guess)  # == 5

    # Build a (n x m) cost matrix
    cost_matrix = np.zeros((n, m))

    for i, champ in enumerate(unknown):
        # find champion in priors
        row = priors_df[priors_df['champion_name'].str.lower() == champ.lower()]
        if row.empty:
            # fallback uniform distribution across 5 roles
            probs = [1.0/m] * m
        else:
            # get probability for each of the 5 roles
            probs = []
            for role_name in roles_for_guess:
                p = float(row[role_name].iloc[0])
                probs.append(p)

        # fill cost matrix row
        for j in range(m):
            p = probs[j]
            cost_matrix[i, j] = 9999.0 if p <= 0 else -log(p)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build final dict
    for i in range(len(row_ind)):
        champ_index = row_ind[i]
        role_index = col_ind[i]

        # which champion?
        assigned_champ = unknown[champ_index]
        # which role among the 5 possible roles?
        assigned_role = roles_for_guess[role_index]

        known[assigned_role] = assigned_champ

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
                added_value = synergy_value.sum() if hasattr(synergy_value, 'sum') else synergy_value
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
                total_log_odds += counter_value.sum() if hasattr(counter_value, 'sum') else counter_value
            except KeyError:
                # If forward lookup fails, attempt reverse lookup: enemy counters champion
                try:
                    counter_value = df_indexed.loc[
                        (enemy_champ, enemy_role, 'Counter', champ, role),
                        'log_odds'
                    ]
                    # Subtract the value since this is disadvantageous (the complement)
                    total_log_odds -= counter_value.sum() if hasattr(counter_value, 'sum') else counter_value
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
                    subtracted_value = enemy_synergy.sum() if hasattr(enemy_synergy, 'sum') else enemy_synergy
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
    unfilled_enemy_roles = [r for r in all_roles if r not in enemy_filled_roles]

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
        subset = df_indexed.loc[(slice(None), role_to_fill, slice(None), slice(None), slice(None)), :]
    except KeyError:
        subset = pd.DataFrame()

    if subset.empty:
        # No known synergy/counter data for this role
        return []

    all_role_candidates = subset.reset_index()["champ1"].unique().tolist()
    # Also ensure we include anything from champion_pool if needed:
    all_role_candidates = sorted(list(set(all_role_candidates + champion_pool)))

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
                    total_delta += counter_row.get('delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

            # reverse (e_champ counters candidate_champ) => subtract
            try:
                reverse_row = df_indexed.loc[
                    (e_champ, e_role, 'Counter', candidate_champ, role_to_fill)
                ]
                if reverse_row is not None and not reverse_row.empty:
                    total_log_odds -= reverse_row['log_odds'].sum()
                    total_delta -= reverse_row.get('delta_shrunk_bayes', 0.0).sum()
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
                    total_delta += counter_row.get('delta_shrunk_bayes', 0.0).sum()
            except KeyError:
                pass

            # reverse (ally champ counters this new enemy pick) => subtract
            try:
                reverse_row = df_indexed.loc[
                    (a_champ, a_role, 'Counter', enemy_champ, enemy_role)
                ]
                if reverse_row is not None and not reverse_row.empty:
                    total_log_odds -= reverse_row['log_odds'].sum()
                    total_delta -= reverse_row.get('delta_shrunk_bayes', 0.0).sum()
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
        sum_log_odds, sum_delta = synergy_for_candidate(candidate_champ, ally_team, enemy_team)

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
                        e_log_odds, e_delta = synergy_for_enemy_candidate(e_candidate, e_role, enemy_team, ally_team)

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
        self.entry = tk.Entry(self, textvariable=self.entry_var, width=width, font=font)
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

    def _on_keyrelease(self, event):
        if event.keysym in ("Up","Down","Left","Right","Return","Tab","Escape"):
            return

        text = self.entry_var.get().strip()

        if not text:
            # Show the entire suggestion_list when the box is blank
            matches = list(self.suggestion_list)
            if matches:
                self._show_popup(matches)
            else:
                self._hide_popup()
            return

        matches = self._filter_suggestions(text)
        if matches:
            self._show_popup(matches)
        else:
            self._hide_popup()

    def _show_popup(self, suggestions):
        self._hide_popup()
        self.current_suggestions = suggestions
        self.current_index = 0

        self.popup = tk.Toplevel(self)
        self.popup.wm_overrideredirect(True)
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self.popup.geometry(f"+{x}+{y}")

        self.listbox = tk.Listbox(self.popup, selectmode=tk.SINGLE, height=min(6, len(suggestions)))
        self.listbox.pack(fill="both", expand=True)
        for item in suggestions:
            self.listbox.insert(tk.END, item)

        self._wrap_index()
        self._update_listbox_selection()

        self.listbox.bind("<Button-1>", self._on_listbox_click)
        self.listbox.bind("<Return>", self._on_return)
        self.listbox.bind("<Down>", self._on_down_arrow)
        self.listbox.bind("<Up>", self._on_up_arrow)

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
            widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
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
            self.listbox.see(self.current_index)  # Scroll to make the selection visible

    def get_text(self):
        return self.entry_var.get()

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

        # Pre-load champion icons
        self.champion_icons = {}
        for champ in self.champion_list:
            path = ChampionPickerGUI.ICON_PATH_FORMAT.format(champ)
            if os.path.exists(path):
                pil_img = Image.open(path).resize(
                    ChampionPickerGUI.ICON_SIZE,
                    resample=Image.LANCZOS
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
        Called any time ally‐champ text changes OR the auto‐hide checkbox toggles.
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
            self.df_matchups['log_odds_bayes'] = self.df_matchups['win_rate_shrunk_bayes'].apply(win_rate_to_log_odds)

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

            columns_to_save = [c for c in desired_columns if c in self.df_matchups.columns]
            self.df_matchups.to_csv("data/matchups_shrunk.csv", columns=columns_to_save, index=False)

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
        # ── Update log_odds in df_matchups ───────────────────────────────────
        method = self.adjustment_method.get().lower()
        log_col = f'log_odds_{method}'
        if log_col in self.df_matchups.columns:
            self.df_matchups['log_odds'] = self.df_matchups[log_col]
        else:
            self.df_matchups['log_odds'] = self.df_matchups['log_odds_bayes']

        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

        # ── Build ally_team dict ──────────────────────────────────────────────
        ally_team = {}
        for role, entry in self.ally_champs.items():
            nm = entry.get_text().strip()
            if nm:
                ally_team[role] = nm

        # ── Gather enemy champions and guess roles ────────────────────────────
        enemy_champs = [e.get_text().strip() for e in self.enemy_champ_boxes if e.get_text().strip()]
        enemy_team = guess_enemy_roles(enemy_champs, self.df_priors)

        # Display “Akshan → middle” etc.
        guessed_text = ""
        for role, champ in enemy_team.items():
            guessed_text += f"{champ} → {role}\n"
        self.enemy_guess_label.config(text=guessed_text)

        chosen_metric = self.display_metric_var.get()
        excluded_base = set(ally_team.values()) | set(enemy_team.values()) | set(self.get_banned_champions())

        # Ensure the Suggested Picks area is visible before drawing
        self.results_frame.grid()
        self.rearrange_result_icons()

        # ── For each role, compute top-5 and build a vertical list ────────────
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
                    "MinimaxAllRoles" if self.pick_strategy_var.get().startswith("Minimax")
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

                # … inside on_recommend(), for each (champ, total_log_odds, total_delta) …
                subframe = ttk.Frame(self.icon_frames[role]['container'])
                subframe.grid(row=idx, column=0, padx=2, pady=2, sticky="w")

                subframe.champ_name = champ

                # Icon goes in row=0, column=0
                if photo is not None:
                    icon_lbl = ttk.Label(subframe, image=photo)
                    icon_lbl.image = photo
                    icon_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")
                else:
                    text_lbl = ttk.Label(subframe, text=champ, font=("Helvetica", 10))
                    text_lbl.grid(row=0, column=0, padx=(0, 5), sticky="nw")

                win_pct = 100 * total_log_odds
                delta_pct = 100 * total_delta

                # Create a small “text_frame” in row=0, column=1 that will stack two labels:
                text_frame = ttk.Frame(subframe)
                text_frame.grid(row=0, column=1, sticky="nw")

                win_lbl = ttk.Label(text_frame, text=f"W:{win_pct:.0f}%", font=("Helvetica", 10))
                win_lbl.pack(anchor="w")
                delta_lbl = ttk.Label(text_frame, text=f"Δ:{delta_pct:.0f}%", font=("Helvetica", 10))
                delta_lbl.pack(anchor="sw")

                self.icon_frames[role]['icons'].append(subframe)



                if idx == 0:
                    excluded_base.add(champ)

        # Re-apply auto-hide logic
        self.check_filled_roles()

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
            r: e.get_text().strip()
            for r, e in self.ally_champs.items()
            if e.get_text().strip()
        }
        enemy_list = [e.get_text().strip() for e in self.enemy_champ_boxes if e.get_text().strip()]
        enemy_team = guess_enemy_roles(enemy_list, self.df_priors)

        ally_pct, enemy_pct = calculate_overall_win_rates(
            self.df_matchups_indexed, ally_team, enemy_team
        )
        text = (
            f"Estimated Ally Team Win Rate: {ally_pct:.2%}\n"
            f"Estimated Enemy Team Win Rate: {enemy_pct:.2%}"
        )
        self.overall_win_rate_label.config(text=text)

    def reset_all(self):
        """
        Clear all entry fields and destroy any existing icon subframes.
        """
        for e in self.ally_champs.values():
            e.entry_var.set("")
        for e in self.enemy_champ_boxes:
            e.entry_var.set("")
        for b in self.ban_champ_boxes:
            b.entry_var.set("")

        for role in self.roles_ally:
            for widget in self.icon_frames[role]['icons']:
                widget.destroy()
            self.icon_frames[role]['icons'].clear()

        self.enemy_guess_label.config(text="")
        self.overall_win_rate_label.config(text="")

        self.check_filled_roles()
        self.update_overall_win_rates()

    def get_banned_champions(self):
        """
        Return a list of champion names currently typed into the ban boxes.
        """
        banned = []
        for ban_entry in self.ban_champ_boxes:
            nm = ban_entry.get_text().strip()
            if nm:
                banned.append(nm)
        return banned

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
            values=["Maximize", "Minimax"],
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

        # ── BANNED PICKS FRAME (optional) ───────────────────────────────────────────────
        ban_frame = ttk.LabelFrame(self, text="Banned Champions")
        # If you want it visible by default, uncomment:
        # ban_frame.grid(row=…, column=…)

        self.ban_champ_boxes = []
        for i in range(10):
            lbl = ttk.Label(ban_frame, text=f"Ban #{i+1}:", font=bigger_font)
            lbl.grid(row=i, column=0, sticky="w")

            ban_entry = AutocompleteEntryPopup(
                ban_frame,
                suggestion_list=self.champion_list,
                width=12,
                font=bigger_font,
                callback=self.combined_callback
            )
            ban_entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.ban_champ_boxes.append(ban_entry)

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

##############################################################################
# 8) Run the GUI
###############################################################################
if __name__ == "__main__":
    # Load precomputed matchup data that includes dedicated columns
    df_matchups = load_matchup_data("data/matchups_shrunk.csv")
    df_priors = load_champion_priors("data/champion_priors.csv")

    # Initialize and run the GUI
    app = ChampionPickerGUI(df_matchups, df_priors)
    app.mainloop()
