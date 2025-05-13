import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from math import log, exp
from scipy.optimize import linear_sum_assignment
import pyperclip

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
import tkinter as tk

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
            self.entry.focus_set()
            return "break"
        else:
            matches = self._filter_suggestions(self.entry_var.get().strip())
            if len(matches) == 1:
                self._set_text(matches[0])

    def _on_tab_press(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            return "break"

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
# 7) The main GUI (ChampionPickerGUI) with Bans
###############################################################################
class ChampionPickerGUI(tk.Tk):
    def __init__(self, df_matchups, df_priors):
        super().__init__()
        self.geometry("1280x920")
        self.title("League Champion Picker")

        # Store the raw matchup DataFrame
        self.df_matchups = df_matchups  
        # Prepare the indexed DataFrame for fast lookups
        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)  
        
        self.df_priors = df_priors

        # We can pull champion list from df_priors
        self.champion_list = list(df_priors['champion_name'].unique())

        # Roles
        self.roles_ally = ["top", "jungle", "middle", "bottom", "support"]

        # Build UI
        self._build_ui()

    def combined_callback(self):
        self.update_overall_win_rates()
        self.on_recommend()


    def _build_ui(self):
        bigger_font = ("Helvetica", 14)

        # Combobox for pick strategy
        strategy_frame = ttk.LabelFrame(self, text="Pick Strategy")
        strategy_frame.grid(row=5, column=0, padx=10, pady=10, sticky="nw")

        strategy_label = ttk.Label(strategy_frame, text="Strategy:", font=bigger_font)
        strategy_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.pick_strategy_var = tk.StringVar(value="Maximize")  # or "Minimax"
        strategy_dropdown = ttk.Combobox(
            strategy_frame,
            textvariable=self.pick_strategy_var,
            values=["Maximize", "Minimax"],
            font=bigger_font,
            state="readonly",
            width=10
        )
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        display_metric_frame = ttk.LabelFrame(self, text="Display Metric")
        display_metric_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nw")

        display_metric_label = ttk.Label(display_metric_frame, text="Sort By:", font=bigger_font)
        display_metric_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.display_metric_var = tk.StringVar(value="Delta")  # default
        display_metric_dropdown = ttk.Combobox(
            display_metric_frame, 
            textvariable=self.display_metric_var, 
            values=["Win Rate", "Delta"], 
            font=bigger_font, 
            state="readonly"
        )
        display_metric_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Add dropdown for selecting the adjustment method
        adjustment_frame = ttk.LabelFrame(self, text="Adjustment Method")
        adjustment_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nw")

        self.adjustment_method = tk.StringVar(value="Bayesian")  # Default: Bayesian adjustment
        adjustment_label = ttk.Label(adjustment_frame, text="Select Adjustment:", font=bigger_font)
        adjustment_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        adjustment_dropdown = ttk.Combobox(
            adjustment_frame, textvariable=self.adjustment_method, values=["Bayesian", "ADVI", "Hierarchical"], font=bigger_font
        )
        adjustment_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # m Value Frame
        m_frame = ttk.LabelFrame(self, text="Bayesian Adjustment (m)")
        m_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

        m_label = ttk.Label(m_frame, text="Set m value:", font=bigger_font)
        m_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.m_var = tk.IntVar(value=0)  # Default m value
        self.m_entry = ttk.Entry(m_frame, textvariable=self.m_var, width=10, font=bigger_font)
        self.m_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        m_button = ttk.Button(m_frame, text="Recalculate", command=self.recalculate_matchups)
        m_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # ALLY picks frame
        ally_frame = ttk.LabelFrame(self, text="Ally Team (known roles)")
        ally_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.ally_champs = {}  # role -> AutocompleteEntryPopup
        for i, role in enumerate(self.roles_ally):
            label = ttk.Label(ally_frame, text=role.capitalize() + ":", font=bigger_font)
            label.grid(row=i, column=0, sticky="w")

            champ_entry = AutocompleteEntryPopup(
                ally_frame,
                suggestion_list=self.champion_list,
                width=25,
                font=bigger_font,
                callback=self.combined_callback
            )
            champ_entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.ally_champs[role] = champ_entry

        # ENEMY picks frame
        enemy_frame = ttk.LabelFrame(self, text="Enemy Team (champions only)")
        enemy_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        self.enemy_champ_boxes = []
        for i in range(5):
            label = ttk.Label(enemy_frame, text=f"Enemy #{i+1}:", font=bigger_font)
            label.grid(row=i, column=0, sticky="w")

            champ_entry = AutocompleteEntryPopup(
                enemy_frame,
                suggestion_list=self.champion_list,
                width=25,
                font=bigger_font,
                callback=self.combined_callback
            )
            champ_entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.enemy_champ_boxes.append(champ_entry)

        # BANNED picks frame
        ban_frame = ttk.LabelFrame(self, text="Banned Champions")
        ban_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nw")

        self.ban_champ_boxes = []
        for i in range(10):  # up to 10 bans
            label = ttk.Label(ban_frame, text=f"Ban #{i+1}:", font=bigger_font)
            label.grid(row=i, column=0, sticky="w")

            ban_entry = AutocompleteEntryPopup(
                ban_frame,
                suggestion_list=self.champion_list,
                width=25,
                font=bigger_font,
                callback=self.combined_callback
            )
            ban_entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.ban_champ_boxes.append(ban_entry)

        # Frame for "Which role am I picking?"
        pick_frame = ttk.LabelFrame(self, text="My Next Pick")
        pick_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        pick_label = ttk.Label(pick_frame, text="Select Your Role:", font=bigger_font)
        pick_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.pick_role_var = tk.StringVar()
        self.pick_role_combo = ttk.Combobox(
            pick_frame,
            textvariable=self.pick_role_var,
            values=self.roles_ally + ["all"],  # include "all"
            width=10,
            font=bigger_font
        )
        self.pick_role_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.pick_role_combo.set("all")

        btn_recommend = ttk.Button(pick_frame, text="Recommend", command=self.on_recommend)
        btn_recommend.grid(row=0, column=2, padx=20, pady=5)

        btn_reset = ttk.Button(pick_frame, text="Reset", command=self.reset_all)
        btn_reset.grid(row=0, column=3, padx=20, pady=5)  # Adjust position as needed

        # Create a frame to hold five side-by-side result labels and copy buttons
        self.results_frame = ttk.Frame(pick_frame)
        self.results_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Initialize dictionaries to hold labels and copy buttons for each role
        self.result_labels = {}
        self.copy_buttons = {}

        for i, role in enumerate(self.roles_ally):
            # Create a subframe for each role to hold label and button
            subframe = ttk.Frame(self.results_frame)
            subframe.grid(row=0, column=i, padx=5, pady=5, sticky="nw")

            # Create a Label for each role's recommendation list
            lbl = ttk.Label(subframe, text="", justify="left", font=bigger_font)
            lbl.grid(row=0, column=0, sticky="nw")
            self.result_labels[role] = lbl

            # Create a "Copy" button for this role
            btn = ttk.Button(subframe, text="Copy", command=lambda r=role: self.copy_to_clipboard(r))
            btn.grid(row=1, column=0, pady=(5,0))
            self.copy_buttons[role] = btn

        # A separate label to show enemy role guesses
        self.enemy_guess_label = ttk.Label(pick_frame, text="", justify="left", font=bigger_font)
        self.enemy_guess_label.grid(row=2, column=0, columnspan=3, sticky="w")

        # A separate label showing each team's estimated overall win rate 
        self.overall_win_rate_label = ttk.Label(pick_frame, text="", justify="left", font=bigger_font)
        self.overall_win_rate_label.grid(row=3, column=0, columnspan=3, sticky="w")

    def get_banned_champions(self):
        """
        Gather up all typed bans from ban_champ_boxes.
        Returns a list of ban strings (champion names).
        """
        banned = []
        for ban_entry in self.ban_champ_boxes:
            champ_name = ban_entry.get_text().strip()
            if champ_name:
                banned.append(champ_name)
        return banned

    def on_recommend(self):
        pick_strategy = self.pick_strategy_var.get()  # "Maximize" or "Minimax"

        # Determine selected adjustment method
        selected_method = self.adjustment_method.get().lower()  # 'bayesian', 'advi', 'hierarchical'
        selected_log_odds_column = f'log_odds_{selected_method}'
        if selected_log_odds_column in self.df_matchups.columns:
            self.df_matchups['log_odds'] = self.df_matchups[selected_log_odds_column]
        else:
            self.df_matchups['log_odds'] = self.df_matchups['log_odds_bayes']

        # Rebuild multi-index
        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)
        
        # Gather ally team
        ally_team = {}
        for role, entry_widget in self.ally_champs.items():
            champ = entry_widget.get_text().strip()
            if champ:
                ally_team[role] = champ
        
        # Gather enemy champs
        enemy_champs = []
        for entry_widget in self.enemy_champ_boxes:
            champ = entry_widget.get_text().strip()
            if champ:
                enemy_champs.append(champ)

        # Guess enemy roles
        enemy_team = guess_enemy_roles(enemy_champs, self.df_priors)

        # Show guessed roles
        guessed_roles_str = "Guessed roles for enemies:\n"
        for role, champ in enemy_team.items():
            guessed_roles_str += f"  {champ} => {role}\n"
        self.enemy_guess_label.config(text=guessed_roles_str)

        selected_role = self.pick_role_var.get().strip().lower()

        # Retrieve chosen display metric
        chosen_display_metric = self.display_metric_var.get()  # "Win Rate" or "Delta"

        # Build a set of excluded champions:
        # all current picks on ally side + enemy side, plus banned champions
        excluded_champs = set(ally_team.values()) | set(enemy_team.values()) | set(self.get_banned_champions())

        if selected_role == "all":
            for role in self.roles_ally:
                scores = get_champion_scores_for_role(
                    df_indexed=self.df_matchups_indexed,
                    role_to_fill=role,
                    ally_team=ally_team,
                    enemy_team=enemy_team,
                    pick_strategy=("MinimaxAllRoles" if pick_strategy.startswith("Minimax") else "Maximize"),
                    champion_pool=self.champion_list,
                    excluded_champions=excluded_champs
                )
                # scores is list of (champ, sum_log_odds, sum_delta)

                # -------------- Sort by user-chosen metric ---------------
                if chosen_display_metric == "Delta":
                    # Sort by sum_delta descending
                    scores.sort(key=lambda x: x[2], reverse=True)
                else:
                    # Default is "Win Rate", sort by sum_log_odds descending
                    scores.sort(key=lambda x: x[1], reverse=True)

                n = 10
                top_n = scores[:n]
                result_text = f"Top {len(top_n)} for {role.capitalize()}:\n"
                for champ, total_log_odds, total_delta in top_n:
                    result_text += (
                        f"{champ}: W={10 * total_log_odds:.1f}, D={10 * total_delta:.1f}\n"
                    )
                self.result_labels[role].config(text=result_text)
        else:
            # Clear other role labels
            for role in self.roles_ally:
                if role != selected_role:
                    self.result_labels[role].config(text="")

            scores = get_champion_scores_for_role(
                df_indexed=self.df_matchups_indexed,
                role_to_fill=selected_role,
                ally_team=ally_team,
                enemy_team=enemy_team,
                pick_strategy=("MinimaxAllRoles" if pick_strategy.startswith("Minimax") else "Maximize"),
                champion_pool=self.champion_list,
                excluded_champions=excluded_champs
            )

            if chosen_display_metric == "Delta":
                scores.sort(key=lambda x: x[2], reverse=True)
            else:
                scores.sort(key=lambda x: x[1], reverse=True)

            n = 10
            top_n = scores[:n]
            result_text = f"Top {len(top_n)} for {selected_role.capitalize()}:\n"
            for champ, total_log_odds, total_delta in top_n:
                result_text += (
                    f"{champ}: W={10 * total_log_odds:.1f}, D={10 * total_delta:.1f}\n"
                )
            self.result_labels[selected_role].config(text=result_text)

    def update_overall_win_rates(self):
        # Determine selected adjustment method and update active log_odds column
        selected_method = self.adjustment_method.get().lower()
        selected_log_odds_column = f'log_odds_{selected_method}'
        if selected_log_odds_column in self.df_matchups.columns:
            self.df_matchups['log_odds'] = self.df_matchups[selected_log_odds_column]
        else:
            self.df_matchups['log_odds'] = self.df_matchups['log_odds_bayes']

        # Rebuild the indexed DataFrame to use the updated log_odds values
        self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

        # Gather current ally team composition from UI
        ally_team = {}
        for role, entry_widget in self.ally_champs.items():
            champ = entry_widget.get_text().strip()
            if champ:
                ally_team[role] = champ

        # Gather enemy team and guess roles
        enemy_champs = [entry.get_text().strip() for entry in self.enemy_champ_boxes if entry.get_text().strip()]
        enemy_team = guess_enemy_roles(enemy_champs, self.df_priors)

        # Calculate overall win rates
        ally_win_rate, enemy_win_rate = calculate_overall_win_rates(
            self.df_matchups_indexed, ally_team, enemy_team
        )

        overall_text = (
            f"Estimated Ally Team Win Rate: {ally_win_rate:.2%}\n"
            f"Estimated Enemy Team Win Rate: {enemy_win_rate:.2%}"
        )
        self.overall_win_rate_label.config(text=overall_text)

        guessed_roles_str = "Guessed roles for enemies:\n"
        for role, champ in enemy_team.items():
            guessed_roles_str += f"  {champ} => {role}\n"
        self.enemy_guess_label.config(text=guessed_roles_str)

    def reset_all(self):
        # Clear ally champion entries
        for entry_widget in self.ally_champs.values():
            entry_widget.entry_var.set("")

        # Clear enemy champion entries
        for entry_widget in self.enemy_champ_boxes:
            entry_widget.entry_var.set("")

        # Clear banned champion entries
        for ban_entry in self.ban_champ_boxes:
            ban_entry.entry_var.set("")

        # Clear recommendation results for each role
        for label in self.result_labels.values():
            label.config(text="")

        # Clear enemy guess label and overall win rate label
        self.enemy_guess_label.config(text="")
        self.overall_win_rate_label.config(text="")

        # Optionally, update overall win rates to recalculate based on cleared state
        self.update_overall_win_rates()

    def copy_to_clipboard(self, role):
        label = self.result_labels[role]
        text = label.cget("text")
        if text:
            pyperclip.copy(text)  # Use pyperclip to handle clipboard operations

    def recalculate_matchups(self):
        try:
            m_value = self.m_var.get()

            # ------------------------------------------------------------------
            # (1) BAYESIAN ADJUSTMENT FOR WIN RATE
            # ------------------------------------------------------------------
            self.df_matchups['win_rate_shrunk_bayes'] = (
                (self.df_matchups['win_rate'] * self.df_matchups['sample_size'] +
                 50.0 * m_value) / (self.df_matchups['sample_size'] + m_value)
            )

            # Update log odds for Bayesian column
            self.df_matchups['log_odds_bayes'] = self.df_matchups['win_rate_shrunk_bayes'].apply(win_rate_to_log_odds)

            # ------------------------------------------------------------------
            # (2) BAYESIAN ADJUSTMENT FOR DELTA (OPTIONAL)
            # Check if 'delta' is in the DataFrame
            # ------------------------------------------------------------------
            if 'delta' in self.df_matchups.columns:
                self.df_matchups['delta_shrunk_bayes'] = (
                    (self.df_matchups['delta'] * self.df_matchups['sample_size'] + 
                     0.000 * m_value)  # prior = 0.035 corresponds to 52% win rate
                    / (self.df_matchups['sample_size'] + m_value)
                )

            # Recreate the multi-indexed DataFrame for Bayesian recalculations
            self.df_matchups_indexed = prepare_multiindex(self.df_matchups)

            # ------------------------------------------------------------------
            # (3) SAVE COLUMNS BACK TO CSV
            # ------------------------------------------------------------------
            desired_columns = [
                "champ1", "role1", "type", "champ2", "role2", 
                "win_rate", "sample_size", 
                "win_rate_shrunk_bayes", "log_odds_bayes",
                "win_rate_shrunk_advi", "log_odds_advi",
                "win_rate_shrunk_hierarchical", "log_odds_hierarchical",
            ]
            
            # If we computed 'delta_shrunk_bayes', add it
            if 'delta_shrunk_bayes' in self.df_matchups.columns:
                desired_columns.append('delta_shrunk_bayes')

            if 'delta' in self.df_matchups.columns:
                desired_columns.append('delta')

            # Keep only columns that exist in the DataFrame
            columns_to_save = [col for col in desired_columns if col in self.df_matchups.columns]

            self.df_matchups.to_csv("matchups_shrunk.csv", columns=columns_to_save, index=False)

            # ------------------------------------------------------------------
            # (4) UPDATE THE GUI
            # ------------------------------------------------------------------
            self.update_overall_win_rates()
            self.on_recommend()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to recalculate matchups: {e}")


###############################################################################
# 8) Run the GUI
###############################################################################
if __name__ == "__main__":
    # Load precomputed matchup data that includes dedicated columns
    df_matchups = load_matchup_data("matchups_shrunk.csv")
    df_priors = load_champion_priors("champion_priors.csv")

    # Initialize and run the GUI
    app = ChampionPickerGUI(df_matchups, df_priors)
    app.mainloop()
