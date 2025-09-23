
import pandas as pd
###############################################################################
# get_champion_scores_for_role, with "excluded" filter
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
    # Ensure both are lists before concatenation
    if not isinstance(all_role_candidates, list):
        all_role_candidates = [all_role_candidates]
    if not isinstance(champion_pool, list):
        champion_pool = list(champion_pool) if champion_pool is not None else []
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