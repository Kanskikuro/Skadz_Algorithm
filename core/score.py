from math import log, exp


###############################################################################
# 4) Synergy/Delta scoring
###############################################################################
# def prepare_multiindex(df):
#     # Create a multi-index based on columns commonly used in lookups.
#     df_indexed = df.set_index(['champ1', 'role1', 'type', 'champ2', 'role2'])
#     return df_indexed.sort_index()  # Sort the index for optimal performance

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