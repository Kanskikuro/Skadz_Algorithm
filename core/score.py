from math import log, exp


###############################################################################
# 4) Synergy / Delta scoring
###############################################################################

def log_odds_to_probability(log_odds):
    return 1 / (1 + exp(-log_odds))


def win_rate_to_log_odds(win_rate):
    p = win_rate / 100.0
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return log(p / (1 - p))


def _as_scalar(value, aggregation="mean"):
    """
    Converts a pandas scalar/Series result into a float.

    If duplicate rows exist in df_indexed, .loc may return a Series.
    aggregation="mean" is usually safer than "sum", because duplicates
    should usually not inflate the matchup strength.
    """
    if hasattr(value, aggregation):
        return float(getattr(value, aggregation)())
    return float(value)


def lookup_pair(
    df_indexed,
    champ_a,
    role_a,
    relation,
    champ_b,
    role_b,
    aggregation="mean",
):
    """
    Looks up a pair relation from the indexed dataframe.

    Expected index format:
        (champion_1, role_1, relation, champion_2, role_2)

    Expected column:
        "log_odds"

    Returns:
        (value, "forward") if A -> B exists
        (value, "reverse") if B -> A exists
        (None, None) if neither exists

    For Synergy:
        forward and reverse are equivalent.

    For Counter:
        forward means champ_a counters champ_b.
        reverse means champ_b counters champ_a.
    """
    try:
        value = df_indexed.loc[
            (champ_a, role_a, relation, champ_b, role_b),
            "log_odds",
        ]
        return _as_scalar(value, aggregation), "forward"
    except KeyError:
        pass

    try:
        value = df_indexed.loc[
            (champ_b, role_b, relation, champ_a, role_a),
            "log_odds",
        ]
        return _as_scalar(value, aggregation), "reverse"
    except KeyError:
        return None, None


def calculate_overall_win_rates(
    df_indexed,
    ally_team,
    enemy_team,
    synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.2,
    predicted_enemy_picks=None,
    use_enemy_prediction=False,
    predicted_counter_weight=0.6,
    aggregation="mean",
    normalize=False,
):
    """
    Calculates ally and enemy win-rate estimates.

    ally_team and enemy_team should be dictionaries:

        {
            "Top": "Aatrox",
            "Jungle": "Lee Sin",
            "Middle": "Ahri",
            "Bottom": "Jinx",
            "Support": "Thresh",
        }

    predicted_enemy_picks is optional and should be a list of tuples:

        [
            ("Middle", "Yasuo", 0.85),
            ("Support", "Leona", 0.60),
            ("Jungle", "Diana", 0.50),
        ]

    Confidence should usually be between 0.0 and 1.0.
    """
    ally_log_odds = calculate_team_log_odds(
        df_indexed=df_indexed,
        team=ally_team,
        opponent_team=enemy_team,
        synergy_weight=synergy_weight,
        counter_weight=counter_weight,
        enemy_synergy_weight=enemy_synergy_weight,
        predicted_enemy_picks=predicted_enemy_picks,
        use_enemy_prediction=use_enemy_prediction,
        predicted_counter_weight=predicted_counter_weight,
        aggregation=aggregation,
        normalize=normalize,
    )

    ally_win_rate = log_odds_to_probability(ally_log_odds)
    enemy_win_rate = 1 - ally_win_rate

    return ally_win_rate, enemy_win_rate


def calculate_team_log_odds(
    df_indexed,
    team,
    opponent_team,
    synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.2,
    predicted_enemy_picks=None,
    use_enemy_prediction=False,
    predicted_counter_weight=0.6,
    aggregation="mean",
    normalize=False,
):
    """
    Calculates a log-odds score for one team against the opponent team.

    Score components:

        + own team synergy
        + counters against current enemy picks
        + optional counters against predicted future enemy picks
        - enemy team synergy

    Suggested solo queue weights:

        synergy_weight = 0.4
        counter_weight = 1.0
        enemy_synergy_weight = 0.2
        predicted_counter_weight = 0.6

    For full 5v5 scoring:
        normalize=False

    For partial draft scoring:
        normalize=True can be useful, but may reduce the impact of having
        more available information.
    """
    total_log_odds = 0.0
    terms = 0

    # -------------------------------------------------------------------------
    # 1. Own team synergy
    # -------------------------------------------------------------------------
    ally_items = list(team.items())

    for i in range(len(ally_items)):
        role_i, champ_i = ally_items[i]

        for j in range(i + 1, len(ally_items)):
            role_j, champ_j = ally_items[j]

            value, _ = lookup_pair(
                df_indexed=df_indexed,
                champ_a=champ_i,
                role_a=role_i,
                relation="Synergy",
                champ_b=champ_j,
                role_b=role_j,
                aggregation=aggregation,
            )

            if value is not None:
                total_log_odds += synergy_weight * value
                terms += 1

    # -------------------------------------------------------------------------
    # 2. Counters against already picked enemies
    # -------------------------------------------------------------------------
    for role, champ in team.items():
        for enemy_role, enemy_champ in opponent_team.items():
            value, direction = lookup_pair(
                df_indexed=df_indexed,
                champ_a=champ,
                role_a=role,
                relation="Counter",
                champ_b=enemy_champ,
                role_b=enemy_role,
                aggregation=aggregation,
            )

            if value is None:
                continue

            if direction == "forward":
                total_log_odds += counter_weight * value
            else:
                total_log_odds -= counter_weight * value

            terms += 1

    # -------------------------------------------------------------------------
    # 3. Optional: counters against predicted future enemy picks
    # -------------------------------------------------------------------------
    if use_enemy_prediction and predicted_enemy_picks:
        for predicted_role, predicted_champ, confidence in predicted_enemy_picks:
            confidence = max(0.0, min(float(confidence), 1.0))

            for role, champ in team.items():
                value, direction = lookup_pair(
                    df_indexed=df_indexed,
                    champ_a=champ,
                    role_a=role,
                    relation="Counter",
                    champ_b=predicted_champ,
                    role_b=predicted_role,
                    aggregation=aggregation,
                )

                if value is None:
                    continue

                weighted_value = predicted_counter_weight * confidence * value

                if direction == "forward":
                    total_log_odds += weighted_value
                else:
                    total_log_odds -= weighted_value

                terms += 1

    # -------------------------------------------------------------------------
    # 4. Subtract existing enemy synergy
    # -------------------------------------------------------------------------
    enemy_items = list(opponent_team.items())

    for i in range(len(enemy_items)):
        role_i, champ_i = enemy_items[i]

        for j in range(i + 1, len(enemy_items)):
            role_j, champ_j = enemy_items[j]

            value, _ = lookup_pair(
                df_indexed=df_indexed,
                champ_a=champ_i,
                role_a=role_i,
                relation="Synergy",
                champ_b=champ_j,
                role_b=role_j,
                aggregation=aggregation,
            )

            if value is not None:
                total_log_odds -= enemy_synergy_weight * value
                terms += 1

    if normalize and terms > 0:
        total_log_odds /= terms

    return total_log_odds