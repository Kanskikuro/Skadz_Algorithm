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


def lookup_pair_values(
    matchup_repo,
    method,
    champ_a,
    role_a,
    relation,
    champ_b,
    role_b,
):
    """
    Looks up a Synergy/Counter pair via matchup_repo, trying:

        champ_a, role_a, relation, champ_b, role_b

    first, then falling back to the reversed pair:

        champ_b, role_b, relation, champ_a, role_a

    Returns (log_odds, delta, direction):
        direction="forward" if champ_a -> champ_b was found
        direction="reverse" if champ_b -> champ_a was found
        direction=None (with 0.0, 0.0) if neither exists

    This is the single place both scoring paths resolve a pair through -
    the full-team draft score below (calculate_team_log_odds) and the
    per-candidate recommendation scoring in core.recommend. Duplicate-row
    averaging lives once, in MatchupRepository.fast_lookup/fast_delta_lookup,
    so neither caller needs to reimplement it.
    """
    log_odds = matchup_repo.get_pair_score(
        champ1=champ_a,
        role1=role_a,
        relation_type=relation,
        champ2=champ_b,
        role2=role_b,
        method=method,
        default=None,
    )

    if log_odds is not None:
        delta = matchup_repo.get_pair_delta(
            champ1=champ_a,
            role1=role_a,
            relation_type=relation,
            champ2=champ_b,
            role2=role_b,
            method=method,
            default=0.0,
        )
        return log_odds, delta, "forward"

    log_odds = matchup_repo.get_pair_score(
        champ1=champ_b,
        role1=role_b,
        relation_type=relation,
        champ2=champ_a,
        role2=role_a,
        method=method,
        default=None,
    )

    if log_odds is not None:
        delta = matchup_repo.get_pair_delta(
            champ1=champ_b,
            role1=role_b,
            relation_type=relation,
            champ2=champ_a,
            role2=role_a,
            method=method,
            default=0.0,
        )
        return log_odds, delta, "reverse"

    return 0.0, 0.0, None


def calculate_overall_win_rates(
    matchup_repo,
    ally_team,
    enemy_team,
    method="Bayesian",
    synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.2,
    predicted_enemy_picks=None,
    use_enemy_prediction=False,
    predicted_counter_weight=0.6,
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
        matchup_repo=matchup_repo,
        team=ally_team,
        opponent_team=enemy_team,
        method=method,
        synergy_weight=synergy_weight,
        counter_weight=counter_weight,
        enemy_synergy_weight=enemy_synergy_weight,
        predicted_enemy_picks=predicted_enemy_picks,
        use_enemy_prediction=use_enemy_prediction,
        predicted_counter_weight=predicted_counter_weight,
        normalize=normalize,
    )

    ally_win_rate = log_odds_to_probability(ally_log_odds)
    enemy_win_rate = 1 - ally_win_rate

    return ally_win_rate, enemy_win_rate


def calculate_team_log_odds(
    matchup_repo,
    team,
    opponent_team,
    method="Bayesian",
    synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.2,
    predicted_enemy_picks=None,
    use_enemy_prediction=False,
    predicted_counter_weight=0.6,
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

            value, _delta, direction = lookup_pair_values(
                matchup_repo=matchup_repo,
                method=method,
                champ_a=champ_i,
                role_a=role_i,
                relation="Synergy",
                champ_b=champ_j,
                role_b=role_j,
            )

            if direction is not None:
                total_log_odds += synergy_weight * value
                terms += 1

    # -------------------------------------------------------------------------
    # 2. Counters against already picked enemies
    # -------------------------------------------------------------------------
    for role, champ in team.items():
        for enemy_role, enemy_champ in opponent_team.items():
            value, _delta, direction = lookup_pair_values(
                matchup_repo=matchup_repo,
                method=method,
                champ_a=champ,
                role_a=role,
                relation="Counter",
                champ_b=enemy_champ,
                role_b=enemy_role,
            )

            if direction is None:
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
                value, _delta, direction = lookup_pair_values(
                    matchup_repo=matchup_repo,
                    method=method,
                    champ_a=champ,
                    role_a=role,
                    relation="Counter",
                    champ_b=predicted_champ,
                    role_b=predicted_role,
                )

                if direction is None:
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

            value, _delta, direction = lookup_pair_values(
                matchup_repo=matchup_repo,
                method=method,
                champ_a=champ_i,
                role_a=role_i,
                relation="Synergy",
                champ_b=champ_j,
                role_b=role_j,
            )

            if direction is not None:
                total_log_odds -= enemy_synergy_weight * value
                terms += 1

    if normalize and terms > 0:
        total_log_odds /= terms

    return total_log_odds
