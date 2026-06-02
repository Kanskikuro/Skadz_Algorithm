import pandas as pd
from enums import ROLES

ALL_ROLES = ROLES


def _as_scalar(value, column=None, aggregation="mean", default=0.0):
    """
    Convert pandas scalar/DataFrame/Series results into one float.

    aggregation="mean" is usually safer than "sum", because duplicate rows
    should usually not inflate the score.
    """
    if value is None:
        return default

    if isinstance(value, pd.DataFrame):
        if value.empty or column not in value.columns:
            return default
        series = value[column]
        return float(getattr(series, aggregation)())

    if isinstance(value, pd.Series):
        if column is not None:
            if column not in value:
                return default
            return float(value[column])
        return float(getattr(value, aggregation)())

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def lookup_pair_values(
    df_indexed,
    champ_a,
    role_a,
    relation,
    champ_b,
    role_b,
    aggregation="mean",
):
    """
    Looks up:

        champ_a, role_a, relation, champ_b, role_b

    Returns:
        (log_odds, delta, direction)

    direction:
        "forward" = A -> B was found
        "reverse" = B -> A was found
        None = no row found
    """
    try:
        row = df_indexed.loc[
            (champ_a, role_a, relation, champ_b, role_b)
        ]
        return (
            _as_scalar(row, "log_odds", aggregation),
            _as_scalar(row, "delta_shrunk_bayes", aggregation),
            "forward",
        )
    except KeyError:
        pass

    try:
        row = df_indexed.loc[
            (champ_b, role_b, relation, champ_a, role_a)
        ]
        return (
            _as_scalar(row, "log_odds", aggregation),
            _as_scalar(row, "delta_shrunk_bayes", aggregation),
            "reverse",
        )
    except KeyError:
        return 0.0, 0.0, None


def get_candidates_for_role(
    df_indexed,
    role_to_fill,
    champion_pool=None,
    excluded_champions=None,
):
    """
    Finds champions that have data for the given role, plus optionally supplied
    champion_pool champions.
    """
    if champion_pool is None:
        champion_pool = []

    if excluded_champions is None:
        excluded_champions = set()
    else:
        excluded_champions = set(excluded_champions)

    role_to_fill = role_to_fill.lower()

    try:
        subset = df_indexed.loc[
            (slice(None), role_to_fill, slice(None), slice(None), slice(None)),
            :
        ]
    except KeyError:
        subset = pd.DataFrame()

    candidates = set()

    if not subset.empty:
        candidates.update(subset.reset_index()["champ1"].dropna().unique().tolist())

    candidates.update(champion_pool)

    return sorted(
        champ for champ in candidates
        if champ not in excluded_champions
    )


def score_candidate_pick(
    df_indexed,
    candidate_champ,
    role_to_fill,
    ally_team,
    enemy_team,
    ally_synergy_weight=0.4,
    counter_weight=1.0,
    aggregation="mean",
):
    """
    Scores only the candidate champion's direct contribution:

        + synergy with allies
        + counters against known enemies
        - penalties when known enemies counter the candidate
    """
    total_log_odds = 0.0
    total_delta = 0.0

    # 1. Candidate synergy with allies
    for ally_role, ally_champ in ally_team.items():
        if ally_role == role_to_fill and ally_champ == candidate_champ:
            continue

        log_odds, delta, direction = lookup_pair_values(
            df_indexed,
            candidate_champ,
            role_to_fill,
            "Synergy",
            ally_champ,
            ally_role,
            aggregation,
        )

        if direction is not None:
            total_log_odds += ally_synergy_weight * log_odds
            total_delta += ally_synergy_weight * delta

    # 2. Candidate counters against enemies
    for enemy_role, enemy_champ in enemy_team.items():
        log_odds, delta, direction = lookup_pair_values(
            df_indexed,
            candidate_champ,
            role_to_fill,
            "Counter",
            enemy_champ,
            enemy_role,
            aggregation,
        )

        if direction == "forward":
            total_log_odds += counter_weight * log_odds
            total_delta += counter_weight * delta
        elif direction == "reverse":
            total_log_odds -= counter_weight * log_odds
            total_delta -= counter_weight * delta

    return total_log_odds, total_delta


def score_enemy_candidate_pick(
    df_indexed,
    enemy_candidate,
    enemy_role,
    enemy_team,
    ally_team,
    enemy_synergy_weight=0.4,
    enemy_counter_weight=1.0,
    aggregation="mean",
):
    """
    Scores how threatening one possible enemy pick is:

        + synergy with existing enemy team
        + counters against our ally team
        - penalties if our ally team counters it
    """
    total_log_odds = 0.0
    total_delta = 0.0

    # 1. Enemy candidate synergy with existing enemy picks
    for existing_role, existing_champ in enemy_team.items():
        if existing_role == enemy_role:
            continue

        log_odds, delta, direction = lookup_pair_values(
            df_indexed,
            enemy_candidate,
            enemy_role,
            "Synergy",
            existing_champ,
            existing_role,
            aggregation,
        )

        if direction is not None:
            total_log_odds += enemy_synergy_weight * log_odds
            total_delta += enemy_synergy_weight * delta

    # 2. Enemy candidate counters against our ally team
    for ally_role, ally_champ in ally_team.items():
        log_odds, delta, direction = lookup_pair_values(
            df_indexed,
            enemy_candidate,
            enemy_role,
            "Counter",
            ally_champ,
            ally_role,
            aggregation,
        )

        if direction == "forward":
            total_log_odds += enemy_counter_weight * log_odds
            total_delta += enemy_counter_weight * delta
        elif direction == "reverse":
            total_log_odds -= enemy_counter_weight * log_odds
            total_delta -= enemy_counter_weight * delta

    return total_log_odds, total_delta


def get_champion_scores_for_role(
    df_indexed,
    role_to_fill,
    ally_team,
    enemy_team,
    pick_strategy="Maximize",
    champion_pool=None,
    excluded_champions=None,
    enemy_candidate_pool=None,
    ally_synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.4,
    enemy_counter_weight=1.0,
    minimax_weight=1.0,
    aggregation="mean",
):
    """
    Returns:
        [
            (champion_name, final_log_odds, final_delta),
            ...
        ]

    pick_strategy:
        "Maximize":
            Scores candidate based only on ally synergy and known enemy counters.

        "MinimaxAllRoles":
            Scores candidate, then subtracts the strongest possible enemy response
            across all unfilled enemy roles.

    champion_pool:
        Candidate champions for our role.

    enemy_candidate_pool:
        Candidate champions the enemy may pick in minimax mode.
        If None, champion_pool is used.

    excluded_champions:
        Champions already picked or banned.
    """
    if champion_pool is None:
        champion_pool = []

    if enemy_candidate_pool is None:
        enemy_candidate_pool = champion_pool

    if excluded_champions is None:
        excluded_champions = set()
    else:
        excluded_champions = set(excluded_champions)

    role_to_fill = role_to_fill.lower()

    ally_team = dict(ally_team)
    enemy_team = dict(enemy_team)

    candidates = get_candidates_for_role(
        df_indexed=df_indexed,
        role_to_fill=role_to_fill,
        champion_pool=champion_pool,
        excluded_champions=excluded_champions,
    )

    if not candidates:
        return []

    enemy_filled_roles = set(enemy_team.keys())
    unfilled_enemy_roles = [
        role for role in ALL_ROLES
        if role not in enemy_filled_roles
    ]

    results = []

    for candidate_champ in candidates:
        simulated_ally_team = dict(ally_team)
        simulated_ally_team[role_to_fill] = candidate_champ

        base_log_odds, base_delta = score_candidate_pick(
            df_indexed=df_indexed,
            candidate_champ=candidate_champ,
            role_to_fill=role_to_fill,
            ally_team=simulated_ally_team,
            enemy_team=enemy_team,
            ally_synergy_weight=ally_synergy_weight,
            counter_weight=counter_weight,
            aggregation=aggregation,
        )

        if pick_strategy == "Maximize":
            final_log_odds = base_log_odds
            final_delta = base_delta

        elif pick_strategy == "MinimaxAllRoles":
            worst_enemy_log_odds = 0.0
            worst_enemy_delta = 0.0

            for enemy_role in unfilled_enemy_roles:
                for enemy_candidate in enemy_candidate_pool:
                    if enemy_candidate in excluded_champions:
                        continue
                    if enemy_candidate == candidate_champ:
                        continue

                    simulated_enemy_team = dict(enemy_team)
                    simulated_enemy_team[enemy_role] = enemy_candidate

                    enemy_log_odds, enemy_delta = score_enemy_candidate_pick(
                        df_indexed=df_indexed,
                        enemy_candidate=enemy_candidate,
                        enemy_role=enemy_role,
                        enemy_team=simulated_enemy_team,
                        ally_team=simulated_ally_team,
                        enemy_synergy_weight=enemy_synergy_weight,
                        enemy_counter_weight=enemy_counter_weight,
                        aggregation=aggregation,
                    )

                    worst_enemy_log_odds = max(worst_enemy_log_odds, enemy_log_odds)
                    worst_enemy_delta = max(worst_enemy_delta, enemy_delta)

            final_log_odds = base_log_odds - minimax_weight * worst_enemy_log_odds
            final_delta = base_delta - minimax_weight * worst_enemy_delta

        else:
            final_log_odds = base_log_odds
            final_delta = base_delta

        results.append((candidate_champ, final_log_odds, final_delta))

    return results