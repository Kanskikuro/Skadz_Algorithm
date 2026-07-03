from core.enums import ROLES, Role


def _role_value(role) -> str:
    return role.value if isinstance(role, Role) else str(role).lower()


ALL_ROLES = [_role_value(role) for role in ROLES]


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
    Looks up:

        champ_a, role_a, relation, champ_b, role_b

    Returns:
        (log_odds, delta, direction)

    direction:
        "forward" = A -> B was found
        "reverse" = B -> A was found
        None = no row found
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


def get_candidates_for_role(
    matchup_repo,
    role_to_fill,
    champion_pool=None,
    excluded_champions=None,
):
    """
    Finds champions that have data for the given role.
    Restricts champion_pool to champions that actually have data for this role.
    """
    if champion_pool is None:
        champion_pool = []

    if excluded_champions is None:
        excluded_champions = set()
    else:
        excluded_champions = set(excluded_champions)

    role_to_fill = str(role_to_fill).lower()

    if champion_pool:
        candidates = {
            champ
            for champ in champion_pool
            if role_to_fill in matchup_repo.roles_for_champion(champ)
        }
    else:
        candidates = {
            champ
            for champ, role in matchup_repo.champion_roles()
            if role == role_to_fill
        }

    return sorted(
        champ for champ in candidates
        if champ not in excluded_champions
    )


def score_candidate_pick(
    matchup_repo,
    method,
    candidate_champ,
    role_to_fill,
    ally_team,
    enemy_team,
    ally_synergy_weight=0.4,
    counter_weight=1.0,
):
    """
    Scores only the candidate champion's direct contribution:

        + synergy with allies
        + counters against known enemies
        - penalties when known enemies counter the candidate
    """
    total_log_odds = 0.0
    total_delta = 0.0

    for ally_role, ally_champ in ally_team.items():
        if ally_role == role_to_fill and ally_champ == candidate_champ:
            continue

        log_odds, delta, direction = lookup_pair_values(
            matchup_repo=matchup_repo,
            method=method,
            champ_a=candidate_champ,
            role_a=role_to_fill,
            relation="Synergy",
            champ_b=ally_champ,
            role_b=ally_role,
        )

        if direction is not None:
            total_log_odds += ally_synergy_weight * log_odds
            total_delta += ally_synergy_weight * delta

    for enemy_role, enemy_champ in enemy_team.items():
        log_odds, delta, direction = lookup_pair_values(
            matchup_repo=matchup_repo,
            method=method,
            champ_a=candidate_champ,
            role_a=role_to_fill,
            relation="Counter",
            champ_b=enemy_champ,
            role_b=enemy_role,
        )

        if direction == "forward":
            total_log_odds += counter_weight * log_odds
            total_delta += counter_weight * delta
        elif direction == "reverse":
            total_log_odds -= counter_weight * log_odds
            total_delta -= counter_weight * delta

    return total_log_odds, total_delta


def score_enemy_candidate_pick(
    matchup_repo,
    method,
    enemy_candidate,
    enemy_role,
    enemy_team,
    ally_team,
    enemy_synergy_weight=0.4,
    enemy_counter_weight=1.0,
):
    """
    Scores how threatening one possible enemy pick is:

        + synergy with existing enemy team
        + counters against our ally team
        - penalties if our ally team counters it
    """
    total_log_odds = 0.0
    total_delta = 0.0

    for existing_role, existing_champ in enemy_team.items():
        if existing_role == enemy_role:
            continue

        log_odds, delta, direction = lookup_pair_values(
            matchup_repo=matchup_repo,
            method=method,
            champ_a=enemy_candidate,
            role_a=enemy_role,
            relation="Synergy",
            champ_b=existing_champ,
            role_b=existing_role,
        )

        if direction is not None:
            total_log_odds += enemy_synergy_weight * log_odds
            total_delta += enemy_synergy_weight * delta

    for ally_role, ally_champ in ally_team.items():
        log_odds, delta, direction = lookup_pair_values(
            matchup_repo=matchup_repo,
            method=method,
            champ_a=enemy_candidate,
            role_a=enemy_role,
            relation="Counter",
            champ_b=ally_champ,
            role_b=ally_role,
        )

        if direction == "forward":
            total_log_odds += enemy_counter_weight * log_odds
            total_delta += enemy_counter_weight * delta
        elif direction == "reverse":
            total_log_odds -= enemy_counter_weight * log_odds
            total_delta -= enemy_counter_weight * delta

    return total_log_odds, total_delta


def should_use_minimax_for_role(
    role_to_fill,
    enemy_team,
    pick_strategy,
):
    """
    Strategy behavior:

    Maximize:
        Always use normal maximize.

    MinimaxAllRoles:
        Always use original full minimax.

    Hybrid:
        Use Maximize if the same-role enemy matchup is already known.
        Use MinimaxAllRoles if the same-role enemy matchup is unknown.
    """
    role_to_fill = str(role_to_fill).lower()

    if pick_strategy == "Maximize":
        return False

    if pick_strategy == "MinimaxAllRoles":
        return True

    if pick_strategy == "Hybrid":
        return role_to_fill not in enemy_team

    return False


def get_champion_scores_for_role(
    matchup_repo,
    method,
    role_to_fill,
    ally_team,
    enemy_team,
    pick_strategy="Hybrid",
    champion_pool=None,
    excluded_champions=None,
    enemy_candidate_pool=None,
    ally_synergy_weight=0.4,
    counter_weight=1.0,
    enemy_synergy_weight=0.4,
    enemy_counter_weight=1.0,
    minimax_weight=1.0,
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
            Original full minimax behavior. Scores candidate, then subtracts
            the strongest possible enemy response across all unfilled enemy roles.

        "Hybrid":
            Uses Maximize when the same-role enemy matchup is known.
            Uses MinimaxAllRoles when the same-role enemy matchup is unknown.
    """
    if champion_pool is None:
        champion_pool = []

    if enemy_candidate_pool is None:
        enemy_candidate_pool = champion_pool

    if excluded_champions is None:
        excluded_champions = set()
    else:
        excluded_champions = set(excluded_champions)

    role_to_fill = str(role_to_fill).lower()

    ally_team = {
        _role_value(role): champ
        for role, champ in dict(ally_team).items()
        if champ
    }

    enemy_team = {
        _role_value(role): champ
        for role, champ in dict(enemy_team).items()
        if champ
    }

    candidates = get_candidates_for_role(
        matchup_repo=matchup_repo,
        role_to_fill=role_to_fill,
        champion_pool=champion_pool,
        excluded_champions=excluded_champions,
    )

    if not candidates:
        return []

    use_minimax = should_use_minimax_for_role(
        role_to_fill=role_to_fill,
        enemy_team=enemy_team,
        pick_strategy=pick_strategy,
    )

    if use_minimax:
        enemy_filled_roles = set(enemy_team.keys())

        unfilled_enemy_roles = [
            role for role in ALL_ROLES
            if role not in enemy_filled_roles
        ]

        enemy_candidates_by_role = {
            enemy_role: [
                champ
                for champ in enemy_candidate_pool
                if champ not in excluded_champions
                and enemy_role in matchup_repo.roles_for_champion(champ)
            ]
            for enemy_role in unfilled_enemy_roles
        }
    else:
        unfilled_enemy_roles = []
        enemy_candidates_by_role = {}

    results = []

    for candidate_champ in candidates:
        simulated_ally_team = dict(ally_team)
        simulated_ally_team[role_to_fill] = candidate_champ

        base_log_odds, base_delta = score_candidate_pick(
            matchup_repo=matchup_repo,
            method=method,
            candidate_champ=candidate_champ,
            role_to_fill=role_to_fill,
            ally_team=simulated_ally_team,
            enemy_team=enemy_team,
            ally_synergy_weight=ally_synergy_weight,
            counter_weight=counter_weight,
        )

        if use_minimax:
            worst_enemy_log_odds = 0.0
            worst_enemy_delta = 0.0

            for enemy_role in unfilled_enemy_roles:
                for enemy_candidate in enemy_candidates_by_role[enemy_role]:
                    if enemy_candidate == candidate_champ:
                        continue

                    simulated_enemy_team = dict(enemy_team)
                    simulated_enemy_team[enemy_role] = enemy_candidate

                    enemy_log_odds, enemy_delta = score_enemy_candidate_pick(
                        matchup_repo=matchup_repo,
                        method=method,
                        enemy_candidate=enemy_candidate,
                        enemy_role=enemy_role,
                        enemy_team=simulated_enemy_team,
                        ally_team=simulated_ally_team,
                        enemy_synergy_weight=enemy_synergy_weight,
                        enemy_counter_weight=enemy_counter_weight,
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