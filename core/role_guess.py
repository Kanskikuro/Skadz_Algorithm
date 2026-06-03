import numpy as np
from scipy.optimize import linear_sum_assignment

from core.enums import ROLES, Role


###############################################################################
# Hungarian-based guess of roles for enemy champions
###############################################################################


def _role_value(role) -> str:
    return role.value if isinstance(role, Role) else str(role).lower()


NORMALIZED_ROLES = [_role_value(role) for role in ROLES]


def normalize_champion_name(name: str) -> str:
    return str(name).strip().lower()


def build_priors_lookup(priors_df) -> dict[str, dict[str, float]]:
    """
    Builds a fast lookup:

        {
            "yasuo": {"top": 0.05, "jungle": 0.00, ...},
            ...
        }
    """
    required_columns = ["champion_name", *NORMALIZED_ROLES]

    missing = [col for col in required_columns if col not in priors_df.columns]
    if missing:
        raise ValueError(f"Missing required prior columns: {missing}")

    lookup: dict[str, dict[str, float]] = {}

    for _, row in priors_df.iterrows():
        champ_key = normalize_champion_name(row["champion_name"])

        role_probs: dict[str, float] = {}

        for role in NORMALIZED_ROLES:
            try:
                role_probs[role] = max(float(row[role]), 0.0)
            except (TypeError, ValueError, KeyError):
                role_probs[role] = 0.0

        lookup[champ_key] = role_probs

    return lookup


def _clean_champion_list(enemy_champs: list[str]) -> list[str]:
    cleaned_champs: list[str] = []
    seen: set[str] = set()

    for champ in enemy_champs:
        champ_name = str(champ).strip()

        if not champ_name:
            continue

        champ_key = normalize_champion_name(champ_name)

        if champ_key in seen:
            continue

        seen.add(champ_key)
        cleaned_champs.append(champ_name)

    return cleaned_champs


def get_role_probabilities_for_champion(
    champ: str,
    priors_repo,
    roles: list[str] | None = None,
    fallback_uniform: bool = True,
) -> list[tuple[str, float]]:
    """
    Returns sorted role probabilities for one champion:

        [
            ("bottom", 0.82),
            ("middle", 0.12),
            ...
        ]
    """
    if roles is None:
        roles = list(NORMALIZED_ROLES)
    else:
        roles = [_role_value(role) for role in roles]

    champ_name = str(champ).strip()

    if not champ_name:
        return []

    priors_df = priors_repo.get_df()
    priors_lookup = build_priors_lookup(priors_df)

    champ_key = normalize_champion_name(champ_name)

    if champ_key in priors_lookup:
        role_probs = priors_lookup[champ_key]
        probs = [max(float(role_probs.get(role, 0.0)), 0.0) for role in roles]
    elif fallback_uniform:
        probs = [1.0 / len(roles)] * len(roles)
    else:
        probs = [0.0] * len(roles)

    if sum(probs) <= 0:
        probs = [1.0 / len(roles)] * len(roles)

    total = sum(probs)

    normalized = [
        (role, prob / total)
        for role, prob in zip(roles, probs)
    ]

    normalized.sort(key=lambda item: item[1], reverse=True)
    return normalized


def guess_enemy_role_probabilities(
    enemy_champs: list[str],
    priors_repo,
    roles: list[str] | None = None,
    top_n: int | None = None,
    fallback_uniform: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """
    Returns role probability details per champion:

        {
            "Smolder": [("bottom", 0.82), ("middle", 0.10), ...],
            "Amumu": [("jungle", 0.91), ("support", 0.07), ...],
        }
    """
    if roles is None:
        roles = list(NORMALIZED_ROLES)
    else:
        roles = [_role_value(role) for role in roles]

    cleaned_champs = _clean_champion_list(enemy_champs)

    result: dict[str, list[tuple[str, float]]] = {}

    for champ in cleaned_champs:
        probabilities = get_role_probabilities_for_champion(
            champ=champ,
            priors_repo=priors_repo,
            roles=roles,
            fallback_uniform=fallback_uniform,
        )

        if top_n is not None:
            probabilities = probabilities[:top_n]

        result[champ] = probabilities

    return result


def guess_enemy_roles(
    enemy_champs: list[str],
    priors_repo,
    roles: list[str] | None = None,
    fallback_uniform: bool = True,
    epsilon: float = 1e-9,
) -> dict[str, str]:
    """
    Guesses enemy champion roles using Hungarian assignment.

    Returns:
        {
            "top": "Gragas",
            "middle": "Yasuo",
            "bottom": "Jinx",
        }
    """
    if roles is None:
        roles = list(NORMALIZED_ROLES)
    else:
        roles = [_role_value(role) for role in roles]

    cleaned_champs = _clean_champion_list(enemy_champs)

    if not cleaned_champs:
        return {}

    if len(cleaned_champs) > len(roles):
        raise ValueError(
            f"Cannot assign {len(cleaned_champs)} champions to only {len(roles)} roles."
        )

    priors_df = priors_repo.get_df()
    priors_lookup = build_priors_lookup(priors_df)

    n = len(cleaned_champs)
    m = len(roles)

    cost_matrix = np.zeros((n, m), dtype=float)

    for i, champ in enumerate(cleaned_champs):
        champ_key = normalize_champion_name(champ)

        if champ_key in priors_lookup:
            role_probs = priors_lookup[champ_key]
            probs = [max(float(role_probs.get(role, 0.0)), 0.0) for role in roles]
        elif fallback_uniform:
            probs = [1.0 / m] * m
        else:
            probs = [0.0] * m

        if sum(probs) <= 0:
            probs = [1.0 / m] * m

        total = sum(probs)
        probs = [p / total for p in probs]

        for j, p in enumerate(probs):
            safe_p = max(p, epsilon)
            cost_matrix[i, j] = -np.log(safe_p)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_roles: dict[str, str] = {}

    for champ_index, role_index in zip(row_ind, col_ind):
        champ = cleaned_champs[champ_index]
        role = roles[role_index]

        assigned_roles[role] = champ

    return assigned_roles