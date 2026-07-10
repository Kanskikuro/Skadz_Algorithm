import re
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
    """
    Normalizes champion names for lookup.

    Examples:
        "Kha'Zix" -> "khazix"
        "Cho'Gath" -> "chogath"
        "Dr. Mundo" -> "drmundo"
    """
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _normalize_probability_value(value) -> float:
    """
    Converts prior values to decimal probabilities.

    Supports both:
        0.9949 -> 0.9949
        99.49  -> 0.9949
    """
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0

    probability = max(probability, 0.0)

    if probability > 1.0:
        probability /= 100.0

    return probability


_PRIORS_LOOKUP_ATTR = "_role_priors_lookup_cache"


def build_priors_lookup(priors_df) -> dict[str, dict[str, float]]:
    """
    Builds a fast lookup:

        {
            "khazix": {
                "top": 0.0,
                "jungle": 0.9949,
                "middle": 0.0,
                "bottom": 0.0,
                "support": 0.0,
            }
        }

    The result is cached on priors_df.attrs, since this is rebuilt on every
    role guess (e.g. every LCU poll tick) and the priors DataFrame doesn't
    change during a PriorsRepository's lifetime. The cache lives and dies
    with the DataFrame instance, so there's no separate cache to invalidate.
    """
    cached = priors_df.attrs.get(_PRIORS_LOOKUP_ATTR)

    if cached is not None:
        return cached

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
                role_probs[role] = _normalize_probability_value(row[role])
            except (TypeError, ValueError, KeyError):
                role_probs[role] = 0.0

        lookup[champ_key] = role_probs

    priors_df.attrs[_PRIORS_LOOKUP_ATTR] = lookup

    return lookup


def _clean_champion_list(champions: list[str]) -> list[str]:
    cleaned_champs: list[str] = []
    seen: set[str] = set()

    for champ in champions:
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
    Returns role probabilities for one champion in fixed role order.

    Example:
        [
            ("top", 0.0),
            ("jungle", 0.9949),
            ("middle", 0.0),
            ("bottom", 0.0),
            ("support", 0.0),
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
        probs = [
            max(float(role_probs.get(role, 0.0)), 0.0)
            for role in roles
        ]
    elif fallback_uniform:
        probs = [1.0 / len(roles)] * len(roles)
    else:
        probs = [0.0] * len(roles)

    total = sum(probs)

    if total <= 0:
        if fallback_uniform:
            probs = [1.0 / len(roles)] * len(roles)
        else:
            probs = [0.0] * len(roles)
    elif total > 1.000001:
        # Safety fallback for broken rows that sum above 100%.
        probs = [p / total for p in probs]

    return [
        (role, probability)
        for role, probability in zip(roles, probs)
    ]


def guess_enemy_role_probabilities(
    enemy_champs: list[str],
    priors_repo,
    roles: list[str] | None = None,
    top_n: int | None = None,
    fallback_uniform: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """
    Returns role probability details per champion.

    top_n is supported for compatibility. If top_n is None, all roles are returned.
    If top_n=5, all standard roles are still returned.
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
            probs = [
                max(float(role_probs.get(role, 0.0)), 0.0)
                for role in roles
            ]
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


def resolve_ally_role_assignments(
    ally_champs: list[str],
    ally_champs_by_role: dict[str, str],
    priors_repo,
    roles: list[str] | None = None,
) -> dict[str, str]:
    """
    Merges known ally lane assignments with a role guess for the rest.

    ally_champs_by_role holds champions whose lane is already known (e.g.
    from the LCU's assignedPosition) and is used as-is. Any remaining ally
    champions are assigned to the remaining roles via guess_enemy_roles.

    Returns:
        {
            "top": "Aatrox",       # from ally_champs_by_role
            "jungle": "LeeSin",    # from ally_champs_by_role
            "middle": "Yasuo",     # guessed
        }
    """
    if roles is None:
        roles = list(NORMALIZED_ROLES)
    else:
        roles = [_role_value(role) for role in roles]

    unassigned_champs = [
        champ for champ in ally_champs
        if champ not in ally_champs_by_role.values()
    ]

    unassigned_roles = [
        role for role in roles
        if role not in ally_champs_by_role
    ]

    guessed_roles = guess_enemy_roles(
        unassigned_champs,
        priors_repo,
        roles=unassigned_roles,
    )

    assignments = {
        role: champ
        for role, champ in ally_champs_by_role.items()
        if role in roles
    }
    assignments.update(guessed_roles)

    return assignments