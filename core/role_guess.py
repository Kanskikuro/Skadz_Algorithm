import numpy as np
from scipy.optimize import linear_sum_assignment
from math import log
from core.repo import PriorsRepository

###############################################################################
# Hungarian-based guess of roles for enemy champions
###############################################################################
def guess_enemy_roles(enemy_champs: list[str], priors_repo: PriorsRepository):
    """
    enemy_champs: list of championName (strings), e.g. ['Gragas', 'Yasuo', ...].
    priors_df:    DataFrame with columns [champion_name, top, jungle, middle, bottom, support].
    
    Returns: dict { "top": champName, "jungle": champName, ... }
             from those champions by maximizing prior probabilities,
             allowing any typed champion to take any of the 5 roles.
    """
    priors_df = priors_repo.get_df()

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