
import pandas as pd
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