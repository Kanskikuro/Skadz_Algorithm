import pymc as pm
import pandas as pd
import numpy as np
import arviz as az  # For summaries and plots
import os
from math import log
import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim as optim
from scipy.special import expit, logit  # expit is the sigmoid function

def canonicalize_row(row):
    # Only sort for non-directional types
    if row['type'] == 'Synergy':
        # For synergy, order doesn't matter; sort lexicographically
        pair1 = (row['champ1'], row['role1'])
        pair2 = (row['champ2'], row['role2'])
        sorted_pairs = sorted([pair1, pair2])
        return (*sorted_pairs[0], *sorted_pairs[1], row['type'])
    else:
        # For directional types, preserve order
        return (row['champ1'], row['role1'], row['champ2'], row['role2'], row['type'])

def reduce_symmetry(data):
    data['canonical'] = data.apply(canonicalize_row, axis=1)
    
    def aggregate_group(group):
        total_samples = group['sample_size'].sum()
        if total_samples > 0:
            # Weighted average for 'win_rate'
            win_rate = np.average(group['win_rate'], weights=group['sample_size'])
            # Weighted average (or your choice) for 'delta'
            delta_agg = np.average(group['delta'], weights=group['sample_size'])
        else:
            # Fallback to simple mean if you have zero total_samples
            win_rate = group['win_rate'].mean()
            delta_agg = group['delta'].mean()

        return pd.Series({
            'win_rate': win_rate,
            'sample_size': total_samples,
            'delta': delta_agg  # Now included in the returned Series
        })

    # Perform groupby on the 'canonical' row signature and 'type'
    reduced_data = (
        data.groupby(['canonical', 'type'], as_index=False)
            .apply(aggregate_group, include_groups=False)
    )

    # Re-expand columns from the 'canonical' tuple
    canonical_expanded = pd.DataFrame(
        reduced_data['canonical'].tolist(),
        columns=['champ1', 'role1', 'champ2', 'role2', '_ignored_type']
    )
    canonical_expanded.drop(columns=['_ignored_type'], inplace=True)

    # Combine the aggregated results with the expanded canonical columns
    reduced_data = pd.concat([canonical_expanded, reduced_data.drop(columns=['canonical'])], axis=1)

    return reduced_data

def win_rate_to_log_odds(win_rate):
    p = win_rate / 100.0
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return log(p / (1 - p))

def calculate_bayesian_adjusted_win_rate(df, global_win_rate=52.0, m=200):
    df['win_rate_shrunk_bayes'] = (
        (df['win_rate'] * df['sample_size'] + global_win_rate * m) / (df['sample_size'] + m)
    )
    return df

def calculate_bayesian_adjusted_delta(df, global_delta=0.0, m=200):
    """
    Shrink the 'delta' column towards a prior (default 0.0) using a 'm' pseudo-sample size.
    """
    # Only apply if 'delta' is in the dataframe
    if 'delta' not in df.columns:
        return df

    df['delta_shrunk_bayes'] = (
        (df['delta'] * df['sample_size'] + global_delta * m) / (df['sample_size'] + m)
    )
    return df

def run_advi(data):
    """
    Build and fit a hierarchical Bayesian model that accounts for:
      - Synergy vs. counter interactions (type_idx)
      - Champion-level random effects (champion_idx)
      - Champion-role interaction effects
    Returns the data with an added column 'win_rate_shrunk_advi'.
    """
    
    # Encode synergy/counter
    data['type_idx'] = data['type'].astype('category').cat.codes
    n_types = data['type_idx'].nunique()  # Should be 2

    # Champion indices
    all_champs = pd.Series(pd.concat([data['champ1'], data['champ2']], ignore_index=True).unique())
    champ_idx_map = {champ: i for i, champ in enumerate(all_champs)}
    data['champ1_idx'] = data['champ1'].map(champ_idx_map)
    data['champ2_idx'] = data['champ2'].map(champ_idx_map)
    n_champions = len(champ_idx_map)

    # Role indices
    all_roles = pd.Series(pd.concat([data['role1'], data['role2']], ignore_index=True).unique())
    role_idx_map = {role: i for i, role in enumerate(all_roles)}
    data['role1_idx'] = data['role1'].map(role_idx_map)
    data['role2_idx'] = data['role2'].map(role_idx_map)
    n_roles = len(role_idx_map)

    # If you have no 'wins' / 'games' columns yet:
    data['wins'] = (data['win_rate'] / 100.0) * data['sample_size']
    data['games'] = data['sample_size']

    # Convert columns to arrays
    type_idx = data['type_idx'].values
    champ1_idx = data['champ1_idx'].values
    champ2_idx = data['champ2_idx'].values
    role1_idx = data['role1_idx'].values
    role2_idx = data['role2_idx'].values
    wins = data['wins'].values
    games = data['games'].values

    with pm.Model() as model:
        # Type-specific baseline
        baseline = pm.Normal('baseline', mu=0, sigma=0.2, shape=n_types)

        # Champion-level effects
        raw_champ = pm.Normal('raw_champ', mu=0, sigma=1, shape=(n_types, n_champions))
        sigma_champ = pm.HalfNormal('sigma_champ', sigma=0.1)
        champ_effect = pm.Deterministic('champ_effect', raw_champ * sigma_champ)

        # Champion-role interaction
        raw_cr = pm.Normal('raw_cr', mu=0, sigma=1, shape=(n_types, n_champions, n_roles))
        sigma_cr = pm.HalfNormal('sigma_cr', sigma=0.05)
        cr_effect = pm.Deterministic('cr_effect', raw_cr * sigma_cr)

        # Logit
        logit_p = (
            baseline[type_idx]
            + champ_effect[type_idx, champ1_idx] 
            + champ_effect[type_idx, champ2_idx]
            + cr_effect[type_idx, champ1_idx, role1_idx]
            + cr_effect[type_idx, champ2_idx, role2_idx]
        )
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))

        obs = pm.Binomial('obs', n=games, p=p, observed=wins)

        # Fit
        approx = pm.fit(n=20000, method='fullrank_advi')
        posterior = approx.sample(draws=500)

    # Posterior means
    baseline_mean = posterior.posterior['baseline'].mean(dim=['chain','draw']).values
    champ_mean = posterior.posterior['champ_effect'].mean(dim=['chain','draw']).values
    cr_mean = posterior.posterior['cr_effect'].mean(dim=['chain','draw']).values

    # Row-by-row
    row_baseline = baseline_mean[type_idx]
    row_champ1 = champ_mean[type_idx, champ1_idx]
    row_champ2 = champ_mean[type_idx, champ2_idx]
    row_cr1 = cr_mean[type_idx, champ1_idx, role1_idx]
    row_cr2 = cr_mean[type_idx, champ2_idx, role2_idx]
    logit_p_mean = row_baseline + row_champ1 + row_champ2 + row_cr1 + row_cr2

    data['win_rate_shrunk_advi'] = expit(logit_p_mean) * 100
    return data

def run_hierarchical(data):
    data['champ1_role'] = data['champ1'] + "_" + data['role1']
    data['champ2_role'] = data['champ2'] + "_" + data['role2']

    unique_combos = pd.unique(data[['champ1_role', 'champ2_role']].values.ravel('K'))
    combo_idx = {combo: i for i, combo in enumerate(unique_combos)}

    data['champ1_role_idx'] = data['champ1_role'].map(combo_idx)
    data['champ2_role_idx'] = data['champ2_role'].map(combo_idx)

    wins = (data['win_rate'] / 100) * data['sample_size']
    games = data['sample_size'].values
    champ1_role_idx = data['champ1_role_idx'].values
    champ2_role_idx = data['champ2_role_idx'].values

    with pm.Model() as hierarchical_model:
        mu = pm.Normal('mu', mu=0.5, sigma=0.05)
        sigma = pm.HalfNormal('sigma', sigma=0.1)

        raw_effect = pm.Normal('raw_effect', mu=0, sigma=0.1, shape=len(unique_combos))
        champ_role_effect = pm.Deterministic('champ_role_effect', mu + sigma * raw_effect)

        matchup_effect = pm.Normal('matchup_effect', mu=0, sigma=0.2, shape=(len(unique_combos), len(unique_combos)))

        total_effect = (champ_role_effect[champ1_role_idx] - champ_role_effect[champ2_role_idx] +
                        matchup_effect[champ1_role_idx, champ2_role_idx])

        p = pm.math.sigmoid(total_effect)
        obs = pm.Binomial('obs', n=games, p=p, observed=wins)

        trace = pm.sample()

        posterior_means = az.summary(trace, var_names=['champ_role_effect'])['mean'].values

        log_odds_diff = posterior_means[champ1_role_idx] - posterior_means[champ2_role_idx]
        win_probabilities = expit(log_odds_diff)

        data['win_rate_shrunk_hierarchical'] = win_probabilities * 100
        data['log_odds_hierarchical'] = log_odds_diff
    return data

def compute_expected_logodds(row, baseline_dict1, baseline_dict2, global_avg=0.52):
    key1 = (row['champ1'], row['role1'])
    p1 = baseline_dict1.get(key1, 0.52)
    key2 = (row['champ2'], row['role2'])
    p2 = baseline_dict2.get(key2, 0.52)
    r1 = logit(p1)
    r2 = logit(p2)
    if row['type'] == 'Synergy':
        global_logit = logit(global_avg)
        expected = (r1 + r2) - global_logit
    else:
        expected = r1 - r2
    return expected

def model(E, n, wins, matchup_index, matchup_sign, num_matchups):
    sigma_delta = pyro.sample("sigma_delta", dist.HalfNormal(torch.tensor(1.0)))
    delta_unique = pyro.sample(
        "delta_unique",
        dist.Normal(torch.zeros(num_matchups), sigma_delta * torch.ones(num_matchups)).to_event(1)
    )
    # For each observation, get the matchup-specific delta (with proper sign)
    delta = delta_unique[matchup_index] * matchup_sign
    mu = E + delta
    p = torch.sigmoid(mu)

    # Wrap the observations in a plate to indicate independence
    with pyro.plate("data", E.shape[0]):
        pyro.sample("obs", dist.Binomial(total_count=n, probs=p), obs=wins)

def guide(E, n, wins, matchup_index, matchup_sign, num_matchups):
    sigma_delta_loc = pyro.param("sigma_delta_loc", torch.tensor(1.0),
                                   constraint=pyro.distributions.constraints.positive)
    sigma_delta = pyro.sample("sigma_delta", dist.Delta(sigma_delta_loc))
    
    delta_loc = pyro.param("delta_loc", torch.zeros(num_matchups))
    delta_scale = pyro.param("delta_scale", 0.1 * torch.ones(num_matchups),
                             constraint=pyro.distributions.constraints.positive)
    pyro.sample(
        "delta_unique",
        dist.Normal(delta_loc, delta_scale).to_event(1)
    )

def create_matchup_index_and_sign(df):
    matchup_index = []
    matchup_sign = []
    matchup_dict = {}
    next_index = 0
    for i, row in df.iterrows():
        if row['type'] == 'Synergy':
            # For synergy, the order doesn't matter; use a sorted key and set sign to +1 always.
            key = ('Synergy',) + tuple(sorted([(row['champ1'], row['role1']), (row['champ2'], row['role2'])]))
            sign = 1
        else:  # For Counter type matchups.
            # Create a key that includes the type and the sorted champion–role pairs.
            key = ('Counter',) + tuple(sorted([(row['champ1'], row['role1']), (row['champ2'], row['role2'])]))
            # Determine sign: if (champ1, role1) is the first element in the sorted tuple, sign is +1; else -1.
            sign = 1 if (row['champ1'], row['role1']) == key[1] else -1
        if key not in matchup_dict:
            matchup_dict[key] = next_index
            next_index += 1
        matchup_index.append(matchup_dict[key])
        matchup_sign.append(sign)
    df["matchup_index"] = matchup_index
    df["matchup_sign"] = matchup_sign
    return df, next_index

def main(model_type='hierarchical', m=200):
    reduced_file = 'data/reduced_matchups.csv'
    output_file = 'data/matchups_shrunk.csv'
    unique_key_columns = ['champ1', 'role1', 'champ2', 'role2', 'type']

    if os.path.exists(reduced_file):
        print(f"Loading reduced data from {reduced_file}")
        data = pd.read_csv(reduced_file)
    else:
        print("Calculating delta column...")
        data = pd.read_csv('data/matchups.csv')

        # === Compute baseline for champ1, role1 ===
        baseline_dict1 = (
            data.groupby(['champ1', 'role1'])
                .apply(lambda g: np.average(g['win_rate'], weights=g['sample_size']), include_groups=False)
                .divide(100.0)  # convert percentage to fraction
                .to_dict()
        )

        # === Compute baseline for champ2, role2 ===
        # Since win_rate is given from champ1's perspective, for champ2 we take 100 - win_rate.
        baseline_dict2 = (
            data.groupby(['champ2', 'role2'])
                .apply(lambda g: np.average(100 - g['win_rate'], weights=g['sample_size']), include_groups=False)
                .divide(100.0)  # convert percentage to fraction
                .to_dict()
        )

        # Create the expected_logodds column
        data["expected_logodds"] = data.apply(lambda row: compute_expected_logodds(row, baseline_dict1, baseline_dict2), axis=1)

        # Compute observed wins (convert win_rate to fraction and multiply by sample_size)
        data["wins"] = (data["win_rate"] / 100.0 * data["sample_size"]).round().astype(int)
        data["n"] = data["sample_size"]

        # --- Convert relevant columns to torch tensors ---
        E_tensor = torch.tensor(data["expected_logodds"].values, dtype=torch.float32)
        n_tensor = torch.tensor(data["n"].values, dtype=torch.float32)
        wins_tensor = torch.tensor(data["wins"].values, dtype=torch.float32)

        print("Champ1 baseline dictionary:")
        print(baseline_dict1)
        print("Champ2 baseline dictionary:")
        print(baseline_dict2)

        data, num_matchups = create_matchup_index_and_sign(data)
        # Convert the new columns to torch tensors (make sure they have the appropriate dtype, e.g. long for indices)
        matchup_index_tensor = torch.tensor(data["matchup_index"].values, dtype=torch.long)
        matchup_sign_tensor = torch.tensor(data["matchup_sign"].values, dtype=torch.float32)

        # --- Inference using SVI ---
        optimizer = optim.Adam({"lr": 0.01})
        svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=pyro.infer.Trace_ELBO())

        num_steps = 5000
        for step in range(num_steps):
            loss = svi.step(E_tensor, n_tensor, wins_tensor, matchup_index_tensor, matchup_sign_tensor, num_matchups)
            if step % 500 == 0:
                print(f"Step {step}: Loss = {loss:.2f}")

        # --- Extract posterior estimates for the matchup effects delta ---
        delta_estimates = pyro.param("delta_loc").detach().numpy()
        
        # Ensure matchup_index and matchup_sign are numpy arrays.
        matchup_index_arr = data["matchup_index"].to_numpy()
        matchup_sign_arr = data["matchup_sign"].to_numpy()

        # Use vectorized indexing to map the unique delta estimates to each observation,
        # then multiply by the appropriate sign.
        delta_estimates_obs = delta_estimates[matchup_index_arr] * matchup_sign_arr

        # Now assign the per-observation delta estimates back to the DataFrame.
        data["delta"] = delta_estimates_obs

        print("Reducing dataset symmetry...")
        # (Assuming reduce_symmetry is a function you've defined elsewhere.)
        data = reduce_symmetry(data)
        data.to_csv(reduced_file, index=False)
        print(f"Reduced data saved to {reduced_file}")

    # Load existing results if available and merge only model-specific columns
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        # Identify columns that are model-specific (not in unique_key_columns or base columns)
        model_cols = [col for col in existing_data.columns 
                      if col not in unique_key_columns + ['win_rate', 'sample_size', 'delta']]
        # Merge these specific columns into our data
        merge_cols = unique_key_columns + model_cols
        # Also keep 'delta' if it existed in existing_data
        if 'delta' in existing_data.columns and 'delta' not in merge_cols:
            merge_cols.append('delta')
        data = data.merge(existing_data[merge_cols], on=unique_key_columns, how='left')

    # Compute model-specific results
    if model_type == 'bayesian':
        # Bayesian shrink for win_rate
        data = calculate_bayesian_adjusted_win_rate(data.copy(), global_win_rate=50.0, m=m)
        data['log_odds_bayes'] = data['win_rate_shrunk_bayes'].apply(win_rate_to_log_odds)

        # Bayesian shrink for delta, if present
        if 'delta' in data.columns:
            data = calculate_bayesian_adjusted_delta(data, global_delta=0.0, m=m)

    elif model_type == 'advi':
        data = run_advi(data.copy())
        data['log_odds_advi'] = data['win_rate_shrunk_advi'].apply(win_rate_to_log_odds)

    elif model_type == 'hierarchical':
        data = run_hierarchical(data.copy())
        # log_odds_hierarchical is computed inside run_hierarchical

    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main(model_type='bayesian')
