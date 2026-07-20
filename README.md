# Skadz Algorithm

Skadz Algorithm is a League of Legends champion recommendation tool. It suggests champion picks based on ally synergy, enemy counters, champion role priors, and matchup statistics.

The project is named after Skadz (Olav), who initiated and laid the foundation for the application.

## Features

- Champion recommendations for all five roles.
- Ally synergy and enemy counter evaluation using matchup data.
- Enemy role prediction using champion role priors.
- Tkinter-based graphical user interface.
- Recommendation scoring based on matchup strength, role likelihood, and draft context.
- Data update scripts for champions, icons, role priors, matchup data, and processed datasets.

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)
- Google Chrome, required by some scraping scripts

Install dependencies:

```bash
uv sync
```

## Running the Application

From the `champ_rec/` directory:

```bash
uv run python main.py
```

## Required Runtime Data

The application needs the following files to run:

```text
data/matchups_shrunk.csv
data/champion_priors.csv
data/champions.csv
data/champion_icons/
```

`matchups_shrunk.csv` contains the processed matchup data used by the recommender.

`champion_priors.csv` contains the probability of each champion appearing in each role.

`champions.csv` and `champion_icons/` are used by the UI.

## Updating the Dataset

To rebuild or update all data, run:

```bash
uv run python -m scripts.script
```

The full pipeline runs these steps:

```bash
uv run python -m scripts.download_champions_and_icons
uv run python -m scripts.download_champion_links
uv run python -m scripts.download_champion_priors
uv run python -m scripts.download_champion_matchups
uv run python -m scripts.process_dataset
```

The scripts download champion metadata, role priors, matchup statistics, and then process the raw data into smaller files used by the application.

## Project Structure

```text
champ_rec/
  main.py
  pyproject.toml
  uv.lock

  data/
    champions.csv
    champion_icons/
    champion_links.txt
    champion_priors.csv
    matchups.csv
    reduced_matchups.csv
    matchups_shrunk.csv

  core/
    enums.py
    recommend.py
    role_guess.py
    score.py
    repo/
    services/

  ui/
    app.py
    autocompleteEntryPopup.py
    components/

  scripts/
    __init__.py
    config.py
    script.py
    download_champions_and_icons.py
    download_champion_links.py
    download_champion_priors.py
    download_champion_matchups.py
    process_dataset.py
```

## How the Algorithm Works

The recommendation engine evaluates possible champion picks using matchup statistics and role probabilities.

The main steps are:

1. Load champion matchup data.
2. Load role priors for each champion.
3. Estimate enemy role assignments using the Hungarian algorithm.
4. Evaluate ally synergy and enemy counters.
5. Convert matchup values into log-odds-based draft scores.
6. Rank champions for each role while excluding already selected champions.

The displayed value should be interpreted as a recommendation score, not as a guaranteed game win-rate.

Two consumers build on this scoring logic:

- `core/score.py` scores a full team against its opponent (the "Ally/Enemy Draft Score" shown above each team).
- `core/recommend.py` scores one candidate champion's marginal contribution to a role, for ranking suggestions (including the `MinimaxAllRoles`/`Hybrid` worst-case enemy response check).

Both go through the same pair-lookup primitive (`lookup_pair_values` in `core/score.py`, backed by `MatchupRepository`'s cached dictionary lookup) so they can't independently drift on forward/reverse matchup resolution or on how duplicate data rows are averaged.

## Data Sources

The dataset is based on matchup and role data collected from Lolalytics.

The scraping scripts depend on the current Lolalytics page structure. If Lolalytics changes its layout, selectors or XPath expressions may need to be updated.

## Notes

- Running the update pipeline may overwrite generated data files.
- Scraping can take time and may fail if the website changes or rate-limits requests.
- The recommender depends heavily on the quality and freshness of the matchup dataset.
