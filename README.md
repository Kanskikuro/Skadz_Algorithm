# Skadz Algorithm

A League of Legends champion recommendation tool that suggests champion picks based on team synergy, enemy counters, role priors, and matchup statistics.

The project is named after Skadz (Olav), who initiated and laid the foundation for the application.

## Features

* Recommends champions for every role.
* Uses matchup data to evaluate champion synergies and counters.
* Predicts enemy role assignments using champion role priors.
* Provides a graphical user interface for selecting ally and enemy champions.
* Displays recommendations with win-rate and delta-based scoring metrics.
* Includes tools for updating champion data, lane links, matchup datasets, and processed recommendation data.

## Project Structure

```text
champ_rec/
  main.py
  script.py
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
    download_champions_and_icons.py
    download_champion_links.py
    download_dataset.py
    process_dataset.py
```

## Requirements

This project uses [`uv`](https://github.com/astral-sh/uv) for Python package management.

Install all dependencies with:

```bash
uv sync
```

If any Selenium, ChromeDriver, or browser-scraping dependencies fail, ensure that Google Chrome is installed on your system.

## Running the Application

From the project root directory, run:

```bash
uv run python main.py
```

## Updating the Dataset

Run the combined script:

```bash
uv run python -m scripts.script
```

Alternatively, delete any old data, then run the data pipeline in the following order:

```bash
uv run python -m scripts.download_champions_and_icons
uv run python -m scripts.download_champion_links
uv run python -m scripts.download_champion_priors
uv run python -m scripts.download_champion_matchups
uv run python -m scripts.process_dataset
```

## How the Algorithm Works

The recommendation engine evaluates champions using matchup and role data.

The process consists of:

1. Loading champion synergy and counter statistics.
2. Loading champion role priors that estimate the likelihood of a champion being played in Top, Jungle, Mid, Bottom, or Support.
3. Predicting enemy role assignments using the Hungarian algorithm.
4. Calculating team strength using log-odds derived from synergy and counter data.
5. Generating champion recommendations while accounting for ally synergy, enemy counters, already selected champions, and excluded picks.

## Notes

The scraping scripts depend on the current Lolalytics page structure. If Lolalytics changes its layout, selectors or XPath expressions in `download_dataset.py` may need to be updated.

Only the following files are required to run the application:

```text
data/matchups_shrunk.csv
data/champion_priors.csv
```
