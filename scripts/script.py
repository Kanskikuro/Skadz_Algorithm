from pathlib import Path

from scripts import download_champions_and_icons
from scripts import download_champion_links
from scripts import download_champion_priors
from scripts import download_champion_matchups
from scripts import process_dataset
from scripts.config import (
    CHAMPION_PRIORS_FILE,
    MATCHUPS_FILE,
    REDUCED_MATCHUPS_FILE,
    SHRUNK_MATCHUPS_FILE,
)


def remove_file(path: str | Path) -> None:
    path = Path(path)

    if path.exists():
        path.unlink()
        print(f"Removed old file: {path}")
    else:
        print(f"Skipped missing file: {path}")


def remove_old_dataset_files() -> None:
    files_to_remove = [
        CHAMPION_PRIORS_FILE,
        MATCHUPS_FILE,
        REDUCED_MATCHUPS_FILE,
        SHRUNK_MATCHUPS_FILE,
    ]

    for file_path in files_to_remove:
        remove_file(file_path)


def main() -> None:
    # Updates champion list and downloads only missing icons.
    download_champions_and_icons.main()

    # Updates/cleans champion lane links.
    download_champion_links.main()

    # Remove generated data that should be recreated from current scrape.
    remove_old_dataset_files()

    # Recreate role priors from current Lolalytics data.
    download_champion_priors.main()

    # Recreate raw matchup data.
    download_champion_matchups.main()

    # Recreate reduced/shrunk app-ready data.
    process_dataset.main()


if __name__ == "__main__":
    main()