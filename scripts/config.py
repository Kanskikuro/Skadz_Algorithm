from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

LANES = ["top", "jungle", "middle", "bottom", "support"]

BASE_URL = "https://lolalytics.com/lol/{champ_name}/build/?lane={lane}&patch=30"

CHAMPION_LINKS_FILE = DATA_DIR / "champion_links.txt"
CHAMPION_PRIORS_FILE = DATA_DIR / "champion_priors.csv"
CHAMPIONS_FILE = DATA_DIR / "champions.csv"

MATCHUPS_FILE = DATA_DIR / "matchups.csv"
REDUCED_MATCHUPS_FILE = DATA_DIR / "reduced_matchups.csv"
SHRUNK_MATCHUPS_FILE = DATA_DIR / "matchups_shrunk.csv"

ICONS_URL = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/"
    "global/default/v1/champion-icons/"
)

CHAMP_SUMMARY_URL = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/"
    "global/default/v1/champion-summary.json"
)

DEST_FOLDER = DATA_DIR / "champion_icons"