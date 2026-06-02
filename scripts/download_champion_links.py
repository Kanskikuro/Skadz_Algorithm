import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from scripts.config import (
    BASE_URL,
    CHAMPION_LINKS_FILE,
    CHAMP_SUMMARY_URL,
    LANES,
)


def lolalytics_slug(name: str) -> str:
    """
    Lolalytics URL slug.

    Examples:
    "Aurelion Sol" -> "aurelionsol"
    "Kai'Sa" -> "kaisa"
    "Dr. Mundo" -> "drmundo"
    "Nunu & Willump" -> "nunuwillump"
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_champions_from_communitydragon() -> list[str]:
    response = requests.get(CHAMP_SUMMARY_URL, timeout=15)
    response.raise_for_status()

    data = response.json()
    champions: list[str] = []

    for champ in data:
        name = str(champ.get("name", "")).strip()

        if not name:
            continue

        if "Doom Bot" in name:
            continue

        champions.append(name)

    return sorted(set(champions), key=lolalytics_slug)


def make_lane_url(champion_name: str, lane: str) -> str:
    return BASE_URL.format(champ_name=lolalytics_slug(champion_name), lane=lane)


def load_existing_links(path: str | Path = CHAMPION_LINKS_FILE) -> set[str]:
    path = Path(path)

    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}

def is_direct_lane_build_url(url: str) -> bool:
    """
    Accept:
    https://lolalytics.com/lol/aatrox/build/?lane=top
    https://lolalytics.com/lol/aatrox/build/?lane=top&patch=30

    Reject:
    https://lolalytics.com/lol/aatrox/vs/ahri/build/?lane=top
    https://lolalytics.com/lol/aatrox/build/?lane=top&item=...
    https://lolalytics.com/lol/aatrox/build/?lane=top&keystone=...
    """
    parsed = urlparse(url)

    if parsed.netloc != "lolalytics.com":
        return False

    if "/vs/" in parsed.path:
        return False

    parts = parsed.path.strip("/").split("/")
    if len(parts) != 3:
        return False

    if parts[0] != "lol" or parts[2] != "build":
        return False

    query = parse_qs(parsed.query)

    allowed_keys = {"lane", "patch"}
    if not set(query.keys()).issubset(allowed_keys):
        return False

    lane_values = query.get("lane")
    if not lane_values:
        return False

    if lane_values[0] not in LANES:
        return False

    patch_values = query.get("patch")
    if patch_values and patch_values[0] != "30":
        return False

    return True

def save_links(links: set[str], path: str | Path = CHAMPION_LINKS_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_links = sorted(link for link in links if is_direct_lane_build_url(link))

    with path.open("w", encoding="utf-8", newline="") as file:
        for link in clean_links:
            file.write(link + "\n")

    print(f"Saved {len(clean_links)} clean links to {path}")


def main() -> None:
    champions = load_champions_from_communitydragon()
    existing_links = load_existing_links()

    expected_links: set[str] = {
        make_lane_url(champion, lane)
        for champion in champions
        for lane in LANES
    }

    missing_links = expected_links - existing_links
    final_links = existing_links.union(missing_links)

    print(f"Champions from CommunityDragon: {len(champions)}")
    print(f"Expected champion/lane links: {len(expected_links)}")
    print(f"Existing links: {len(existing_links)}")
    print(f"Missing links added: {len(missing_links)}")

    for link in sorted(missing_links):
        print(f"Adding missing link: {link}")

    save_links(final_links)
    print("Done.")


if __name__ == "__main__":
    main()