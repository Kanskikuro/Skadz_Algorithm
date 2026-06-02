import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from scripts.config import (
    CHAMPIONS_FILE,
    CHAMP_SUMMARY_URL,
    DEST_FOLDER,
    ICONS_URL,
)


@dataclass(frozen=True)
class Champion:
    id: int
    name: str
    sanitized_name: str
    alias: str


def sanitize_champion_name(name: str) -> str:
    """
    File/data-safe champion name.

    Examples:
    "Aurelion Sol" -> "aurelion_sol"
    "Kai'Sa" -> "kaisa"
    "Dr. Mundo" -> "dr_mundo"
    """
    name = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", name)


def get_champions() -> list[Champion]:
    response = requests.get(CHAMP_SUMMARY_URL, timeout=15)
    response.raise_for_status()

    data = response.json()
    champions: list[Champion] = []

    for champ in data:
        name = str(champ.get("name", "")).strip()

        if not name:
            continue

        if "Doom Bot" in name:
            continue

        champ_id = champ.get("id")
        if champ_id is None:
            continue

        champions.append(
            Champion(
                id=int(champ_id),
                name=name,
                sanitized_name=sanitize_champion_name(name),
                alias=str(champ.get("alias", "")).strip(),
            )
        )

    return sorted(champions, key=lambda c: c.sanitized_name)


def save_champions_csv(champions: list[Champion], path: Path = CHAMPIONS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "name", "sanitized_name", "alias"],
        )
        writer.writeheader()

        for champ in champions:
            writer.writerow(
                {
                    "id": champ.id,
                    "name": champ.name,
                    "sanitized_name": champ.sanitized_name,
                    "alias": champ.alias,
                }
            )

    print(f"Saved {len(champions)} champions to {path}")


def download_icon(champ_id: int) -> Optional[bytes]:
    url = urljoin(ICONS_URL, f"{champ_id}.png")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.RequestException as error:
        print(f"[WARN] Failed to download icon for champion {champ_id}: {error}")
        return None


def save_missing_icons(champions: list[Champion], icon_dir: Path = DEST_FOLDER) -> None:
    icon_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for champ in champions:
        icon_path = icon_dir / f"{champ.sanitized_name}.png"

        if icon_path.exists():
            skipped += 1
            continue

        icon_bytes = download_icon(champ.id)
        if icon_bytes is None:
            failed += 1
            continue

        icon_path.write_bytes(icon_bytes)
        downloaded += 1
        print(f"Downloaded icon: {icon_path.name}")

    print(
        f"Icons done. Downloaded={downloaded}, skipped_existing={skipped}, failed={failed}"
    )


def main() -> None:
    champions = get_champions()
    save_champions_csv(champions)
    save_missing_icons(champions)


if __name__ == "__main__":
    main()