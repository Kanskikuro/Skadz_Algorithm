import requests
from dataclasses import dataclass
from typing import List
import sqlite3
from urllib.parse import urljoin

from config import DB_PATH, ICONS_URL, CHAMP_SUMMARY_URL

"""
Download missing champion and their icons from CommunityDragon, and store them in a SQLite database.
"""


@dataclass
class Champion:
    id: int
    name: str
    alias: str = ""
    icon: bytes = None  # Will store the binary image

def get_champions() -> List[Champion]:
    resp = requests.get(CHAMP_SUMMARY_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [
        Champion(
            champ.get("id"),
            champ.get("name") or f"champion_{champ.get('id')}",
            champ.get("description", "")
        )
        for champ in data if "Doom Bot" not in champ.get("name", "")
    ]

def download_icon(champ_id: int) -> bytes: 
    url = urljoin(ICONS_URL, f"{champ_id}.png")
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.content
    else:
        print(f"[ERROR] Failed to download icon for champion {champ_id} - Status {resp.status_code}")
        return None

def save_to_db(champions: List[Champion], db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS champions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            alias TEXT,
            icon BLOB
        )
    """)
    
    # Idea : Check for len of champ match len of icons to avoid chekcing?
    for champ in champions:
        if champ.icon is None:
            champ.icon = download_icon(champ.id)

        cur.execute(
            "INSERT OR REPLACE INTO champions (id, name, alias, icon) VALUES (?, ?, ?, ?)",
            (champ.id, champ.name, champ.alias, champ.icon)
        )

    conn.commit()
    conn.close()

def main():
    champs = get_champions()
    save_to_db(champs)
    print(f"Saved {len(champs)} champions (with icons) into {DB_PATH}")

if __name__ == "__main__":
    main()
