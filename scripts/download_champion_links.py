import sqlite3
from playwright.sync_api import sync_playwright
import time
from config import LANES, BASE_URL, DB_PATH

def fetch_links_for_champion(champion_name: str):
    """Fetch all build links for a single champion."""
    all_links = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for lane in LANES:
            url = BASE_URL.format(champion_name.lower(), lane)
            page.goto(url, timeout=60000)

            # Scroll to load all builds
            for _ in range(40):
                page.keyboard.press("PageDown")
                time.sleep(0.2)

            # Collect all build links
            a_tags = page.query_selector_all("a[href*='build']")
            for a in a_tags:
                href = a.get_attribute("href")
                if href and "lolalytics.com/lol/" in href:
                    all_links.add(href)

        browser.close()
    return list(all_links)


def save_links_to_db(champion_id: int, links: list, db_path="champions.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS champion_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            champion_id INTEGER NOT NULL,
            link TEXT NOT NULL,
            FOREIGN KEY (champion_id) REFERENCES champions(id)
        )
    """)

    # idea: check if there are 5 links for each champion ?
    for link in links:
        cur.execute(
            "INSERT INTO champion_links (champion_id, link) VALUES (?, ?)",
            (champion_id, link)
        )

    conn.commit()
    conn.close()


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # fetch all champions from DB
    cur.execute("SELECT id, name FROM champions")
    champions = cur.fetchall()
    conn.close()

    print(f"Found {len(champions)} champions in DB")

    for champ_id, champ_name in champions:
        print(f"Fetching links for {champ_name}...")
        links = fetch_links_for_champion(champ_name)
        print(f" → Found {len(links)} links")
        save_links_to_db(champ_id, links)

    print("Done!")


if __name__ == "__main__":
    main()
