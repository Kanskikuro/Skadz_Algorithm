import os
import requests
from urllib.parse import urljoin

ICONS_URL = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/"
CHAMP_SUMMARY_URL = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json"
DEST_FOLDER = "data/champion_icons"

def get_champion_ids_and_names():
    response = requests.get(CHAMP_SUMMARY_URL)
    response.raise_for_status()
    data = response.json()
    id_name_pairs = []
    for champ in data:
        champ_id = champ.get("id")
        champ_name = champ.get("name", f"champion_{champ_id}")
        id_name_pairs.append((champ_id, champ_name))
    return id_name_pairs

def download_images(id_name_pairs, dest_folder=DEST_FOLDER):
    os.makedirs(dest_folder, exist_ok=True)
    for champ_id, champ_name in id_name_pairs:
        img_url = urljoin(ICONS_URL, f"{champ_id}.png")
        file_path = os.path.join(dest_folder, f"{champ_name}.png")
        if os.path.isfile(file_path):
            print(f"[SKIP] Already downloaded: {champ_name}.png")
            continue
        print(f"[DOWNLOADING] {champ_name}.png...")
        resp = requests.get(img_url)
        if resp.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(resp.content)
        else:
            print(f"[ERROR] Failed to download {champ_name} (ID {champ_id}) - Status {resp.status_code}")

def main():
    try:
        pairs = get_champion_ids_and_names()
        print(f"Found {len(pairs)} valid champions.")
        download_images(pairs)
        print("Done!")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
