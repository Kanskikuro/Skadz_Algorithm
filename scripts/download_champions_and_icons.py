from pathlib import Path

import pandas as pd
import requests


CHAMPIONS_CSV = Path("data/champions.csv")
ICON_DIR = Path("data/champion_icons")


def latest_ddragon_version() -> str:
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    versions = response.json()

    if not versions:
        raise RuntimeError("Could not fetch Data Dragon versions.")

    return versions[0]


def filename_sanitize(display_name: str) -> str:
    """
    Sanitizes champion names for local icon filenames.

    Examples:
        "Lee Sin" -> "lee_sin"
        "Dr. Mundo" -> "dr_mundo"
        "Nunu & Willump" -> "nunu__willump"
        "Kai'Sa" -> "kaisa"
    """
    return (
        str(display_name)
        .lower()
        .strip()
        .replace("'", "")
        .replace(".", "")
        .replace(" ", "_")
        .replace("&", "")
        .replace("-", "_")
    )


def fetch_champions(version: str) -> pd.DataFrame:
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()["data"]

    rows = []

    for riot_key, champ_data in data.items():
        display_name = champ_data["name"]
        champion_id = int(champ_data["key"])
        sanitized_name = filename_sanitize(display_name)

        rows.append(
            {
                "champion_id": champion_id,
                "display_name": display_name,
                "sanitized_name": sanitized_name,
                "alias": riot_key,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("display_name").reset_index(drop=True)

    return df


def save_champions_csv(df: pd.DataFrame) -> None:
    CHAMPIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CHAMPIONS_CSV, index=False)
    print(f"Saved champion metadata: {CHAMPIONS_CSV}")


def download_icon(version: str, riot_key: str, display_name: str, output_path: Path) -> bool:
    """
    Downloads one champion icon from Data Dragon.

    Uses riot_key first, then fallback keys.
    """
    compact_display = (
        display_name
        .replace(" ", "")
        .replace(".", "")
        .replace("'", "")
        .replace("&", "")
        .replace("-", "")
    )

    candidate_keys = [
        riot_key,
        display_name,
        compact_display,
    ]

    # Manual known exceptions, in case an older CSV has bad aliases.
    exceptions = {
        "FiddleSticks": "Fiddlesticks",
    }

    if riot_key in exceptions:
        candidate_keys.insert(0, exceptions[riot_key])

    seen = set()
    unique_keys = []

    for key in candidate_keys:
        key = str(key).strip()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_keys.append(key)

    for key in unique_keys:
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{key}.png"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f"Saved: {output_path} using key {key}")
            return True

    print(f"FAILED: {display_name} / {riot_key}. Tried keys: {unique_keys}")
    return False


def download_icons(version: str, df: pd.DataFrame, force: bool = False) -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in df.iterrows():
        display_name = str(row["display_name"]).strip()
        sanitized_name = str(row["sanitized_name"]).strip()
        riot_key = str(row["alias"]).strip()

        if not display_name or display_name.lower() == "nan":
            skipped += 1
            continue

        if not sanitized_name or sanitized_name.lower() == "nan":
            skipped += 1
            continue

        if not riot_key or riot_key.lower() == "nan":
            skipped += 1
            continue

        output_path = ICON_DIR / f"{sanitized_name}.png"

        if output_path.exists() and not force:
            skipped += 1
            continue

        ok = download_icon(
            version=version,
            riot_key=riot_key,
            display_name=display_name,
            output_path=output_path,
        )

        if ok:
            downloaded += 1
        else:
            failed += 1

    print()
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing/invalid: {skipped}")
    print(f"Failed: {failed}")


def main() -> None:
    version = latest_ddragon_version()
    print(f"Using Data Dragon version: {version}")

    df = fetch_champions(version)

    save_champions_csv(df)
    download_icons(version, df, force=False)


if __name__ == "__main__":
    main()