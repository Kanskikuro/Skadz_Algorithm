import csv
import re
import subprocess
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from scripts.config import CHAMPION_PRIORS_FILE, LANES


TIER_LIST_URLS = {
    "top": "https://lolalytics.com/lol/tierlist/?lane=top&patch=30",
    "jungle": "https://lolalytics.com/lol/tierlist/?lane=jungle&patch=30",
    "middle": "https://lolalytics.com/lol/tierlist/?lane=middle&patch=30",
    "bottom": "https://lolalytics.com/lol/tierlist/?lane=bottom&patch=30",
    "support": "https://lolalytics.com/lol/tierlist/?lane=support&patch=30",
}

MAX_SCROLLS = 50


def sanitize_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", name)


def create_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    return webdriver.Chrome(service=service, options=options)


def parse_pick_rate(text: str) -> float:
    cleaned = (
        text.strip()
        .replace("%", "")
        .replace(",", "")
        .replace("+", "")
    )

    if not cleaned:
        return 0.0

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def scrape_priors() -> dict[str, dict[str, float]]:
    champion_data: dict[str, dict[str, float]] = {}

    driver = create_driver()

    try:
        for lane_name, url in TIER_LIST_URLS.items():
            print(f"Scraping priors for {lane_name}: {url}")

            try:
                driver.get(url)
            except TimeoutException:
                print(f"[WARN] Timeout loading {url}")
                continue

            body = driver.find_element(By.TAG_NAME, "body")

            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0

            while scroll_attempts < MAX_SCROLLS:
                body.send_keys("\ue00f")  # PageDown
                time.sleep(1.0)

                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break

                last_height = new_height
                scroll_attempts += 1

            i = 0

            while True:
                champion_container_xpath = f"/html/body/main/div[6]/div[{3 + i}]"
                champion_name_xpath = champion_container_xpath + "/div[3]/a"
                champion_pick_rate_xpath = champion_container_xpath + "/div[5]/div"

                try:
                    champion_name_elem = driver.find_element(By.XPATH, champion_name_xpath)
                    pick_rate_elem = driver.find_element(By.XPATH, champion_pick_rate_xpath)
                except Exception:
                    break

                champion_name = sanitize_name(champion_name_elem.text)
                pick_rate = parse_pick_rate(pick_rate_elem.text)

                if champion_name:
                    if champion_name not in champion_data:
                        champion_data[champion_name] = {
                            lane: 0.0 for lane in LANES
                        }

                    champion_data[champion_name][lane_name] = pick_rate

                i += 1

            print(f"Found {i} champion rows for {lane_name}")

    finally:
        driver.quit()

    return champion_data


def save_champion_priors(
    champion_data: dict[str, dict[str, float]],
    path: str | Path = CHAMPION_PRIORS_FILE,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["champion_name", "top", "jungle", "middle", "bottom", "support"])

        for champion_name in sorted(champion_data):
            row = [
                champion_name,
                champion_data[champion_name].get("top", 0.0),
                champion_data[champion_name].get("jungle", 0.0),
                champion_data[champion_name].get("middle", 0.0),
                champion_data[champion_name].get("bottom", 0.0),
                champion_data[champion_name].get("support", 0.0),
            ]
            writer.writerow(row)

    print(f"Saved {len(champion_data)} champion priors to {path}")


def main() -> None:
    champion_data = scrape_priors()

    if not champion_data:
        raise RuntimeError("No champion prior data was scraped.")

    save_champion_priors(champion_data)


if __name__ == "__main__":
    main()