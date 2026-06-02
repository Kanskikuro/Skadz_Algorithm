import csv
import logging
import random
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from scripts.config import CHAMPION_LINKS_FILE, MATCHUPS_FILE


# Safer than 5. Still faster than the old script because each worker reuses its browser.
NUM_THREADS = 3
QUEUE_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 60
WAIT_TIMEOUT = 20

# Randomized delays reduce the rigid bot-like timing pattern.
TAB_CLICK_DELAY_RANGE = (0.4, 0.9)
PAGE_DELAY_RANGE = (1.5, 3.5)

progress_bar = None
progress_lock = threading.Lock()
csv_lock = threading.Lock()

USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/15.1 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
]

FIELDNAMES = [
    "champ1",
    "role1",
    "type",
    "champ2",
    "role2",
    "win_rate",
    "delta",
    "sample_size",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Thread %(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("WDM").setLevel(logging.WARNING)
logging.getLogger("webdriver_manager").setLevel(logging.WARNING)


def load_links(path: str | Path = CHAMPION_LINKS_FILE) -> list[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: uv run python -m scripts.download_champion_links"
        )

    with path.open("r", encoding="utf-8") as file:
        links = [line.strip() for line in file if line.strip()]

    return sorted(set(links))


def save_many_to_csv(rows: list[dict], path: str | Path = MATCHUPS_FILE) -> None:
    """Write all rows from one champion page in one locked file operation."""
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with csv_lock:
        file_exists = path.exists()

        with path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerows(rows)


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
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")

    service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    return driver


def parse_float_text(text: str) -> float:
    cleaned = (
        text.strip()
        .replace("%", "")
        .replace("+", "")
        .replace("−", "-")
        .replace(",", "")
    )
    return float(cleaned)


def parse_int_text(text: str) -> int:
    return int(text.strip().replace(",", ""))


def get_champ1_and_role1(wait: WebDriverWait) -> tuple[str, str]:
    try:
        h1_element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//h1[contains(@class, 'font-bold')]")
            )
        )
        h1_text = h1_element.text.strip()

        # Example: "Aatrox Build, Runes & Counters for Top"
        marker = " Build, Runes & Counters for "
        if marker not in h1_text:
            logger.warning("Could not parse h1 text: %s", h1_text)
            return "Unknown", "unknown"

        champ, rest = h1_text.split(marker, 1)
        role = rest.split()[0].strip().lower()

        return champ.strip(), role

    except Exception as error:
        logger.error("Error extracting champion name and role: %s", error)
        return "Unknown", "unknown"


def extract_data_with_selenium(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    view_type: str,
    champ1: str,
    role1: str,
) -> list[dict]:
    rows: list[dict] = []

    if view_type.lower() == "counter":
        row_indices = range(2, 7)
    elif view_type.lower() == "synergy":
        row_indices = range(2, 6)
    else:
        logger.error("Unsupported view_type: %s", view_type)
        return rows

    for i in row_indices:
        row_xpath = f"/html/body/main/div[6]/div[1]/div[{i}]"

        try:
            row = wait.until(EC.presence_of_element_located((By.XPATH, row_xpath)))
        except Exception as error:
            logger.debug("Row at index %s not found: %s", i, error)
            continue

        try:
            lane_img = row.find_element(
                By.XPATH,
                ".//div[contains(@class, 'w-[80px]')]/img[contains(@alt, 'lane')]",
            )
            lane_alt = lane_img.get_attribute("alt")
            role2 = lane_alt.split()[0].lower() if lane_alt else "unknown"
        except Exception:
            role2 = "unknown"

        champion_sections = row.find_elements(By.XPATH, ".//div[2]/div/div")

        for champion_section in champion_sections:
            try:
                champ_img = champion_section.find_element(By.XPATH, ".//a/span/img")
                champ2 = champ_img.get_attribute("alt")

                win_rate_text = champion_section.find_element(By.XPATH, ".//div[1]/span").text
                delta_text = champion_section.find_element(By.XPATH, ".//div[3]").text
                sample_text = champion_section.find_element(By.XPATH, ".//div[5]").text

                if not champ2:
                    continue

                rows.append(
                    {
                        "champ1": champ1,
                        "role1": role1,
                        "type": view_type,
                        "champ2": champ2,
                        "role2": role2,
                        "win_rate": parse_float_text(win_rate_text),
                        "delta": parse_float_text(delta_text),
                        "sample_size": parse_int_text(sample_text),
                    }
                )

            except StaleElementReferenceException:
                break
            except Exception as error:
                logger.debug("Failed to extract champion section: %s", error)
                continue

    return rows


def click_and_extract(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    data_type: str,
    view_type: str,
    champ1: str,
    role1: str,
) -> list[dict]:
    button = wait.until(
        EC.element_to_be_clickable((By.XPATH, f"//div[@data-type='{data_type}']"))
    )

    driver.execute_script(
        """
        const element = arguments[0];
        const offset = arguments[1];
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        window.scrollTo({ top: elementPosition - offset, behavior: 'auto' });
        """,
        button,
        100,
    )

    button.click()

    wait.until(
        EC.presence_of_all_elements_located(
            (
                By.XPATH,
                "//div[contains(@class, 'cursor-grab')]/div[contains(@class, 'flex')]/div",
            )
        )
    )

    # Safer than a fixed 0.25s delay. Random timing is less rigid and still faster than 1s.
    time.sleep(random.uniform(*TAB_CLICK_DELAY_RANGE))

    return extract_data_with_selenium(driver, wait, view_type, champ1, role1)


def process_champion(
    champion_link: str,
    driver: webdriver.Chrome,
    wait: WebDriverWait,
) -> None:
    rows: list[dict] = []

    try:
        logger.info("Processing %s", champion_link)

        driver.get(champion_link)

        wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//h1[contains(@class, 'font-bold')]")
            )
        )

        champ1, role1 = get_champ1_and_role1(wait)

        if champ1 == "Unknown" or role1 == "unknown":
            logger.warning("Skipping page with unknown champ/role: %s", champion_link)
            return

        rows.extend(
            click_and_extract(driver, wait, "strong_counter", "Counter", champ1, role1)
        )
        rows.extend(
            click_and_extract(driver, wait, "weak_counter", "Counter", champ1, role1)
        )
        rows.extend(
            click_and_extract(driver, wait, "good_synergy", "Synergy", champ1, role1)
        )
        rows.extend(
            click_and_extract(driver, wait, "bad_synergy", "Synergy", champ1, role1)
        )

        save_many_to_csv(rows)
        logger.info("Saved %s rows for %s %s", len(rows), champ1, role1)

        # Delay between champion/lane pages. This matters more for bot-ban risk than tab delay.
        time.sleep(random.uniform(*PAGE_DELAY_RANGE))

    except TimeoutException:
        logger.error("Timeout processing %s", champion_link)
    except Exception as error:
        logger.error("Error processing %s: %s", champion_link, error)


def worker(queue: Queue) -> None:
    global progress_bar

    driver = None

    try:
        # One browser per worker, reused for many pages.
        driver = create_driver()
        wait = WebDriverWait(driver, WAIT_TIMEOUT)

        while True:
            try:
                champion_link = queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                break

            try:
                process_champion(champion_link, driver, wait)
            finally:
                with progress_lock:
                    if progress_bar:
                        progress_bar.update(1)

                queue.task_done()

    except Exception as error:
        logger.error("Worker error: %s", error)

    finally:
        if driver is not None:
            driver.quit()


def main() -> None:
    global progress_bar

    all_links = load_links(CHAMPION_LINKS_FILE)

    if not all_links:
        raise RuntimeError(
            f"No links found in {CHAMPION_LINKS_FILE}. "
            "Run: uv run python -m scripts.download_champion_links"
        )

    logger.info("Loaded %s champion links.", len(all_links))

    queue_obj: Queue = Queue()

    for link in all_links:
        queue_obj.put(link)

    progress_bar = tqdm(
        total=len(all_links),
        desc="Processing champion pages",
        unit="page",
    )

    threads = []

    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=worker,
            args=(queue_obj,),
            name=f"Worker-{i + 1}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    queue_obj.join()

    for thread in threads:
        thread.join(timeout=5)

    if progress_bar:
        progress_bar.close()

    logger.info("All champion pages processed.")


if __name__ == "__main__":
    main()
