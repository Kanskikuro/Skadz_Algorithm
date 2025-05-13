import os
import csv
import re
import math
import time
import random
import threading
import logging
from tqdm import tqdm
from queue import Queue, Empty
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    ElementClickInterceptedException,
    TimeoutException,
    NoSuchElementException,
)
from webdriver_manager.chrome import ChromeDriverManager

# Constants
CHAMPION_LINKS_FILE = "champion_links.txt"
CHAMPION_PRIORS_FILE = "champion_priors.csv"
CSV_FILE = 'matchups.csv'
NUM_THREADS = 5
QUEUE_TIMEOUT = 300  
MAX_SCROLLS = 50

progress_bar = None
progress_lock = threading.Lock()

# List of common User-Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/116.0.0.0 Safari/537.36",
]

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Thread %(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Suppress INFO logging from webdriver_manager
logging.getLogger("WDM").setLevel(logging.WARNING)
logging.getLogger("webdriver_manager").setLevel(logging.WARNING)

csv_lock = threading.Lock()

TIER_LIST_URLS = {
    "top": "https://lolalytics.com/lol/tierlist/?lane=top",
    "jungle": "https://lolalytics.com/lol/tierlist/?lane=jungle",
    "middle": "https://lolalytics.com/lol/tierlist/?lane=middle",
    "bottom": "https://lolalytics.com/lol/tierlist/?lane=bottom",
    "support": "https://lolalytics.com/lol/tierlist/?lane=support",
}

def save_links(links, filename=CHAMPION_LINKS_FILE):
    with open(filename, "w") as f:
        for link in links:
            f.write(link + "\n")
    logger.debug(f"Saved {len(links)} links to {filename}")

def load_links(filename=CHAMPION_LINKS_FILE):
    with open(filename, "r") as f:
        links = [line.strip() for line in f if line.strip()]
    logger.debug(f"Loaded {len(links)} links from {filename}")
    return links

def save_to_csv(data, filename=CSV_FILE):
    with csv_lock:
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['champ1', 'role1', 'type', 'champ2', 'role2', 'win_rate', 'delta', 'sample_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    logger.debug(f"Saved data to CSV: {data}")

def save_champion_priors(champion_data):
    """
    Writes champion_data to champion_priors.csv with columns:
    champion_name, top, jungle, middle, bottom, support
    """
    with open(CHAMPION_PRIORS_FILE, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["champion_name", "top", "jungle", "middle", "bottom", "support"])

        # Sort champions alphabetically
        for champ_name in sorted(champion_data.keys()):
            row = [
                champ_name,
                champion_data[champ_name]["top"],
                champion_data[champ_name]["jungle"],
                champion_data[champ_name]["middle"],
                champion_data[champ_name]["bottom"],
                champion_data[champ_name]["support"]
            ]
            writer.writerow(row)

def get_champ1_and_role1(wait):
    try:
        h1_element = wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(@class, 'font-bold')]")))
        h1_text = h1_element.text.strip()
        pattern = r"^(?P<champ1>.+?) Build, Runes & Counters for (?P<role1>\w+).+$"
        match = re.match(pattern, h1_text, re.IGNORECASE)
        if match:
            return match.group("champ1").strip(), match.group("role1").strip().lower()
        logger.warning("Regex pattern did not match.")
        return "Unknown", "unknown"
    except Exception as e:
        logger.error(f"Error extracting champion name and role: {e}")
        return "Unknown", "unknown"

def extract_data_with_selenium(driver, wait, view_type, champ1, role1):
    try:
        # Set the appropriate range for row indices based on view_type
        if view_type.lower() == "counter":
            row_indices = range(2, 7)  # 2 to 6 inclusive
        elif view_type.lower() == "synergy":
            row_indices = range(2, 6)  # 2 to 5 inclusive
        else:
            logger.error(f"Unsupported view_type: {view_type}")
            return

        for i in row_indices:
            # Build the absolute XPath for each row
            row_xpath = f"/html/body/main/div[6]/div[1]/div[{i}]"
            try:
                row = wait.until(EC.presence_of_element_located((By.XPATH, row_xpath)))
            except Exception as e:
                logger.error(f"Row at index {i} not found: {e}")
                continue

            # Extract role once per row using the row element
            try:
                lane_img = row.find_element(By.XPATH, ".//div[contains(@class, 'w-[80px]')]/img[contains(@alt, 'lane')]")
                lane_alt = lane_img.get_attribute("alt")
                role2 = lane_alt.split()[0].lower() if lane_alt else "unknown"
            except Exception as lane_e:
                logger.warning(f"Lane image not found in row {i}: {lane_e}")
                role2 = "unknown"

            # Locate champion sections within the row.
            champion_sections = row.find_elements(By.XPATH, ".//div[2]/div/div")

            for champion_section in champion_sections:
                try:
                    champ_img = champion_section.find_element(By.XPATH, ".//a/span/img")
                    champ2 = champ_img.get_attribute("alt")

                    win_rate_text_element = champion_section.find_element(By.XPATH, ".//div[1]/span")
                    win_rate_text = win_rate_text_element.text.strip()
                    win_rate = float(win_rate_text)

                    delta_text_element = champion_section.find_element(By.XPATH, ".//div[3]")
                    delta_text = delta_text_element.text.strip()
                    delta = float(delta_text)

                    sample_text_element = champion_section.find_element(By.XPATH, ".//div[5]")
                    sample_text = sample_text_element.text.strip().replace(',', '')
                    sample_size = int(sample_text)

                    data_to_save = {
                        "champ1": champ1,
                        "role1": role1,
                        "type": view_type,
                        "champ2": champ2,
                        "role2": role2,
                        "win_rate": win_rate,
                        "delta": delta,
                        "sample_size": sample_size
                    }
                    save_to_csv(data_to_save)
                    logger.debug(f"{view_type} - Enemy: {champ2}, role2: {role2}, win_rate: {win_rate}, Delta: {delta}, Sample: {sample_size}")

                except StaleElementReferenceException:
                    # We've reached an element that isn't loaded anymore; stop processing further champion sections in this row.
                    logger.debug(f"Encountered a stale element in row {i}, moving to the next row.")
                    break
                except Exception as champ_e:
                    logger.error(f"Failed to extract data from a champion section: {champ_e}")
                    continue

    except TimeoutException:
        logger.error(f"Timeout while waiting for elements in {view_type}.")

def process_champion(champion_link):
    logger.debug(f"Starting process for champion link: {champion_link}")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f'user-agent={user_agent}')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)

    try:
        logger.debug(f"Navigating to {champion_link}")
        driver.get(champion_link)
        wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(@class, 'font-bold')]")))
        logger.debug("Page loaded successfully.")

        champ1, role1 = get_champ1_and_role1(wait)
        logger.debug(f"Champion: {champ1}, Role: {role1}")

        logger.debug("Locating Strong counter button.")
        #delta_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@data-type='delta_counter']")))
        strong_counter_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@data-type='strong_counter']")))
        logger.debug("Scrolling down.")
        driver.execute_script("""
                const element = arguments[0];
                const offset = arguments[1];
                const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
                window.scrollTo({
                    top: elementPosition - offset,
                    behavior: 'auto'  // Changed from 'smooth' to 'auto'
                });
            """, strong_counter_button, 100)
        logger.debug("Strong counter button located. Clicking now.")
        strong_counter_button.click()
        wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[contains(@class, 'cursor-grab')]/div[contains(@class, 'flex')]/div")
        ))
        logger.debug("Champion rows loaded.")
        time.sleep(1)
        logger.debug("Extracting counter data.")
        extract_data_with_selenium(driver, wait, "Counter", champ1, role1)

        logger.debug("Locating Weak counter button.")
        weak_counter_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@data-type='weak_counter']")))
        logger.debug("Weak counter button located. Clicking now.")
        weak_counter_button.click()
        wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[contains(@class, 'cursor-grab')]/div[contains(@class, 'flex')]/div")
        ))
        logger.debug("Champion rows loaded.")
        time.sleep(1)
        logger.debug("Extracting counter data.")
        extract_data_with_selenium(driver, wait, "Counter", champ1, role1)

        logger.debug("Locating Good synergy button.")
        good_synergy_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@data-type='good_synergy']")))
        logger.debug("Good synergy button located. Clicking now.")
        good_synergy_button.click()
        wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[contains(@class, 'cursor-grab')]/div[contains(@class, 'flex')]/div")
        ))
        logger.debug("Champion rows loaded.")
        time.sleep(1)
        logger.debug("Extracting synergy data.")
        extract_data_with_selenium(driver, wait, "Synergy", champ1, role1)

        logger.debug("Locating Bad synergy button.")
        bad_synergy_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@data-type='bad_synergy']")))
        logger.debug("Bad synergy button located. Clicking now.")
        bad_synergy_button.click()
        wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[contains(@class, 'cursor-grab')]/div[contains(@class, 'flex')]/div")
        ))
        logger.debug("Champion rows loaded.")
        time.sleep(1)
        logger.debug("Extracting synergy data.")
        extract_data_with_selenium(driver, wait, "Synergy", champ1, role1)

        pause_duration = random.uniform(1, 3)
        logger.debug(f"Pausing for {pause_duration:.2f} seconds to respect rate limits.")
        time.sleep(pause_duration)
    except Exception as e:
        logger.error(f"Error processing champion {champion_link}: {e}")
    finally:
        driver.quit()
        logger.debug(f"Closed WebDriver for {champion_link}")

def worker(queue):
    global progress_bar
    while True:
        try:
            champion_link = queue.get(timeout=QUEUE_TIMEOUT)
            process_champion(champion_link)
            with progress_lock:
                if progress_bar:
                    progress_bar.update(1)
            queue.task_done()
        except Empty:
            logger.debug("Queue is empty. Worker is terminating.")
            break
        except Exception as e:
            logger.error(f"Worker encountered an error: {e}")
            queue.task_done()

def main():
    global progress_bar

    # Check if we already have champion links. If so, load them.
    if os.path.exists(CHAMPION_LINKS_FILE):
        all_links = load_links(CHAMPION_LINKS_FILE)
        logger.info(f"Loaded {len(all_links)} champion links from file.")
    else:
        # We do NOT have champion links -> scrape them from each tier list URL
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        user_agent = random.choice(USER_AGENTS)
        options.add_argument(f'user-agent={user_agent}')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 10)

        all_links_set = set()

        # Prepare a structure to store champion pick rates
        # champion_data[ch_name] = {"top": float, "jungle": float, "middle": float, "bottom": float, "support": float}
        champion_data = {}

        try:
            # Loop over each lane -> scrape champion build links & pick rates
            for lane_name, url in TIER_LIST_URLS.items():
                logger.info(f"Navigating to {url}")
                driver.get(url)

                # Scroll to load all champions
                body = driver.find_element(By.TAG_NAME, 'body')
                last_height = driver.execute_script("return document.body.scrollHeight")
                scroll_attempts = 0

                while scroll_attempts < MAX_SCROLLS:
                    body.send_keys(Keys.PAGE_DOWN)
                    time.sleep(2)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                    scroll_attempts += 1

                # Collect build links
                a_tags = driver.find_elements(By.XPATH, "//div[contains(@class, 'h-[52px]')]//a[contains(@href, 'build')]")
                for a in a_tags:
                    href = a.get_attribute("href")
                    if href and "lolalytics.com/lol/" in href and "build" in href:
                        all_links_set.add(href)

                # Gather champion name & pick rate
                # -------------------------------------------------------------
                # For this example, let's assume each champion row is:
                # /html/body/main/div[6]/div[3 + i]/div[3]/a/text()  => champion name
                # /html/body/main/div[6]/div[3 + i]/div[5]/div/text() => pick rate
                #
                # If your actual page is different, adapt the indexing below.
                i = 0
                while True:
                    # Build the XPaths for champion name & pick rate
                    champion_container_xpath = f"/html/body/main/div[6]/div[{3 + i}]"
                    champion_name_xpath      = champion_container_xpath + "/div[3]/a"
                    champion_pick_rate_xpath = champion_container_xpath + "/div[5]/div"

                    try:
                        champion_name_elem = driver.find_element(By.XPATH, champion_name_xpath)
                        pick_rate_elem     = driver.find_element(By.XPATH, champion_pick_rate_xpath)
                    except:
                        # Not found -> end of champion list for this lane
                        break

                    champion_name = champion_name_elem.text.strip()
                    pick_rate_str = pick_rate_elem.text.strip()
                    try:
                        pick_rate = float(pick_rate_str)
                    except ValueError:
                        pick_rate = 0.0

                    # Initialize champion in dictionary if new
                    if champion_name not in champion_data:
                        champion_data[champion_name] = {
                            "top": 0.0, "jungle": 0.0, "middle": 0.0,
                            "bottom": 0.0, "support": 0.0
                        }

                    # Store the pick rate for the current lane
                    champion_data[champion_name][lane_name] = pick_rate

                    i += 1

            # Convert links set to a list
            all_links = list(all_links_set)
            save_links(all_links)
            logger.info(f"Total Champions: {len(all_links)}")

            # Save champion pick rates to CSV
            save_champion_priors(champion_data)
            logger.info(f"Champion priors saved to {CHAMPION_PRIORS_FILE}")

        except Exception as e:
            logger.error(f"Error during initial scraping: {e}")
            all_links = []
        finally:
            driver.quit()
            logger.debug("Closed initial WebDriver.")

    # Now we have all_links (either loaded or scraped). Process them with a threaded queue.
    queue_obj = Queue()
    for link in all_links:
        queue_obj.put(link)

    progress_bar = tqdm(total=len(all_links), desc="Processing champions", unit="champ")

    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(queue_obj,), name=f"Worker-{i+1}")
        t.daemon = True
        t.start()
        threads.append(t)
        logger.debug(f"Started thread {t.name}")

    # Wait for all queue tasks to complete
    queue_obj.join()

    # Close the progress bar
    if progress_bar:
        progress_bar.close()

    logger.info("All champions have been processed.")

if __name__ == "__main__":
    main()