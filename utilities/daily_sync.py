"""
Daily Sermon Synchronizer
Runs daily to check for newly published sermons on The Standing Church archive
and channels, appends them to the index, and regenerates embeddings if needed.
"""

import json
import logging
import os
import sys
import time
from bs4 import BeautifulSoup
import requests

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SERMONS_PATH = os.path.join(BASE_DIR, "utilities", "sermons.jsonl")
BASE_URL = "https://www.thestandingchurch.com/sermons/"
PAGE_URL = "https://www.thestandingchurch.com/sermons/page/{}/"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


def load_existing_urls():
    if not os.path.exists(SERMONS_PATH):
        return set()
    urls = set()
    with open(SERMONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("url"):
                        urls.add(data["url"].rstrip("/"))
                except Exception:
                    pass
    return urls


def fetch(url):
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            return r
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
    return None


def scrape_sermon(url):
    r = fetch(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one(".wpfc-sermon-single-title")
    title = title_el.get_text(strip=True) if title_el else ""

    content = soup.select_one(".wpfc-sermon-single-main")
    description = ""
    if content:
        paras = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
        description = "\n\n".join(paras)

    audio_el = soup.select_one("a.wpfc-sermon-single-audio-download")
    audio_url = audio_el["href"] if audio_el and audio_el.get("href") else ""

    image_el = soup.select_one("img.wpfc-sermon-single-image-img")
    image_url = image_el["src"] if image_el and image_el.get("src") else ""

    preacher_el = soup.select_one("a[href*='preacher']")
    preacher = preacher_el.get_text(strip=True) if preacher_el else "Pastor Tara Akinkuade"

    date_el = soup.select_one("time[datetime]")
    date = date_el.get("datetime", date_el.get_text(strip=True)) if date_el else ""

    return {
        "title": title,
        "description": description,
        "audio_url": audio_url,
        "image_url": image_url,
        "preacher": preacher,
        "date": date,
        "url": url,
    }


def sync_latest(pages_to_check: int = 3):
    logging.info(f"Starting daily sync (checking latest {pages_to_check} pages)...")
    existing_urls = load_existing_urls()
    logging.info(f"Currently have {len(existing_urls)} existing sermons in database.")

    new_sermons = []

    for page in range(1, pages_to_check + 1):
        url = BASE_URL if page == 1 else PAGE_URL.format(page)
        r = fetch(url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"].rstrip("/") for a in soup.select("h3.wpfc-sermon-title a") if a.get("href")]

        for link in links:
            if link not in existing_urls:
                logging.info(f"Discovered new sermon: {link}")
                sermon_data = scrape_sermon(link)
                if sermon_data:
                    new_sermons.append(sermon_data)
                    existing_urls.add(link)
                time.sleep(1.5)

    if new_sermons:
        logging.info(f"Found {len(new_sermons)} new sermons! Appending to {SERMONS_PATH}...")
        # Prepend new sermons so they are at the top
        with open(SERMONS_PATH, "r", encoding="utf-8") as f:
            old_content = f.read()

        with open(SERMONS_PATH, "w", encoding="utf-8") as f:
            for s in new_sermons:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.write(old_content)

        logging.info("Sermons file successfully updated.")
        # Remove old cached embeddings so next run regenerates with new sermons
        cache_path = os.path.join(BASE_DIR, "utilities", "channel_embeddings.npy")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logging.info("Cleared old embeddings cache for fresh indexing.")
    else:
        logging.info("No new sermons found. Everything is up to date!")


if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sync_latest(pages)
