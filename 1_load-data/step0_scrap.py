import os
import time
import requests
import random
from bs4 import BeautifulSoup
import config  # Import shared variables

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def download_image(img_url, filename):
    try:
        resp = requests.get(img_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        # Filter: Ignore images < 10KB (often icons or empty pixels)
        if len(resp.content) < 10 * 1024:
            return False

        with open(filename, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  ❌ Download error: {e}")
        return False

def run():
    print(f"\n=== STEP 0: SCRAPING (Chapters {config.START_CHAPTER} to {config.END_CHAPTER}) ===")

    base_url_template = "https://www.scan-vf.net/one_piece/chapitre-{}"

    for chapter in range(config.START_CHAPTER, config.END_CHAPTER + 1):
        chapter_dir = os.path.join(config.RAW_DIR, f"chapitre_{chapter}")
        os.makedirs(chapter_dir, exist_ok=True)

        # Skip if already downloaded
        if len(os.listdir(chapter_dir)) > 5:
            print(f"Chapter {chapter} already present -> Skip")
            continue

        url = base_url_template.format(chapter)
        print(f"Processing: {url}")

        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 404:
                print(f"  ⚠️  Chapter {chapter} not found.")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            imgs = soup.find_all("img")

            count = 1
            for img in imgs:
                if config.MAX_PAGES and count > config.MAX_PAGES:
                    break
                img_url = img.get("data-src") or img.get("data-original") or img.get("src")
                if not img_url:
                    continue
                img_url = img_url.strip()

                # URL cleanup
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = requests.compat.urljoin(url, img_url)

                # Skip ads/logos
                if "logo" in img_url or "pub" in img_url:
                    continue

                filename = os.path.join(chapter_dir, f"page_{count:03d}.webp")
                if download_image(img_url, filename):
                    print(f"  -> {os.path.basename(filename)}")
                    count += 1

            time.sleep(random.uniform(1, 2))  # Anti-ban pause

        except Exception as e:
            print(f"Critical error Chapter {chapter}: {e}")

if __name__ == "__main__":
    run()