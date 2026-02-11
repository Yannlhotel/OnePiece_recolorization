import os
import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def download_image(img_url, filename):
    try:
        resp = requests.get(img_url, headers=HEADERS)
        resp.raise_for_status()
        with open(filename, "wb") as f:
            f.write(resp.content)
        print(f"Téléchargé {filename}")
    except Exception as e:
        print(f"Erreur téléchargement {img_url} : {e}")

def scrape_chapter(base_url, output_dir):
    print(f"Récupération de la page {base_url}")
    try:
        resp = requests.get(base_url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")
        imgs = soup.find_all("img")
        count = 1

        # LIMITER FOR TESTING (max images per chapter)
        MAX_IMAGES = 6

        for img in imgs:
            if count > MAX_IMAGES:
                break

            img_url = img.get("data-src") or img.get("data-original") or img.get("src")
            if not img_url:
                continue
            img_url = img_url.strip()
            if img_url.startswith("data:image"):
                continue
            if img_url.startswith("/"):
                img_url = requests.compat.urljoin(base_url, img_url)

            filename = f"{output_dir}/page_{count:03d}.webp"
            download_image(img_url, filename)
            count += 1
    except Exception as e:
        print(f"Error scraping chapter: {e}")

def main(start_chapter: int = 1, end_chapter: int = 2, base_dir: str = "data/temp_scraping"):
    # Temporary folder (will be removed by the top-level main)
    for chapter in range(start_chapter, end_chapter + 1):
        base_url = f"https://www.scan-vf.net/one_piece/chapitre-{chapter}"
        output_dir = f"{base_dir}/chapitre_{chapter}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Chapter {chapter} ===")
        scrape_chapter(base_url, output_dir)
        time.sleep(1.5)


if __name__ == "__main__":
    main()