import os
import requests
from bs4 import BeautifulSoup

# Base URL de la page chap1 que tu m'as donnée
base_url = "https://www.scan-vf.net/one_piece/chapitre-6"

# Dossier de sortie
output_dir = "one_piece_chap6"
os.makedirs(output_dir, exist_ok=True)

# Headers pour ressembler à un vrai navigateur (les sites aiment ça)
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

def scrape_chapter(url):
    print(f"Récupération de la page {url}")
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    imgs = soup.find_all("img")
    count = 1

    for img in imgs:
        img_url = (
            img.get("data-src")
            or img.get("data-original")
            or img.get("src")
        )

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

# Appel principal
scrape_chapter(base_url)
