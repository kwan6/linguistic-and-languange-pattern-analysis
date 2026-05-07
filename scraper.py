"""
scraper.py
Scrape headline berita Indonesia untuk dataset clickbait.

Portal clickbait    : DetikHOT, Wowkeren, Kapanlagi
Portal non-clickbait: DetikNews, Kompas, Tempo

Usage:
  python scraper.py                           <- scrape semua portal
  python scraper.py --portal detikhot         <- scrape 1 portal saja
  python scraper.py --portal detikNews --pages 50
  python scraper.py --merge                   <- gabung semua hasil scrape

Tekan Ctrl+C kapanpun -> data yang sudah terkumpul otomatis tersimpan!
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
import argparse
import glob

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
OUTPUT_DIR = "data/raw/scraped"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_soup(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None


def sleep_random():
    time.sleep(random.uniform(1.5, 3.0))


def save_partial(data, portal_name):
    if not data:
        print(f"  [INFO] Tidak ada data untuk {portal_name}.")
        return
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["title"], inplace=True)
    path = os.path.join(OUTPUT_DIR, f"scraped_{portal_name}.csv")
    df.to_csv(path, index=False)
    print(f"  [SAVED] {len(df)} headline -> {path}")


def extract_titles(soup, selectors):
    """Coba beberapa selector sampai dapat hasil."""
    for tag, cls in selectors:
        if cls:
            results = soup.find_all(tag, class_=cls)
        else:
            results = soup.find_all(tag)
        titles = [r.get_text(strip=True) for r in results if len(r.get_text(strip=True)) > 10]
        if titles:
            return titles
    return []


# ─── CLICKBAIT PORTALS ────────────────────────────────────────────────────────

def scrape_detikhot(max_pages=100):
    """Clickbait - gosip selebriti Indonesia"""
    headlines = []
    print(f"\n[DetikHOT] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            # DetikHOT: gunakan parameter page di query string
            url = f"https://hot.detik.com/indeks?page={page}"
            soup = get_soup(url)
            if not soup:
                continue

            # Ambil semua link artikel dengan judul
            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            if not titles:
                for cls in ["media__title", "article__title", "title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["h2", "h3"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 1, "source": "detikhot"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                print("  [INFO] Tidak ada judul, berhenti.")
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] DetikHOT dihentikan.")
    finally:
        save_partial(headlines, "detikhot")
    return headlines


def scrape_wowkeren(max_pages=100):
    """Clickbait - URL baru dengan trailing slash"""
    headlines = []
    print(f"\n[Wowkeren] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            # URL baru: /berita/indonesia/page/1/ bukan /berita/indonesia/page/1.html
            url = f"https://www.wowkeren.com/berita/indonesia/page/{page}/"
            soup = get_soup(url)
            if not soup:
                continue

            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            # Fallback selectors
            if not titles:
                for cls in ["list-berita-title", "berita-title", "news-title", "title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["div", "h2", "h3"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 1, "source": "wowkeren"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] Wowkeren dihentikan.")
    finally:
        save_partial(headlines, "wowkeren")
    return headlines


def scrape_kapanlagi(max_pages=100):
    """Clickbait - URL baru /showbiz/selebriti/"""
    headlines = []
    print(f"\n[Kapanlagi] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            url = f"https://www.kapanlagi.com/showbiz/selebriti/?page={page}"
            soup = get_soup(url)
            if not soup:
                continue

            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            if not titles:
                for cls in ["newslist__title", "content-title", "title", "article-title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["h2", "h3", "div"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 1, "source": "kapanlagi"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] Kapanlagi dihentikan.")
    finally:
        save_partial(headlines, "kapanlagi")
    return headlines


# ─── NON-CLICKBAIT PORTALS ────────────────────────────────────────────────────

def scrape_detiknews(max_pages=100):
    """Non-clickbait - berita nasional"""
    headlines = []
    print(f"\n[DetikNews] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            url = f"https://news.detik.com/indeks?page={page}"
            soup = get_soup(url)
            if not soup:
                continue

            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            if not titles:
                for cls in ["media__title", "article__title", "title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["h2", "h3"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 0, "source": "detiknews"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] DetikNews dihentikan.")
    finally:
        save_partial(headlines, "detiknews")
    return headlines


def scrape_kompas(max_pages=100):
    """Non-clickbait"""
    headlines = []
    print(f"\n[Kompas] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            url = f"https://indeks.kompas.com/?site=all&page={page}"
            soup = get_soup(url)
            if not soup:
                continue

            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            if not titles:
                for cls in ["article__title", "artikel__title", "title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["h2", "h3"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 0, "source": "kompas"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] Kompas dihentikan.")
    finally:
        save_partial(headlines, "kompas")
    return headlines


def scrape_tempo(max_pages=100):
    """Non-clickbait"""
    headlines = []
    print(f"\n[Tempo] Scraping {max_pages} halaman...")
    try:
        for page in range(1, max_pages + 1):
            url = f"https://www.tempo.co/indeks/{page}"
            soup = get_soup(url)
            if not soup:
                continue

            titles = []
            for a in soup.find_all("a", href=True):
                h = a.find(["h2", "h3", "h4"])
                if h:
                    t = h.get_text(strip=True)
                    if len(t) > 10:
                        titles.append(t)

            if not titles:
                for cls in ["title", "card-title", "article-title"]:
                    found = [d.get_text(strip=True) for d in soup.find_all(["h2", "h3"], class_=cls)
                             if len(d.get_text(strip=True)) > 10]
                    if found:
                        titles = found
                        break

            for t in titles:
                headlines.append({"title": t, "label": 0, "source": "tempo"})

            print(f"  Halaman {page}/{max_pages}: {len(titles)} judul (total: {len(headlines)})")
            if len(titles) == 0 and page > 5:
                break
            sleep_random()
    except KeyboardInterrupt:
        print(f"\n  [STOP] Tempo dihentikan.")
    finally:
        save_partial(headlines, "tempo")
    return headlines


# ─── PORTAL MAP ───────────────────────────────────────────────────────────────

PORTAL_MAP = {
    "detikhot":   scrape_detikhot,
    "wowkeren":   scrape_wowkeren,
    "kapanlagi":  scrape_kapanlagi,
    "detiknews":  scrape_detiknews,
    "kompas":     scrape_kompas,
    "tempo":      scrape_tempo,
}


# ─── MERGE ────────────────────────────────────────────────────────────────────

def merge_all():
    files = glob.glob(os.path.join(OUTPUT_DIR, "scraped_*.csv"))
    files = [f for f in files if "merged" not in f]

    if not files:
        print("[MERGE] Tidak ada file scraped yang ditemukan.")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged.drop_duplicates(subset=["title"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    out_path = os.path.join(OUTPUT_DIR, "scraped_merged.csv")
    merged.to_csv(out_path, index=False)

    print(f"\n{'='*45}")
    print(f"Total sebelum dedup : {before}")
    print(f"Total setelah dedup : {len(merged)}")
    print(f"Clickbait           : {merged['label'].sum()}")
    print(f"Non-clickbait       : {(merged['label']==0).sum()}")
    print(f"Saved               : {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Scraper berita Indonesia")
    parser.add_argument("--portal", type=str, default="all",
                        choices=["all"] + list(PORTAL_MAP.keys()),
                        help="Portal yang ingin di-scrape (default: all)")
    parser.add_argument("--pages", type=int, default=100,
                        help="Jumlah halaman per portal (default: 100)")
    parser.add_argument("--merge", action="store_true",
                        help="Gabung semua hasil scrape menjadi satu file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge:
        merge_all()
        exit()

    portals = list(PORTAL_MAP.keys()) if args.portal == "all" else [args.portal]

    print(f"{'='*50}")
    print(f"Portal  : {portals}")
    print(f"Pages   : {args.pages}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Tekan Ctrl+C kapanpun -> data tetap tersimpan!")
    print(f"{'='*50}")

    try:
        for portal in portals:
            PORTAL_MAP[portal](max_pages=args.pages)
    except KeyboardInterrupt:
        print("\n[INFO] Scraping dihentikan oleh user.")

    print("\n[INFO] Menggabungkan semua hasil...")
    merge_all()
