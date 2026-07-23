"""Phase 1 — scrape: crawl a docs site to <site>/<url-path>/index.html.

    python main.py [--base-url ...] [--start-url ...] [--workers N]
"""

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
from scraper.pywright import PlaywrightScraper


def url_to_local_path(url):
    """https://playwright.dev/docs/intro -> playwright/docs/intro/index.html

    Top-level folder is the site (playwright); the rest is the URL path, so
    different sites can't overwrite each other.
    """
    parsed = urlparse(url)
    site = parsed.netloc.lower().removeprefix("www.").split(".")[0]
    url_path = parsed.path.strip("/")
    return Path(site, url_path, "index.html")


async def worker(scraper, queue, visited):
    while True:
        url = await queue.get()

        if url in visited:
            queue.task_done()
            continue

        visited.add(url)

        try:
            soup = await scraper.scrape(url)

            path = url_to_local_path(url)
            path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(path, "w") as f:
                await f.write(str(soup))

            links = await scraper.extract_page_links(soup)

            for link in links:
                if link not in visited:
                    await queue.put(link)

        except Exception as e:
            print(f"Failed: {url} -> {e}")

        queue.task_done()

async def crawl(base_url, start_url, num_workers=5):
    scraper = PlaywrightScraper(base_url)
    await scraper.start()

    queue = asyncio.Queue()
    await queue.put(start_url)

    visited = set()

    workers = [
        asyncio.create_task(worker(scraper, queue, visited))
        for _ in range(num_workers)  # tune this (3–10 is typical)
    ]

    await queue.join()

    for w in workers:
        w.cancel()

    await scraper.close()
    return visited


def main():
    ap = argparse.ArgumentParser(description="Phase 1: scrape a docs site to disk.")
    ap.add_argument("--base-url", default="https://playwright.dev/docs",
                    help="Crawl is restricted to links under this URL.")
    ap.add_argument("--start-url", default=None,
                    help="Where to start (default: <base-url>/intro).")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    start_url = args.start_url or args.base_url + "/intro"
    visited = asyncio.run(crawl(args.base_url, start_url, args.workers))
    print(f"Scraped {len(visited)} pages.")


if __name__ == "__main__":
    main()
