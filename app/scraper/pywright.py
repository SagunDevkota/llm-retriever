from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json


async def clean_html(soup: BeautifulSoup):
    junk_selectors = [
        "nav",
        "header",
        "footer",
        "aside",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
        ".navbar",
        ".sidebar",
        ".menu",
        ".toc",
        ".table-of-contents"
    ]

    for sel in junk_selectors:
        for tag in soup.select(sel):
            tag.decompose()

    return soup


class PlaywrightScraper:
    def __init__(self, base_url:str):
        self.playwright = None
        self.browser = None
        self.page = None
        self.base_url = base_url

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

    async def extract_page_links(self, soup: BeautifulSoup):
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()

            # ignore junk links
            if href.startswith("#") or href.startswith("javascript:"):
                continue

            href = href.split("#")[0]

            full_url = urljoin(self.base_url, href)

            # keep only same-domain links (important for docs)
            if urlparse(full_url).netloc == urlparse(self.base_url).netloc and self.base_url in full_url:
                links.add(full_url)

        return links

    async def scrape(self, url: str):
        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            soup = await clean_html(soup)
            return soup

        finally:
            await page.close()

    async def save(self, md, response, idx: int):
        with open(f"output_{idx}.md", "w") as f:
            f.write(md)

        with open(f"output_{idx}.json", "w") as f:
            json.dump(json.loads(str(response)), f, indent=4)

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()