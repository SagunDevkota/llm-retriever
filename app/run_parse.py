"""Phase 2 — parse: scraped HTML -> hierarchical node tree -> checkpoint.

Run after main.py (phase 1) has populated the site folder.
    python run_parse.py
"""

from dataclasses import asdict
from pathlib import Path
import argparse

from bs4 import BeautifulSoup

from core.checkpoint import save_checkpoint
from parsing.parser import parse_docs_page

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 2: parse a docs site to a hierarchical node tree.")
    ap.add_argument("--root", required=True,
                    help="Site folder written by main.py (required).")
    ap.add_argument("--site-url", default="https://playwright.dev",
                    help="Used to rebuild page URLs.")
    ap.add_argument("--base-url", default="https://playwright.dev/docs",
                    help="Link-scope for parsing.")
    args = ap.parse_args()

    root = Path(args.root)                 # site folder written by main.py
    site_url = args.site_url       # used to rebuild page URLs
    base_url = args.base_url  # link-scope for parsing

    nodes = []
    for file_path in sorted(root.rglob("index.html")):
        print(file_path)
        with open(file_path, "r") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")
        title = soup.find("title").get_text()

        # playwright/docs/intro/index.html -> https://playwright.dev/docs/intro
        parts = file_path.relative_to(root).parts
        if parts and parts[-1] == "index.html":
            parts = parts[:-1]
        url = site_url + "/" + "/".join(parts)

        nodes.extend(parse_docs_page(soup, url, base_url, title))

    nodes_dict = [asdict(node) for node in nodes]
    nodes_by_id = {node["id"]: node for node in nodes_dict}

    save_checkpoint(nodes_by_id)
    print(f"Parsed {len(nodes_by_id)} nodes -> checkpoint")
