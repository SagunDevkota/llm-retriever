"""Slug and node-id helpers."""

import re
from urllib.parse import urlparse


def slugify(text: str) -> str:
    """Lower-case, replace non-alphanumeric runs with hyphens, strip edges."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def make_page_slug(url: str) -> str:
    """
    Turn a page URL into a slug.
    e.g. https://docs.example.com/build/memory/ -> build-memory
    """
    path = urlparse(url).path
    # strip leading/trailing slashes, slugify the whole path
    return slugify(path.strip("/")) or "root"


def make_node_id(page_slug: str, level: int, heading_slug: str, index: int) -> str:
    """
    {page_slug}_{level}_{heading_slug}_{index}
    index is the count of nodes that already share
    (page_slug, level, heading_slug) on this page.
    """
    return f"{page_slug}_{level}_{heading_slug}_{index}"
