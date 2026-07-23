"""Parse a documentation HTML page into a hierarchical list of :class:`Node`s."""

from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString

from core.errors import ParseError
from core.models import Node
from parsing.slugs import make_node_id, make_page_slug, slugify

# Sections whose content exceeds MAX_CONTENT_WORDS are split into multiple
# sibling nodes at <p>-block boundaries, each packed as close to the limit as
# possible; every chunk after the first starts with the OVERLAP_WORDS trailing
# words of the previous chunk. A single block longer than the limit is first
# windowed into MAX_CONTENT_WORDS-word pieces (the packer adds the overlap).
MAX_CONTENT_WORDS = 800
OVERLAP_WORDS = 100


def _split_long_block(text: str) -> list[str]:
    """Split one paragraph block into windows of at most MAX_CONTENT_WORDS words."""
    words = text.split()
    if len(words) <= MAX_CONTENT_WORDS:
        return [text]
    return [
        " ".join(words[start : start + MAX_CONTENT_WORDS])
        for start in range(0, len(words), MAX_CONTENT_WORDS)
    ]


def chunk_content_parts(content_parts: list[str]) -> list[str]:
    """Pack paragraph blocks into chunks as close to MAX_CONTENT_WORDS words as possible.

    Blocks are kept whole and grouped in order; a chunk absorbs the next block
    whenever that lands closer to the target size than stopping short, so a
    chunk may somewhat exceed the limit. Blocks longer than the limit are first
    split by _split_long_block. Each chunk after the first starts with the
    OVERLAP_WORDS trailing words of the previous chunk.
    """
    blocks: list[str] = []
    for part in content_parts:
        blocks.extend(_split_long_block(part))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for block in blocks:
        n_words = len(block.split())
        over = current_words + n_words - MAX_CONTENT_WORDS
        if current and over > 0 and over >= MAX_CONTENT_WORDS - current_words:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            tail = " ".join(chunk_text.split()[-OVERLAP_WORDS:])
            current = [tail]
            current_words = len(tail.split())
        current.append(block)
        current_words += n_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def serialize_content(element, url: str, base_url: str):
    """
    Convert an HTML element to plain text, preserving link targets inline
    and collecting outgoing links that stay within base_url.
    Returns (text: str, outgoing_links: list[str])
    """
    element = BeautifulSoup(str(element), "html.parser")

    outgoing_links = []

    # Links
    for a in element.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"].strip()
        absolute = urljoin(url, href)
        if str(base_url) in str(absolute):
            outgoing_links.append(absolute)
        a.replace_with(NavigableString(f"{text} ({absolute})"))

    # Inline code
    for code in element.find_all("code"):
        txt = code.get_text(strip=True)
        code.replace_with(NavigableString(f"`{txt}`"))

    # Pre / code blocks
    for pre in element.find_all("pre"):
        txt = pre.get_text("\n", strip=True)
        pre.replace_with(NavigableString(f"\n```\n{txt}\n```\n"))

    raw = element.get_text("\n", strip=True)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines), list(dict.fromkeys(outgoing_links))


def parse_docs_page(
    soup: BeautifulSoup,
    url: str,
    base_url: str,
    title: str,
) -> list[Node]:
    try:
        return _parse_docs_page(soup, url, base_url, title)
    except Exception as e:  # surface parsing failures with the offending URL
        raise ParseError(f"Failed to parse page {url}: {e}") from e


def _parse_docs_page(
    soup: BeautifulSoup,
    url: str,
    base_url: str,
    title: str,
) -> list[Node]:
    page_slug = make_page_slug(url)

    slug_counter: dict[tuple, int] = {}
    # Headings drive the tree by their level; an `id` is only needed to build a
    # deep-link anchor. Headings without one still become nodes (Docusaurus/
    # playwright.dev always emit ids, but other sites may not) — they just fall
    # back to the plain page URL as their canonical link.
    heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    # ------------------------------------------------------------------
    # Page (root) node
    # ------------------------------------------------------------------
    # Content that appears before the first parsed heading (intro/lead prose)
    # belongs to no section, so it is stored on the page (root) node.
    lead_parts: list[str] = []
    lead_links: list[str] = []
    if heading_tags:
        for sibling in heading_tags[0].find_previous_siblings():
            text, links = serialize_content(sibling, url, base_url)
            if text:
                lead_parts.append(text)
            if links:
                lead_links.extend(links)
        lead_parts.reverse()  # find_previous_siblings yields reverse document order
    lead_links = list(dict.fromkeys(lead_links))

    lead_content = "\n\n".join(lead_parts)
    page_content = "\n\n".join(p for p in [title or "", lead_content] if p)

    page_node_id = make_node_id(page_slug, 0, "root", 0)
    page_node = Node(
        id=page_node_id,
        parent_id=None,
        level=0,
        title=title or "",
        title_path=title or "",
        children=[],
        url=url,
        canonical_url=url,
        content=page_content,
        metadata={"outgoing_links": lead_links} if lead_links else {},
    )

    nodes_by_id: dict[str, Node] = {page_node_id: page_node}
    ordered_nodes: list[Node] = [page_node]

    # ancestor_stack entries: (level, node_id, title)
    ancestor_stack: list[tuple[int, str, str]] = [(0, page_node_id, title or "")]

    for tag in heading_tags:
        level = int(tag.name[1])
        raw_id = tag.get("id")
        anchor = str(raw_id) if raw_id else ""
        heading_text = tag.get_text(strip=True)
        heading_slug = slugify(heading_text) or slugify(anchor)
        canonical_url = f"{url}#{anchor}" if anchor else url

        key = (page_slug, level, heading_slug)

        # Collect content and outgoing links
        content_parts: list[str] = []
        all_links: list[str] = []
        for sibling in tag.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                break
            text, links = serialize_content(sibling, url, base_url)
            if text:
                content_parts.append(text)
            if links:
                all_links.extend(links)

        all_links = list(dict.fromkeys(all_links))
        content = "\n\n".join(content_parts)

        # Resolve parent
        while len(ancestor_stack) > 1 and ancestor_stack[-1][0] >= level:
            ancestor_stack.pop()

        parent_level, parent_id, parent_title_path = ancestor_stack[-1]
        title_path = f"{parent_title_path} > {heading_text}"

        # Oversized sections become multiple sibling nodes split at <p>-block
        # boundaries; each chunk consumes the next id index for this heading.
        if len(content.split()) > MAX_CONTENT_WORDS:
            chunks = chunk_content_parts(content_parts)
        else:
            chunks = [content]

        parent_node = nodes_by_id[parent_id]
        section_node_id = ""
        for chunk_level, chunk_text in enumerate(chunks):
            index = slug_counter.get(key, 0)
            slug_counter[key] = index + 1
            node_id = make_node_id(page_slug, level, heading_slug, index)

            metadata: dict = {"outgoing_links": all_links}
            if anchor:
                metadata["anchor"] = anchor
            node_title = heading_text
            node_title_path = title_path
            if len(chunks) > 1:
                metadata["chunk_level"] = chunk_level
                metadata["chunk_count"] = len(chunks)
                part = f"part {chunk_level + 1} of {len(chunks)}"
                node_title = f"{heading_text} ({part})"
                node_title_path = f"{title_path} ({part})"

            node = Node(
                id=node_id,
                parent_id=parent_id,
                level=level,
                title=node_title,
                title_path=node_title_path,
                children=[],
                url=url,
                canonical_url=canonical_url,
                content=chunk_text,
                metadata=metadata,
            )

            nodes_by_id[node_id] = node
            ordered_nodes.append(node)
            parent_node.children.append(node_id)
            if chunk_level == 0:
                section_node_id = node_id

        # Later subheadings attach to the first chunk of a split section.
        ancestor_stack.append((level, section_node_id, title_path))

    return ordered_nodes
