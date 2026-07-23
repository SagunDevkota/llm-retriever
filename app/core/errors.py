"""Custom exception hierarchy for the hierarchical-RAG pipeline.

Each stage raises a specific subclass so callers can catch failures by stage
(scraping vs. summarization vs. embedding) instead of a bare ``Exception``.
"""


class HierarchicalRagError(Exception):
    """Base class for all errors raised by this project."""


class ScraperError(HierarchicalRagError):
    """A page could not be fetched or rendered."""


class ParseError(HierarchicalRagError):
    """An HTML document could not be parsed into the node tree."""


class SummarizationError(HierarchicalRagError):
    """The summarization LLM call failed or returned unusable output."""


class EmbeddingError(HierarchicalRagError):
    """The embedding model failed to load or encode text."""


class ConfigError(HierarchicalRagError):
    """Required configuration (e.g. an API key) is missing or invalid."""


class RetrieverError(HierarchicalRagError):
    """The LLM-guided retriever's selection call failed."""
