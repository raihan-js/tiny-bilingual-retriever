"""Tiny bilingual retriever: distill bge-m3 into a 30M JA-EN encoder."""

from tinyrerank.eval import (RetrievalResult, build_relevance_judgments,
                             compute_metrics, cosine_search, timed_search)
from tinyrerank.models import Encoder

__version__ = "0.1.0"
__all__ = ["Encoder", "RetrievalResult", "build_relevance_judgments",
           "compute_metrics", "cosine_search", "timed_search"]
