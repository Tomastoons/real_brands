"""Backward-compatible facade for brand extraction.

The implementation is split across focused modules to keep responsibilities clear:
- app.extract_shared: shared config, model loading, and data structures
- app.extract_candidates: candidate extraction, counting, and canonical merging
- app.extract_domains: domain inference logic
- app.extract_pipeline: orchestration entrypoint
"""

from app.extract_pipeline import extract_brand_analysis
from app.extract_shared import BrandResult, _get_nlp

__all__ = ["BrandResult", "extract_brand_analysis", "_get_nlp"]
