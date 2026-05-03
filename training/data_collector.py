"""
training/data_collector.py — Collect figure+ground-truth data pairs

Fetches open-access papers that have deposited raw data alongside their
published figures, creating training pairs for fine-tuning Gemma4.

Target repositories:
  • eLife API   — requires raw data alongside all figures
  • PLOS ONE    — required raw data since 2014
  • Figshare    — papers + deposited CSVs
  • Zenodo      — multi-domain open data
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Output directory for collected training data
_DEFAULT_DATA_DIR = Path("training_data") / "raw"


class DataCollector:
    """Collect figure-data pairs from open-access repositories.

    Parameters
    ----------
    output_dir : Path
        Where to save downloaded figure images and ground-truth CSVs.
    max_papers : int
        Maximum number of papers to attempt per source.
    """

    def __init__(
        self,
        output_dir: Path = _DEFAULT_DATA_DIR,
        max_papers: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_papers = max_papers
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "FigureVault/0.1 (research)"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_from_elife(self) -> list[dict]:
        """Collect figure+data pairs from the eLife API.

        eLife mandates raw data deposition alongside all research articles,
        making it an ideal source for matched figure-CSV training pairs.

        Returns
        -------
        list[dict]
            Each entry contains 'doi', 'figure_path', 'data_path'.
        """
        logger.info("Collecting from eLife API (max_papers=%d)", self.max_papers)
        pairs: list[dict] = []

        # eLife API v2 — list articles
        url = "https://api.elifesciences.org/articles"
        params = {"per-page": 10, "page": 1}

        while len(pairs) < self.max_papers:
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.warning("eLife API error: %s", exc)
                break

            articles = data.get("items", [])
            if not articles:
                break

            for article in articles:
                doi = article.get("doi")
                if not doi:
                    continue
                pairs.extend(self._process_elife_article(doi))
                if len(pairs) >= self.max_papers:
                    break

            params["page"] += 1
            time.sleep(0.5)  # be polite

        logger.info("eLife: collected %d pairs", len(pairs))
        return pairs

    def collect_from_figshare(self, search_term: str = "raw data figure") -> list[dict]:
        """Search Figshare for datasets that include both figures and raw CSV data.

        Parameters
        ----------
        search_term : str
            Figshare search query.

        Returns
        -------
        list[dict]
        """
        logger.info("Collecting from Figshare (query='%s')", search_term)
        pairs: list[dict] = []

        url = "https://api.figshare.com/v2/articles/search"
        body = {
            "search_for": search_term,
            "item_type": 3,   # 3 = dataset
            "page_size": 25,
            "page": 1,
        }

        while len(pairs) < self.max_papers:
            try:
                resp = self._session.post(url, json=body, timeout=30)
                resp.raise_for_status()
                articles = resp.json()
            except requests.RequestException as exc:
                logger.warning("Figshare API error: %s", exc)
                break

            if not articles:
                break

            for article in articles:
                article_id = article.get("id")
                if article_id:
                    pair = self._process_figshare_article(article_id)
                    if pair:
                        pairs.append(pair)
                if len(pairs) >= self.max_papers:
                    break

            body["page"] += 1
            time.sleep(0.3)

        logger.info("Figshare: collected %d pairs", len(pairs))
        return pairs

    def generate_synthetic_pairs(self, n: int = 500) -> list[dict]:
        """Generate synthetic figure+data pairs for common plot types.

        Creates matplotlib figures from known CSV data, giving perfectly
        labelled training pairs at scale.

        Parameters
        ----------
        n : int
            Number of synthetic pairs to generate.

        Returns
        -------
        list[dict]
        """
        logger.info("Generating %d synthetic training pairs", n)
        # TODO: Import and call SyntheticGenerator
        # from training.synthetic_generator import SyntheticGenerator
        # gen = SyntheticGenerator(output_dir=self.output_dir / "synthetic")
        # return gen.generate(n)
        logger.warning("Synthetic generation stub — implement in synthetic_generator.py")
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_elife_article(self, doi: str) -> list[dict]:
        """Download figures and supplementary data for an eLife article."""
        # Stub: implement full eLife figure+data download logic
        logger.debug("Processing eLife DOI: %s", doi)
        return []

    def _process_figshare_article(self, article_id: int) -> Optional[dict]:
        """Fetch files for a Figshare article and match figures to CSVs."""
        try:
            resp = self._session.get(
                f"https://api.figshare.com/v2/articles/{article_id}/files",
                timeout=20,
            )
            resp.raise_for_status()
            files = resp.json()
        except requests.RequestException:
            return None

        figures = [f for f in files if f.get("name", "").lower().endswith((".png", ".jpg", ".tif"))]
        csvs = [f for f in files if f.get("name", "").lower().endswith(".csv")]

        if figures and csvs:
            return {
                "source": "figshare",
                "article_id": article_id,
                "figure_url": figures[0].get("download_url"),
                "data_url": csvs[0].get("download_url"),
            }
        return None
