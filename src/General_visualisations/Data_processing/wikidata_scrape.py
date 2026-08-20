"""Look up StatsBomb player names in Wikidata."""

import requests
import pandas as pd
import time
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

# Rate limiting and retry settings
RATE_LIMIT = 3  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 30

# Headers required to avoid 403 Forbidden
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

def _make_api_request(params: Dict) -> Optional[Dict]:
    """Request a small result set from the Wikidata entity search API."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                WIKIDATA_API_ENDPOINT,
                params={**params, "format": "json"},
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning("Wikidata API rate limit; retrying in %ss", wait_time)
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            logger.warning("Wikidata API attempt %s failed: %s", attempt + 1, error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def get_players_by_names(player_names: List[str], country_name: Optional[str] = None) -> pd.DataFrame:
    """Find Wikidata entities for a bounded list of StatsBomb player names.

    This avoids the expensive country-wide SPARQL query. Search results are
    deliberately limited to footballers and must have a sufficiently similar
    label or alias. Unrelated search results are discarded.
    """
    from fuzzywuzzy import fuzz

    players_data = []
    seen_ids = set()

    for player_name in player_names:
        data = _make_api_request({
            "action": "wbsearchentities",
            "search": player_name,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 10,
        })
        results = (data or {}).get("search", [])

        candidates = []
        for result in results:
            description = result.get("description", "").lower()
            if "football" not in description:
                continue
            names = [result.get("label", "")] + result.get("aliases", [])
            score = max(
                (fuzz.token_set_ratio(player_name.lower(), name.lower()) for name in names if name),
                default=0,
            )
            if score >= 70:
                candidates.append((score, result))

        if not candidates:
            logger.warning("No reliable Wikidata match for '%s'", player_name)
            time.sleep(RATE_LIMIT)
            continue

        footballer = max(candidates, key=lambda item: item[0])[1]
        entity_id = footballer["id"]
        if entity_id in seen_ids:
            time.sleep(RATE_LIMIT)
            continue

        seen_ids.add(entity_id)
        players_data.append({
            "statsbomb_name": player_name,
            "wikidata_id": entity_id,
            "preferred_name": footballer.get("label", player_name),
            "position": "",
            "birth_date": "",
            "country": country_name or "",
        })
        time.sleep(RATE_LIMIT)

    logger.info("Found %s Wikidata entities for %s names", len(players_data), len(player_names))
    return pd.DataFrame(players_data, columns=[
        "statsbomb_name", "wikidata_id", "preferred_name", "position", "birth_date", "country"
    ])


