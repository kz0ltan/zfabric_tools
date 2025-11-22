import json
import logging
import os
import sys
from typing import Optional, List, Dict, Any

import requests

from requests.exceptions import (
    ConnectionError,
    HTTPError,
    Timeout,
    TooManyRedirects,
    RequestException,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class RSSSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = self.config.get("url") or os.getenv("MINIFLUX_URL")
        self.token = self.config.get("token") or os.getenv("MINIFLUX_TOKEN")
        self.category_id = self.config.get("category_id")
        self.ca_bundle_path = self.config.get("requests_ca_bundle")

        if self.ca_bundle_path:
            os.environ["REQUEST_CA_BUNDLE"] = self.ca_bundle_path

    def get_entries(
        self,
        category_id: str = None,
        since_ts: Optional[float] = None,
        status: Optional[str] = None,
        limit: Optional[int] = 1000,
        order: Optional[str] = None,
        direction: Optional[str] = None,
    ):
        """
        https://miniflux.app/docs/api.html#endpoint-get-category-entries
        """

        category_id = category_id or self.config.get("category_id")
        url = self.base_url + f"/categories/{str(category_id)}/entries"
        headers = {"X-Auth-Token": self.token}
        params = {}
        if since_ts:
            params["after"] = since_ts
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
        if order:
            params["order"] = order
        if direction:
            params["direction"] = direction

        try:
            logger.info("GET %s", url)
            response = requests.get(
                url, headers=headers, timeout=10, params=params if len(params) else None
            )
            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise ValueError("Response is not valid JSON") from exc

            logger.info("Success – received %d bytes", len(response.content))
            return data["entries"]

        except (ConnectionError, Timeout, TooManyRedirects) as exc:
            logger.error("Network problem while contacting %s: %s", url, exc)
            raise
        except HTTPError as exc:
            logger.error(
                "HTTP error %s (%s): %s",
                exc.response.status_code,
                exc.response.reason,
                response.text,
            )
            raise
        except RequestException as exc:
            logger.error("Request failed: %s", exc)
            raise

    def get_standard_entries(
        self,
        category_id: str = None,
        since_ts: Optional[float] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = "published_at",
        direction: str = "asc",
    ):
        entry_keys_to_copy = (
            "id",
            "title",
            "url",
            "published_at",
            "created_at",
            "changed_at",
            "author",
        )
        feed_keys_to_copy = ("id", "title", "feed_url", "description")

        ret = []
        for entry in self.get_entries(category_id, since_ts, status, limit, order, direction):
            t_entry = {"source": "rss", "content": {}}
            t_entry["content_metadata"] = {k: entry[k] for k in entry_keys_to_copy}
            t_entry["feed_metadata"] = {k: entry["feed"][k] for k in feed_keys_to_copy}
            ret.append(t_entry)

        return ret

    def update_entries_status(
        self,
        entry_ids: List[int],
        status: str = "read",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Send a PUT /v1/entries request with a JSON body to update entries
        - entry_ids: List of entry identifiers you want to update
        - status: Desired status for the supplied entries
        - timeout: Seconds to wait for a response before raising requests.Timeout
        """

        url = self.base_url + f"/categories/{category_id}/entries"
        headers = {"X-Auth-Token": self.token}
        # params = {}
        # if since_ts:
        #    params["after"] = since_ts
        # if status:
        #    params["status"] = status
        # if limit:
        #    params["limit"] = limit
        # if order:
        #    params["order"] = order
        # if direction:
        #    params["direction"] = direction

        # try:
        #    response = requests.put(
        #        url,
        #        headers=headers,
        #        json=params,
        #        timeout=timeout,
        #    )
        #    response.raise_for_status()
        #    return response.json()
        # except (ConnectionError, Timeout, TooManyRedirects) as exc:
        #    logger.error("Network problem while contacting %s: %s", url, exc)
        #    raise
        # except HTTPError as exc:
        #    logger.error(
        #        "HTTP error %s (%s): %s",
        #        exc.response.status_code,
        #        exc.response.reason,
        #        response.text
        #    )
        #    raise
        # except RequestException as exc:
        #    logger.error("Request failed: %s", exc)
        #    raise
