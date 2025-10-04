#!/usr/bin/env python3

import argparse
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict, Generator
from urllib.parse import urlparse

import requests
import concurrent.futures
import threading
import time
from collections import defaultdict
from langchain_openai import ChatOpenAI

from .jina import JinaAI
from .exceptions import RetrievalError, FailedRetrievalError
from .common import set_up_logging

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark


class WebExtractor:

    PREFERENCES = {
        "thehackernews.com": {
            "extractor": "newspaper3k"
        },
        "www.darkreading.com": {
            "retriever": "jina_api",
            "extractor": "jina.ai"
        },
        "www.bleepingcomputer.com": {
            "retriever": "requests"
            "rate_limit": True,
        }
    }

    DEFAULT_JINA_PROFILE = {"type": "jina.ai"}

    def __init__(
        self,
        extractor: str = "preferred",
        fallback_extractor: str = "jina.ai",
        jina_profile: Optional[Union[str, Dict[str, Any]]] = None,
        llm_clients: List[ChatOpenAI] = [],
        retrievers: List[str] = ["requests", "preferred"],
        fallback_retriever: str = "jina_api",
        proxy: str = None,
        loglevel: int = 20,  # logging.INFO
        max_requests_per_domain: int = 2,
        min_interval_per_domain: float = 1.0,
    ):
        """Extract text content from web pages
        - extractor: library to use for text extraction from raw HTML
        - fallback_extractor: fallback to this if there is no preference for the requested domain
        - profile: profile loaded when using jina.ai lib
        - llm_clients: clients to use for local jina.ai deployment
        - retriever: retriever to use to download raw html
        - default_retriever: fallback to this if there is not preference for the requested domain
        """

        self.extractor = extractor
        self.fallback_extractor = fallback_extractor
        self.llm_clients = llm_clients
        self._saved_jina_profile = jina_profile or self.DEFAULT_JINA_PFORILE
        self._jina_profile = None
        self.retrievers = retrievers
        self.fallback_retriever = fallback_retriever
        self.proxy = proxy
        self.logger = set_up_logging(loglevel)

        # ---------- rate‑limiting state ----------
        self.max_requests_per_domain = max_requests_per_domain
        self.min_interval_per_domain = min_interval_per_domain
        # semaphore per hostname to limit concurrent requests
        self._domain_semaphores = defaultdict(
            lambda: threading.Semaphore(self.max_requests_per_domain)
        )
        # timestamp of last request per hostname
        self._last_request_ts = defaultdict(lambda: 0.0)
        self._ts_lock = threading.Lock()
        # -----------------------------------------

        self.jina = JinaAI(
            self.jina_profile,
            loglevel=loglevel,
            llm_clients=self.llm_clients,
            proxies=self.proxies
        )

    @property
    def proxies(self):
        if self.proxy:
            return {"http": self.proxy, "https": self.proxy}
        return None

    @property
    def jina_profile(self):
        if self._jina_profile:
            return self._jina_profile

        if isinstance(self._saved_jina_profile, Dict):
            self._jina_profile = self._saved_jina_profile
        elif isinstance(self._saved_jina_profile, str):
            self._jina_profile = JinaAI.get_profile_from_env(
                self._saved_jina_profile)
        else:
            raise ValueError(
                "Type handling not implemented for jina "
                f"profile: {self._saved_jina_profile}"
            )

        return self._jina_profile

    def _get_preferred(self, url: str, mod: str):
        if mod == "extractor":
            default = self.fallback_extractor
        elif mod == "retriever":
            default = self.fallback_retriever
        else:
            raise ValueError("Mod can only be 'extractor/retriever'")

        hostname = urlparse(url).hostname
        if hostname in self.PREFERENCES:
            host_pref = self.PREFERENCES[hostname]
            return host_pref.get(mod, default)

        return default

    def _select_extractor(self, extractor: str, url: str):
        if extractor:
            if extractor == "preferred":
                return self._get_preferred(url, "extractor")
            return extractor
        elif self.extractor == "preferred":
            return self._get_preferred(url, "extractor")
        else:
            return self.extractor

    def _select_retrievers(self, retrievers: List[str], url: str):
        if len(retrievers) == 0:
            return self._select_retrievers(self.retrievers, url)

        r_retrievers = []
        for retriever in retrievers:
            if retriever == "preferred":
                r_retrievers.append(self._get_preferred(url, "retriever"))
            else:
                r_retrievers.append(retriever)

        return r_retrievers

    # ------------------------------------------------------------------
    # Rate‑limiting helpers
    # ------------------------------------------------------------------
    def _wait_if_necessary(self, hostname: str) -> None:
        """
        Sleep until at least ``self.min_interval_per_domain`` seconds have passed
        since the previous request to *hostname*.
        """
        with self._ts_lock:
            last_ts = self._last_request_ts[hostname]
            now = time.time()
            elapsed = now - last_ts
            wait = self.min_interval_per_domain - elapsed

            # Record the timestamp of this request (or the future time after sleep)
            self._last_request_ts[hostname] = now

        if wait > 0:
            self.logger.debug(
                f"Rate‑limit: sleeping {wait:.2f}s before next request to {hostname}"
            )
            time.sleep(wait)

    def _rate_limited_get(self, url: str, **kwargs) -> requests.Response:
        """
        Perform a GET request respecting both concurrency and time‑based limits.
        """
        hostname = urlparse(url).hostname
        # concurrency gate
        sem = self._domain_semaphores[hostname]
        with sem:
            # time‑based gate
            self._wait_if_necessary(hostname)
            # actual request
            return requests.get(url, **kwargs)

    def retrieve(
            self,
            url: str,
            retriever: str = "requests",
    ) -> str:

        if retriever == "requests":
            try:
                self.logger.info(
                    f"Using requests to fetch **raw HTML** from: {url}")
                # CloudFlare tends to block this
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                    " AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.10 Safari/605.1.1"
                }
                response = self._rate_limited_get(
                    url,
                    proxies=self.proxies,
                    verify=False if self.proxy else True,
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                self.logger.info(
                    f"Retrieved raw HTML content: {len(response.text)}")
                return response.text
            except requests.exceptions.RequestException as e:
                error = f"Error during HTML retrieveal: {str(e)}"
                self.logger.error(error + f", {url}")
                raise RetrievalError(error)
        elif retriever == "jina_api":
            return self.jina.get_html_content(url, self.proxies)
        elif retriever == "playwright":
            try:
                self.logger.info(
                    f"Using playwright to fetch **raw HTML** from: {url}")
                from playwright.sync_api import sync_playwright
                with sync_playwright as p:
                    browser = p.chromium.launch(
                        proxy={"server": self.proxy} if self.proxy else None)
                    page = browser.new_page()
                    _ = page.goto(url)  # before DOM gets modified by JS
                    html = page.content()
                    self.logger.info(
                        f"Retrieved raw HTML content: {len(html)}")
                    browser.close()
                    return html
            except ModuleNotFoundError:
                self.logger.warning(
                    "Playwright not installed, run:\n"
                    "pip3 install playwright\n"
                    "playwright install chromium --only-shell"
                )
                raise
            except Exception as e:
                error = f"Error during HTML retrieveal: {str(e)}"
                self.logger.error(error + f", {url}")
                raise RetrievalError(error)
        else:
            raise ValueError(f"Unknown retriever {retriever}")

    def retrieve_with_fallback(
            self,
            url: str,
            retrievers: List[str] = ["requests", "jina_api"],
    ) -> str:

        if len(retrievers) == 0:
            retrievers = self.retrievers

        results = []
        for retriever in retrievers:
            try:
                results.append({
                    "html_content": self.retrieve(url, retriever),
                    "status": "success",
                    "retriever": retriever
                })
                return results
            except Exception as e:
                results.append({
                    "html_content": None,
                    "status": "error",
                    "error": str(e)
                })

        raise FailedRetrievalError(results)

    def extract(
        self,
        url: str,
        extractor: Optional[str] = None,
        json: bool = False,
        retrievers: List[str] = [],
        html_content: Optional[str] = None
    ) -> str:
        """
        Extract content from url using extractor, optionally retrieve
        content from URL first
        - url: URL to retrieve content from
        - extractor: single extractor to use
        - json: extractor output format
        - retrievers: list of retrievers to try when retrieving content
        - html_content: html content to extract text from, skips retrieval
        step if set
        """

        selected_extractor = self._select_extractor(extractor, url)
        if not html_content and \
                (selected_extractor != "jina.ai" or self.jina_profile["type"] != "jina.ai"):
            # jina.ai API does not need content, only URL
            retrievers = self._select_retrievers(retrievers, url)
            html_content = self.retrieve_with_fallback(url, retrievers)

        if selected_extractor == "newspaper4k":
            from newspaper import Article

            article = Article(url)
            # article.download()
            article.set_html(html_content)
            article.parse()
            return html_content, article.text

        elif selected_extractor == "trafilatura":
            import trafilatura
            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_links=True,
                # include_images=True,
                # include_formatting=True,
                with_metadata=True,
                output_format="json" if json else "markdown",
                url=url
            )
            return html_content, js.loads(text) if json else text

        elif selected_extractor == "jina.ai":
            content = self.jina.get_markdown_content(
                html_content,
                url=url,
                json=json,
                proxies=self.proxies
            )

            if self.jina_profile["type"] in ("openai",):
                content = self.jina.strip_markdown(content)

            if json:
                return html_content, js.loads(content)
            else:
                return html_content, content

        else:
            raise ValueError(
                "Unknown extraction library: " + selected_extractor)

    def bulk_retrieve(
        self,
        urls: List[str],
        retrievers: List[str] = [],
        max_workers_per_host: int = 2,
    ) -> Generator[int, None, None]:
        """Threaded retrieval of HTML content from urls using a ThreadPoolExecutor.

        Limits the number of concurrent workers per hostname to ``max_workers_per_host``
        to avoid overloading a single target server.

        Yields a tuple ``(index, result)`` where ``index`` is the position of the
        URL in the original ``urls`` list and ``result`` is the value returned by
        the retrieval. Errors during retrieval are logged and ``None`` is
        yielded for that index if all retrievers fail.
        """
        # Semaphore per hostname to limit concurrency per host
        host_semaphores = defaultdict(
            lambda: threading.Semaphore(max_workers_per_host)
        )

        def _retrieve_with_limit(url: str, retrievers: List[str]):
            hostname = urlparse(url).hostname
            sem = host_semaphores[hostname]
            h_retrievers = self._select_retrievers(retrievers, url)
            with sem:
                return self.retrieve_with_fallback(url, h_retrievers)

        # Map each URL to its index so we can return results in order of the
        # original list, even though retrieval runs concurrently.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(
                    _retrieve_with_limit,
                    url,
                    retrievers
                ): idx
                for idx, url in enumerate(urls)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    html_content = future.result()
                    result = {
                        "id": idx,
                        "url": urls[idx],
                        "status": "success",
                        "html_content": html_content
                    }
                except Exception as e:
                    self.logger.error(
                        f"Error retrieving URL at index {idx}: {e}"
                    )
                    result = {
                        "id": idx,
                        "url": urls[idx],
                        "status": "error",
                        "html_content": None,
                        "error": str(e)
                    }
                yield result
