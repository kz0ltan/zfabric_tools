#!/usr/bin/env python3

import argparse
from collections.abc import Callable
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict, Generator, Tuple
from urllib.parse import urlparse

import requests
import concurrent.futures
import threading
import time
from collections import defaultdict
from langchain_openai import ChatOpenAI

from .jina import JinaAI
from .exceptions import RetrievalError, FailedRetrievalError, HTTPStatusError, ExtractionError
from .common import set_up_logging

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark


class WebExtractor:

    DOMAIN_SETTINGS = {
        "thehackernews.com": {
            "retriever": "requests",
            "extractor": "newspaper4k",
        },
        "www.darkreading.com": {
            "retriever": "playwright",
            "extractor": "newspaper4k",
        },
        "www.bleepingcomputer.com": {
            "retriever": "requests",
            "extractor": "newspaper4k",
            "rate_limit": 2,
        }
    }

    DEFAULT_JINA_PROFILE = {"type": "jina.ai"}

    def __init__(
        self,
        extractor: str = "preferred",
        fallback_extractor: str = "jina.ai",
        jina_profile: Optional[Union[str, Dict[str, Any]]] = None,
        llm_clients: List[ChatOpenAI] = [],
        retrievers: List[str] = ["preferred"],
        fallback_retriever: str = "jina_api",
        proxy: str = None,
        loglevel: int = 20,  # logging.INFO
        default_max_concurrent_requests_per_domain: int = 5,
        default_domain_rate_limit: float = 0.5,
        user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

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
        self._saved_jina_profile = jina_profile or self.DEFAULT_JINA_PROFILE
        self._jina_profile = None
        self.retrievers = retrievers
        self.fallback_retriever = fallback_retriever
        self.proxy = proxy
        self.logger = set_up_logging(loglevel)
        self.user_agent = user_agent

        self.default_max_concurrent_reqs_per_domain = default_max_concurrent_requests_per_domain
        self.default_domain_rate_limit = default_domain_rate_limit
        # timestamp of last request per hostname
        self._last_request_ts = defaultdict(lambda: 0.0)
        self._ts_lock = threading.Lock()

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
        if hostname in self.DOMAIN_SETTINGS:
            host_pref = self.DOMAIN_SETTINGS[hostname]
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
        Sleep until at least domain_rate_limit seconds have passed
        since the previous request to *hostname*.
        """
        domain_rate_limit = self.DOMAIN_SETTINGS.get(
            hostname, {}).get("rate_limit") or self.default_domain_rate_limit

        with self._ts_lock:
            last_ts = self._last_request_ts[hostname]
            now = time.time()
            elapsed = now - last_ts
            wait = domain_rate_limit - elapsed

            if wait > 0:
                self.logger.debug(
                    f"Rate‑limit: sleeping {wait:.2f}s before next request to {hostname}"
                )
                time.sleep(wait)

        # Record the timestamp of this request (or the future time after sleep)
        self._last_request_ts[hostname] = now

    def _rate_limited_request(self, req_func: Callable, target_url: str, *args, **kwargs) -> Any:
        """
        Perform a req_func respecting time‑based limits
        """
        hostname = urlparse(target_url).hostname
        self._wait_if_necessary(hostname)
        return req_func(*args, **kwargs)

    def _pw_goto_with_status_check(page, url, **kwargs):
        response = page.goto(url, **kwargs)
        if not response.ok:
            raise HTTPStatusError(response)
        return response

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
                    "User-Agent": self.user_agent
                }
                response = self._rate_limited_request(
                    requests.get,
                    url,
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
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        proxy={"server": self.proxy} if self.proxy else None
                    )
                    context = browser.new_context(
                        user_agent=self.user_agent,
                        ignore_https_errors=True if self.proxy else False
                    )
                    page = context.new_page()
                    response = self._rate_limited_request(page.goto, url, url)
                    if not response.ok:
                        raise HTTPStatusError(response)
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
                    "url": url,
                    "html_content": self.retrieve(url, retriever),
                    "status": "success",
                    "retriever": retriever
                })
                return results
            except Exception as e:
                results.append({
                    "url": url,
                    "html_content": None,
                    "status": "error",
                    "error": str(e),
                    "retriever": retriever
                })

        raise FailedRetrievalError(
            "All retrieval methods failed: " + str(results))

    def extract(
        self,
        url: str,
        extractor: Optional[str] = None,
        json: bool = False,
        retrievers: List[str] = [],
        html_content: Optional[str] = None
    ) -> Tuple[str, str]:
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
            responses = self.retrieve_with_fallback(url, retrievers)
            html_content = responses[0]["html_content"]

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
            if text is None:
                raise ExtractionError("Trafilatura failed to extract text")
            return html_content, js.loads(text) if json else text

        elif selected_extractor == "jina.ai":
            response = self.jina.get_markdown_content(
                html_content,
                url=url,
                json=json,
            )

            content = response["content"]

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
        max_workers_per_host: Optional[int] = None
    ) -> Generator[int, None, None]:
        """Threaded retrieval of HTML content from urls using a ThreadPoolExecutor.

        Limits the number of concurrent workers per hostname to ``max_workers_per_host``
        to avoid overloading a single target server.
        """
        max_workers = max_workers_per_host or self.default_max_concurrent_reqs_per_domain

        # semaphore per hostname to limit concurrent requests
        self._domain_semaphores = defaultdict(
            lambda: threading.Semaphore(max_workers)
        )

        def _retrieve_with_limit(url: str, retrievers: List[str]):
            hostname = urlparse(url).hostname
            sem = self._domain_semaphores[hostname]
            h_retrievers = self._select_retrievers(retrievers, url)
            with sem:
                return self.retrieve_with_fallback(url, h_retrievers)

        # Map each URL to its index so we can return results in order of the
        # original list, even though retrieval runs concurrently.
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(urls) * max_workers) as executor:
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
                    if len(html_content):
                        result = html_content[-1]
                        result["id"] = idx
                    else:
                        raise Exception("No retrieval results were found")
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

    def bulk_extract(
        self,
        html_contents: Generator[Dict[Union[int, str], str], None, None],
        json: bool = False,
        schema: Dict[str, Any] = None,
        max_workers: int = 16  # total workers for all clients
    ) -> Generator[str, None, None]:
        for i, html_content in enumerate(html_contents):
            _, content = self.extract(
                html_content["url"],
                html_content=html_content["html_content"]
            )
            breakpoint()
