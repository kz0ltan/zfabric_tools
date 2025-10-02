#!/usr/bin/env python3

import argparse
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict, Generator
from urllib.parse import urlparse

import requests
import concurrent.futures
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
        }
    }

    def __init__(
        self,
        extractor: str = "preferred",
        fallback_extractor: str = "jina.ai",
        profile: Optional[Union[str, Dict[str, Any]]] = None,
        llm_client: ChatOpenAI = None,
        retrievers: List[str] = ["requests", "preferred"],
        fallback_retriever: str = "jina_api",
        proxy: str = None,
        loglevel: int = 20  # logging.INFO
    ):
        """Extract text content from web pages
        - extractor: library to use for text extraction from raw HTML
        - fallback_extractor: fallback to this if there is no preference for the requested domain
        - profile: profile loaded when using jina.ai lib
        - llm_client: client to use for local jina.ai deployment
        - retriever: retriever to use to download raw html
        - default_retriever: fallback to this if there is not preference for the requested domain
        """

        self.extractor = extractor
        self.fallback_extractor = fallback_extractor
        self.llm_client = llm_client
        self._profile_to_load = profile
        self._profile = None
        self.retrievers = retrievers
        self.fallback_retriever = fallback_retriever
        self.proxy = proxy
        self.logger = set_up_logging(loglevel)

        self.jina = JinaAI(loglevel)

    @property
    def proxies(self):
        if self.proxy:
            return {"http": self.proxy, "https": self.proxy}
        return None

    @property
    def profile(self):
        if self._profile:
            return self._profile

        if isinstance(self._profile_to_load, Dict):
            self._profile = self._profile_to_load
        else:
            self._profile = self.jina.get_profile(self._profile_to_load)

        return self._profile

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
                response = requests.get(
                    url,
                    proxies=self.proxies,
                    verify=False if self.proxy else True,
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                self.logger.info(
                    f"Retrieved raw HTML content: {len(response.text)}")
                return response.text
            except requests.exceptions.RequestException as e:
                error = f"Error during HTML retrieveal: {str(e)}"
                self.logger.error(error)
                raise RetrievalError(error)
        elif retriever == "jina_api":
            return self.jina.get_html_content(url, self.proxies)
        elif retriever == "playwright":
            try:
                self.logger.info(
                    f"Using playwright to fetch **raw HTML** from: {url}")
                from playwright.sync_api import sync_playwright
                with sync_playwright as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    _ = page.goto(url)  # before DOM gets modified by JS
                    html = page.content()
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
                self.logger.error(error)
                raise
        else:
            raise ValueError(f"Unknown retriever {retriever}")

    def retrieve_with_fallback(
            self,
            url: str,
            retrievers: List[str] = ["requests", "jina_api"],
    ) -> str:

        if len(retrievers) == 0:
            retrievers = self.retrievers

        for retriever in retrievers:
            try:
                return self.retrieve(url, retriever)
            except:
                continue

        raise FailedRetrievalError("All retrieval methods failed")

    def extract(
        self,
        url: str,
        extractor: Optional[str] = None,
        json: bool = False,
        retrievers: List[str] = []
    ) -> str:
        """Extract content from url using lib"""

        retrievers = self._select_retrievers(retrievers, url)
        html_content = self.retrieve_with_fallback(url, retrievers)
        selected_extractor = self._select_extractor(extractor, url)

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
                url,
                llm_client=self.llm_client,
                profile=self.profile,
                json=json,
                proxies=self.proxies
            )

            if self.profile["type"] in ("ollama", "openai"):
                content = self.jina.strip_markdown(content)

            if json:
                return html_content, js.loads(content)
            else:
                return html_content, content

        else:
            raise ValueError(
                "Unknown extraction library: " + selected_extractor)

    def bulk_extract(
        self,
        urls: List[str],
        extractor: Optional[str] = None,
        json: bool = False,
        retrievers: List[str] = []
    ) -> Generator[int, None, None]:
        """Threaded extraction of urls using a ThreadPoolExecutor.

        Yields a tuple ``(index, result)`` where ``index`` is the position of the
        URL in the original ``urls`` list and ``result`` is the value returned by
        :meth:`extract`. Errors during extraction are logged and ``None`` is
        yielded for that index.
        """
        # Map each URL to its index so we can return results in order of the
        # original list, even though extraction runs concurrently.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(
                    self.extract,
                    url,
                    extractor,
                    json,
                    retrievers,
                ): idx
                for idx, url in enumerate(urls)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(
                        f"Error extracting URL at index {idx}: {e}"
                    )
                    result = None
                yield idx, result
