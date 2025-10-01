#!/usr/bin/env python3

import argparse
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict

import requests
from langchain_openai import ChatOpenAI

from .jina import JinaAI
from .exceptions import RetrievalError
from .common import set_up_logging

ENV_PATH = "~/.config/zfabric/.env"

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark


class WebExtractor:

    PREFERRED_LIB = {
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
        lib: str = "preferred",
        default_lib: str = "jina.ai",
        profile: Optional[Union[str, Dict[str, Any]]] = None,
        llm_client: ChatOpenAI = None,
        retriever: str = "requests",
        proxy: str = None,
        loglevel: int = 20  # logging.INFO
    ):
        """Extract text content from web pages
        - lib: library to use: preferred/newspaper3k/readability-lxml/jina.ai
        - default_lib: fallback to this if there is no preference for the requested domain
        - profile: profile loaded when using jina.ai lib
        - llm_client: client to use for local jina.ai deployment
        - retriever: retriever to use to download raw html
        """

        self.lib = lib
        self.default_lib = default_lib
        self.llm_client = llm_client
        self._profile_to_load = profile
        self._profile = None
        self.retriever = retriever
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

    def _select_lib(self, lib):
        if lib:
            return lib
        elif self.lib == "preferred":
            from urllib.parse import urlparse
            hostname = urlparse(url).hostname
            if hostname in self.PREFERRED_LIB:
                host_pref = self.PREFERRED_LIB["hostname"]
                return host_pref.get("extractor", self.default_lib)
                # selected_retriever = host_pref.get("retriever", retriever)
            else:
                return self.default_lib
        else:
            return self.lib

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

        for retriever in retrievers:
            try:
                return self.retrieve(url, retriever)
            except:
                continue

    def extract(
        self,
        url: str,
        lib: Optional[str] = None,
        json: bool = False,
        retriever: Union[str, List[str]] = ["requests", "jina_api"]
    ) -> str:
        """Extract content from url using lib"""

        retrievers = retriever if isinstance(retriever, list) else [retriever]
        html_content = self.retrieve_with_fallback(url, retrievers)
        selected_lib = self._select_lib(lib)

        if selected_lib == "newspaper4k":
            from newspaper import Article

            article = Article(url)
            # article.download()
            article.set_html(html_content)
            article.parse()
            return html_content, article.text

        elif selected_lib == "trafilatura":
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
            return html_content, text

        elif selected_lib == "jina.ai":
            text = self.jina.get_markdown_content(
                url,
                llm_client=self.llm_client,
                profile=self.profile,
                json=json,
                retriever=retriever,
                proxies=self.proxies
            )

            if self.profile["type"] in ("ollama", "openai"):
                text = self.jina.strip_markdown(content)

            if json:
                return raw_html, js.loads(content)
            else:
                return html_content, content

        else:
            raise ValueError("Unknown extraction library: " + self.lib)

    def load_documents(self, path: str):
        with open(path, "r") as f:
            self.lines = [js.loads(line.strip())
                          for line in f if not line.strip().startswith("#")]

        return self.lines

    def get_documents(self, path: str = None, ndata: Optional[List] = None):
        if path is not None:
            self.load_documents(path)

        rdata = []
        for parsed_line in ndata if ndata is not None else self.lines:
            _, url_data = self.extract(parsed_line["url"], json=True)
            rdata.append({
                "url": parsed_line["url"],
                "date": parsed_line["date"] if "date" in parsed_line else url_data["date"],
                "title": parsed_line["title"] if "title" in parsed_line else url_data["title"],
                "summary": parsed_line["summary"]
                if "summary" in parsed_line
                else url_data["summary"],
                "tags": parsed_line["tags"]
                if "tags" in parsed_line
                else [] + [urlparse(parsed_line["url"]).hostname] + url_data["keywords"],
                "source": "url_queue",
            })

        return rdata
