#!/usr/bin/env python3

from collections.abc import Callable
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict, Generator, Tuple
import queue

import requests
import concurrent.futures
import threading
import time
from collections import defaultdict
from langchain_openai import ChatOpenAI

from .jina import JinaAI
from .exceptions import (
    RetrievalError,
    FailedRetrievalError,
    HTTPStatusError,
    ExtractionError,
)
from .common import set_up_logging

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark


class WebExtractor:
    # Custom CSS selectors should be avoided unless local Jina model is used

    DOMAIN_SETTINGS = {
        "thehackernews.com": {
            "retriever": "requests",
            "extractor": "trafilatura",
            "rate_limit": 2,
            "custom_css_selectors": [".story-title", "#content"],
        },
        "www.darkreading.com": {
            "retriever": "playwright",
            "extractor": "newspaper4k",
        },
        "www.bleepingcomputer.com": {
            "retriever": "requests",
            "extractor": "newspaper4k",
            "rate_limit": 3,
            "custom_css_selectors": ["article"],
        },
        "www.csoonline.com": {
            "retriever": "requests",
            "extractor": "jina.ai",
            "custom_css_selectors": [".article-hero__title", ".article__main"],
            # title gets left out of the markdown output with reader-lm-1.5b
        },
        "www.cisa.gov": {
            "retriever": "requests",
            "extractor": "trafilatura",
            "custom_css_selectors": ["#main"],
        },
        "www.schneier.com": {
            "retriever": "jina_api",
            "extractor": "trafilatura",
            "custom_css_selectors": [".article"],
        },
        "krebsonsecurity.com": {"custom_css_selectors": [".entry-header", "article"]},
        "www.darktrace.com": {
            "retriever": "requests",
            "extractor": "trafilatura",
            # "custom_css_selectors": [".blog_new-header", ".blog-article_main"],
        },
        "securelist.com": {
            "retriever": "requests",
            "extractor": "trafilatura",
            # "custom_css_selectors": ["article.c-article"],
        },
        "www.f5.com": {
            "retriever": "requests",
            "extractor": "trafilatura",
            # "custom_css_selectors": [.labs-main-content"]
        },
        "darkwebinformer.com": {
            "retriever": "requests",
            "extractor": "trafilatura",
            "custom_css_selectors": ["#main>article"],
        },
    }

    def __init__(
        self,
        extractor: str = "preferred",
        fallback_extractor: str = "jina.ai",
        jina_profile: Optional[Union[str, Dict[str, Any]]] = None,
        llm_clients: List[ChatOpenAI] = None,
        retrievers: List[str] = None,
        fallback_retriever: str = "jina_api",
        proxy: str = None,
        loglevel: int = 20,  # logging.INFO
        default_max_concurrent_requests_per_domain: int = 5,
        default_domain_rate_limit: float = 0.5,
        user_agent: str = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        ),
    ):
        """Extract text content from web pages
        - extractor: library to use for text extraction from raw HTML
        - fallback_extractor: fallback to this if there is no preference for the requested domain
        - jina_profile: profile type loaded when using jina.ai lib
        - llm_clients: clients to use for local jina.ai deployment
        - retriever: retriever to use to download raw html
        - fallback_retriever: fallback to this if there is not preference for the requested domain
        - proxy: http proxy for debugging
        - loglevel: amount of stuff to log
        - default_max_concurrent_requests_per_domain: maximum number of concurrent workers for a single domain
        - default_domain_rate_limit: seconds to wait between requests to the same domain
        - user_agent: user agent to send when using retrievers
        """

        self.extractor = extractor
        self.fallback_extractor = fallback_extractor
        self.llm_clients = llm_clients or []
        self._saved_jina_profile = jina_profile or "jina.ai"
        self._jina_profile = None
        self.retrievers = retrievers or ["preferred"]
        self.fallback_retriever = fallback_retriever
        self.proxy = proxy
        self.logger = set_up_logging(loglevel)
        self.user_agent = user_agent

        self.default_max_concurrent_reqs_per_domain = default_max_concurrent_requests_per_domain
        self.default_domain_rate_limit = default_domain_rate_limit
        self._last_request_ts = defaultdict(lambda: 0.0)
        self._ts_lock = threading.Lock()

        self.jina = JinaAI(
            self.jina_profile,
            loglevel=loglevel,
            llm_clients=self.llm_clients,
            proxies=self.proxies,
        )

    @property
    def proxies(self):
        """Return dict http and https proxies in dictionary format"""
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
            self._jina_profile = JinaAI.get_profile_from_env(self._saved_jina_profile)
        else:
            raise ValueError(
                f"Type handling not implemented for jina profile: {self._saved_jina_profile}"
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

    def _get_custom_css_selector(self, url: str):
        hostname = urlparse(url).hostname
        return self.DOMAIN_SETTINGS.get(hostname, {}).get("custom_css_selectors")

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

    def _wait_if_necessary(self, hostname: str) -> None:
        """
        Sleep until at least domain_rate_limit seconds have passed
        since the previous request to *hostname*.
        """
        domain_rate_limit = (
            self.DOMAIN_SETTINGS.get(hostname, {}).get("rate_limit")
            or self.default_domain_rate_limit
        )

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

    def retrieve(
        self,
        url: str,
        retriever: str = "requests",
    ) -> str:
        """Retrieve HTML code as string pointed by url using retriever"""

        if retriever == "requests":
            try:
                self.logger.info(f"Using requests to fetch **raw HTML** from: {url}")
                # CloudFlare tends to block this
                headers = {"User-Agent": self.user_agent}
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
                if response.status_code == 404:
                    raise RetrievalError(f"{response.status_code}: {response.text}")
                self.logger.info(f"Retrieved raw HTML content: {len(response.text)}")
                return response.text
            except requests.exceptions.RequestException as e:
                error = f"Error during HTML retrieveal: {str(e)}"
                self.logger.error(error + f", {url}")
                raise RetrievalError(error) from e
        elif retriever == "jina_api":
            return self.jina.get_html_content(url, self.proxies)
        elif retriever == "playwright":
            try:
                self.logger.info(f"Using playwright to fetch **raw HTML** from: {url}")
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        proxy={"server": self.proxy} if self.proxy else None
                    )
                    context = browser.new_context(
                        user_agent=self.user_agent,
                        ignore_https_errors=True if self.proxy else False,
                    )
                    page = context.new_page()
                    response = self._rate_limited_request(page.goto, url, url)
                    if not response.ok:
                        raise HTTPStatusError(response)
                    html = page.content()
                    self.logger.info(f"Retrieved raw HTML content: {len(html)}")
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
                raise RetrievalError(error) from e
        else:
            raise ValueError(f"Unknown retriever {retriever}")

    def retrieve_with_fallback(
        self,
        url: str,
        retrievers: List[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieve HTML code pointed by url using retrievers sequentially
        until the first successful retrieval
        """

        retrievers = retrievers or ["requests", "jina_api"]

        if len(retrievers) == 0:
            retrievers = self.retrievers

        results = []
        for retriever in retrievers:
            try:
                results.append({
                    "url": url,
                    "html_content": self.retrieve(url, retriever),
                    "retrieval": {"status": "success", "retriever": retriever},
                })
                return results
            except Exception as e:
                results.append({
                    "url": url,
                    "html_content": None,
                    "retrieval": {
                        "status": "error",
                        "error": str(e),
                        "retriever": retriever,
                    },
                })

        raise FailedRetrievalError(f"All retrieval methods failed for: {url}", results)

    def extract(
        self,
        url: str,
        extractor: Optional[str] = None,
        json: bool = False,
        retrievers: List[str] = [],
        html_content: Optional[str] = None,
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
        if not html_content and (
            selected_extractor != "jina.ai" or self.jina_profile["type"] != "jina.ai"
        ):
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
            return html_content, article.title + "\n\n" + article.text

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
                url=url,
            )
            if text is None:
                raise ExtractionError("Trafilatura failed to extract text")
            return html_content, js.loads(text) if json else text

        elif selected_extractor == "jina.ai":
            custom_css_selectors = self._get_custom_css_selector(url)
            response = self.jina.get_markdown_content(
                html_content,
                url=url,
                json=json,
                custom_css_selectors=custom_css_selectors,
            )

            content = response["content"]

            if self.jina_profile["type"] in ("openai",):
                content = self.jina.strip_markdown(content)

            if json:
                return html_content, js.loads(content)
            else:
                return html_content, content

        else:
            raise ValueError("Unknown extraction library: " + selected_extractor)

    def bulk_retrieve(
        self,
        urls: List[str],
        retrievers: List[str] = None,
        max_workers_per_host: Optional[int] = None,
    ) -> Generator[int, None, None]:
        """Threaded retrieval of HTML content from urls

        Limits the number of concurrent workers per hostname to ``max_workers_per_host``
        to avoid overloading a single target server.
        """
        retrievers = retrievers or []
        max_workers = max_workers_per_host or self.default_max_concurrent_reqs_per_domain

        # semaphore per hostname to limit concurrent requests
        self._domain_semaphores = defaultdict(lambda: threading.Semaphore(max_workers))

        def _retrieve_with_limit(url: str, retrievers: List[str]):
            hostname = urlparse(url).hostname
            sem = self._domain_semaphores[hostname]
            h_retrievers = self._select_retrievers(retrievers, url)
            with sem:
                return self.retrieve_with_fallback(url, h_retrievers)

        # Map each URL to its index so we can return results in order of the
        # original list, even though retrieval runs concurrently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) * max_workers) as executor:
            future_to_index = {
                executor.submit(_retrieve_with_limit, url, retrievers): idx
                for idx, url in enumerate(urls)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    html_content = future.result()
                    if len(html_content):
                        # the last try has to be successful
                        result = html_content[-1]
                    else:
                        raise Exception("No retrieval results were found")
                except Exception as e:
                    self.logger.error(f"Error retrieving URL at index {idx}: {e}")
                    result = {
                        "url": urls[idx],
                        "status": "error",
                        "html_content": None,
                        "error": str(e),
                        "retrieval": getattr(e, "retrievals", None),
                    }
                yield idx, result

    def bulk_extract(
        self,
        html_items: Generator[Dict[Union[int, str], str], None, None],
        json: bool = False,
        schema: Optional[Dict[str, Any]] = None,
        max_workers: int = 8,
        jina_batch_size: int = 8,
    ) -> Generator[Tuple[int, str, Any], None, None]:
        """
        Fully‑parallel bulk extraction.

        * Jina‑based URLs are sent in batches of *jina_batch_size* to the
          JinaAI generator (which itself uses a thread‑pool).
        * All other extractors run concurrently in a separate
          ThreadPoolExecutor.
        * Results are yielded in the original input order.
        """

        # Initialise the Jina generator
        original_items: Dict[int, Dict] = {}
        jina_gen = self.jina.get_markdown_content_batched(
            json=json,
            schema=schema,
            max_workers=max_workers,
        )
        # Prime the generator – the first ``send`` must receive an empty list
        next(jina_gen)

        # Queues and synchronisation primitives
        jina_queue: queue.Queue[Tuple[int, Dict[str, Any]]] = queue.Queue()
        result_lock = threading.Lock()

        # Stores completed results: idx → (url, content)
        completed: Dict[int, Tuple[Optional[str], Any]] = {}
        next_to_yield = 0

        def _get_metadata(res: Dict[str, Any]):
            metadata = {}
            metadata["extractor"] = "jina.ai"
            metadata["type"] = res.get("type")
            metadata["client"] = res.get("client")
            metadata["status"] = res.get("status")
            metadata["attempt"] = res.get("attempt")
            metadata["error"] = res.get("error")
            metadata["prev_errors"] = res.get("prev_errors")
            metadata["usage_metadata"] = res.get("usage_metadata")
            return metadata

        # Background thread that batches Jina items and talks to the generator
        def _jina_worker():
            batch_items: List[Dict[str, Any]] = []
            batch_indices: List[int] = []

            def _flush_batch():
                if not batch_items:
                    return
                try:
                    jina_results = jina_gen.send(batch_items)
                except StopIteration:
                    raise RuntimeError("Jina generator terminated prematurely") from None
                with result_lock:
                    for idx, res in zip(batch_indices, jina_results, strict=True):
                        url = res.get("url", "")
                        content = res.get("content")
                        metadata = _get_metadata(res)
                        if json and content is not None:
                            try:
                                content = js.loads(content)
                            except Exception as exc:
                                self.logger.error(f"JSON decode error for {url}: {exc}")
                                content = None
                        completed[idx] = (url, content, metadata)
                batch_items.clear()
                batch_indices.clear()

            while True:
                item = jina_queue.get()
                if item is None:  # sentinel to stop the thread
                    _flush_batch()
                    break
                idx, payload = item
                batch_items.append({
                    "url": payload["url"],
                    "html_content": payload.get("html_content"),
                    "custom_css_selectors": self._get_custom_css_selector(payload["url"]),
                })
                batch_indices.append(idx)
                if len(batch_items) >= jina_batch_size:
                    _flush_batch()

        jina_thread = threading.Thread(target=_jina_worker, daemon=True)
        jina_thread.start()

        # ThreadPoolExecutor for non‑Jina extractors
        def _run_local(idx_item):
            idx, itm = idx_item
            try:
                _, content = self.extract(
                    itm["url"],
                    extractor=self._get_preferred(itm["url"], "extractor"),
                    json=json,
                    html_content=itm.get("html_content"),
                )
                return idx, itm["url"], content
            except Exception as exc:
                self.logger.error(f"Extraction failed for {itm['url']} (idx={idx}): {exc}")
                return idx, itm["url"], None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as local_pool:
            local_futures: Dict[int, concurrent.futures.Future] = {}

            # Stream the incoming items, dispatching work immediately
            for idx, item in html_items:
                pref = self._get_preferred(item["url"], "extractor")
                original_items[idx] = {**item, "extractor": pref}
                if "url" not in item:
                    self.logger.warning(f"Item {idx} missing 'url' – skipping")
                    continue

                pref = self._get_preferred(item["url"], "extractor")
                if pref == "jina.ai":
                    # Queue for the Jina worker (batching happens there)
                    jina_queue.put((idx, item))
                else:
                    # Submit to the local executor
                    local_futures[idx] = local_pool.submit(_run_local, (idx, item))

                # Yield any results that are ready in order
                while True:
                    with result_lock:
                        if next_to_yield in completed:
                            _, content, metadata = completed.pop(next_to_yield)
                            original = original_items[next_to_yield]
                            original_with_content = {
                                **original,
                                "content": content,
                                "extraction": metadata,
                            }
                            del original_with_content["extractor"]
                            yield original_with_content
                            next_to_yield += 1
                            continue
                    if next_to_yield in local_futures and local_futures[next_to_yield].done():
                        idx_y, _, content_y = local_futures.pop(next_to_yield).result()
                        original = original_items[idx_y]
                        original_with_content = {
                            **original,
                            "content": content_y,
                            "extraction": {
                                "status": "success" if content_y else "fail",
                                "extractor": original["extractor"],
                            },
                        }
                        del original_with_content["extractor"]
                        yield original_with_content
                        next_to_yield += 1
                        continue
                    break

        # Signal the Jina thread to finish and wait for it
        jina_queue.put(None)  # sentinel
        jina_thread.join()

        # Drain any remaining results (still respecting order)
        while True:
            with result_lock:
                if next_to_yield in completed:
                    _, content, metadata = completed.pop(next_to_yield)
                    original = original_items[next_to_yield]
                    original_with_content = {
                        **original,
                        "content": content,
                        "extraction": metadata,
                    }
                    del original_with_content["extractor"]
                    yield original_with_content
                    next_to_yield += 1
                    continue
            if next_to_yield in local_futures:
                idx_y, _, content_y = local_futures.pop(next_to_yield).result()
                original = original_items[idx_y]
                original_with_content = {
                    **original,
                    "content": content_y,
                    "extraction": {
                        "status": "success" if content_y else "fail",
                        "extractor": original["extractor"],
                    },
                }
                del original_with_content["extractor"]
                yield original_with_content
                next_to_yield += 1
                continue
            break

        # Clean‑up the Jina generator
        try:
            jina_gen.close()
        except Exception:
            pass
