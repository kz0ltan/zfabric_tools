#!/usr/bin/env python3

# https://huggingface.co/jinaai/ReaderLM-v2
# https://colab.research.google.com/drive/1FfPjZwkMSocOLsEYH45B3B4NxDryKLGI?usp=sharing#scrollTo=yDR0dRNrwlB8

# https://github.com/BerriAI/litellm/issues/12070

import base64
import itertools
import httpx
import json as js
import os
import re
import time
import threading
from typing import Dict, Optional, Any, List, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .exceptions import RetrievalError
from .common import set_up_logging


class JinaAI:
    # (REMOVE <SCRIPT> to </script> and variations)
    # mach any char zero or more times
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    # text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML <STYLE> to </style> and variations)
    # mach any char zero or more times
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    # text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML <META> to </meta> and variations)
    META_PATTERN = r"<[ ]*meta.*?>"  # mach any char zero or more times
    # text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML COMMENTS <!-- to --> and variations)
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"  # mach any char zero or more times
    # text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    # (REMOVE HTML LINK <LINK> to </link> and variations)
    LINK_PATTERN = r"<[ ]*link.*?>"  # mach any char zero or more times

    # (REPLACE base64 images)
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'

    # (REPLACE <svg> to </svg> and variations)
    SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

    def __init__(
        self,
        profile: Dict[str, str],
        loglevel: int = 20,
        llm_clients: List[ChatOpenAI] = [],
        proxies: Dict[str, str] = None,
    ):
        """
        - profile: Used if llm_clients are NOT set
        """
        self.profile = profile
        self.logger = set_up_logging(loglevel)
        self._profile_client = None
        self.proxies = proxies

        # thread-based round-robin load balancer for multiple OpenAI endpoints
        self.llm_clients = llm_clients
        self.client_cycle = itertools.cycle(self.llm_clients)
        self.lock = threading.Lock()  # not sure if itertools.cycle is thread-safe

    def get_next_client(self) -> Dict[str, Any]:
        with self.lock:
            return next(self.client_cycle)

    def get_html_content(self, url: str, proxies: Optional[Dict[str, str]] = None):
        try:
            self.logger.info(
                f"Using Jina Reader to fetch **raw HTML** from: {url}")
            headers = {"X-Return-Format": "html", "Accept": "application/json"}

            response = requests.get(
                f"https://r.jina.ai/{url}",
                proxies=proxies,
                verify=False if proxies else True,
                headers=headers,
                timeout=10,
            )

            response.raise_for_status()
            if response.ok:
                json_body = js.loads(response.text)
                data = json_body["data"]
                html = data.get("html")
                warning = data.get("warning")
                if warning:
                    self.logger.warning(warning + f", {url}")
                    raise RetrievalError(warning)
                else:
                    self.logger.info(
                        f"Retrieved raw HTML content: {len(html)}")
                    return html
            else:
                raise RetrievalError(
                    f"{response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            error = f"Error during HTML retrieveal: {str(e)}"
            self.logger.error(error + f", {url}")
            raise RetrievalError(error)

    def _create_prompt(
        self, instruction: str = None, schema: str = None, lc_template: bool = False
    ) -> str:
        """
        Create a prompt for the model with optional instruction and JSON schema.

        WARNING: if the output is going to be resolved with langchain, avoid
            adding text to it, otherwise resolving the input againt might
            cause issues!

        Args:
            instruction (str, optional): Custom instruction for the model
            schema (str, optional): JSON schema for structured extraction

        Returns:
            str: The formatted prompt
        """
        if not instruction:
            instruction = (
                "Extract the main content from the given HTML and convert it to Markdown format."
            )

        if schema:
            instruction = "Extract the specified information from the input content and present it in a structured JSON format."

            prompt = (
                instruction
                + "\n```html\n{clean_html}\n```"
                + f"\n\nThe JSON schema is as follows:```json\n{schema}```"
            )
        else:
            prompt = instruction + "\n```html\n{clean_html}\n```"

        if lc_template:
            return [("user", prompt)]

        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    @staticmethod
    def _replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
        return re.sub(
            JinaAI.SVG_PATTERN,
            lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
            html,
            flags=re.DOTALL,
        )

    @staticmethod
    def _replace_base64_images(html: str, new_image_src: str = "#") -> str:
        return re.sub(JinaAI.BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

    @staticmethod
    def _has_base64_images(text: str) -> bool:
        base64_content_pattern = r'data:image/[^;]+;base64,[^"]+'
        return bool(re.search(base64_content_pattern, text, flags=re.DOTALL))

    @staticmethod
    def _has_svg_components(text: str) -> bool:
        return bool(re.search(JinaAI.SVG_PATTERN, text, flags=re.DOTALL))

    @staticmethod
    def _apply_selector(html: str, selectors: [str]) -> str:
        if not selectors:
            return html
        soup = BeautifulSoup(html, "html.parser")
        elements = []
        for selector in selectors:
            elements.extend(soup.select(selector))
        return "".join([str(elem) for elem in elements])

    def clean_html(
        self,
        html: str,
        clean_svg: bool = False,
        clean_base64: bool = False,
        custom_css_selectors: List[str] = None,
    ):
        html = re.sub(
            JinaAI.SCRIPT_PATTERN, "", html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )
        html = re.sub(
            JinaAI.STYLE_PATTERN, "", html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )
        html = re.sub(
            JinaAI.META_PATTERN, "", html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )
        html = re.sub(
            JinaAI.COMMENT_PATTERN, "", html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )
        html = re.sub(
            JinaAI.LINK_PATTERN, "", html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL)
        )

        if clean_svg:
            html = self._replace_svg(html)

        if clean_base64:
            html = self._replace_base64_images(html)

        if custom_css_selectors:
            html = self._apply_selector(html, custom_css_selectors)

        return html

    @staticmethod
    def _basic_auth(username: str, password: str):
        token = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _get_openai_client_from_profile(
        self, profile: Dict[str, str], proxies: Optional[Dict[str, str]] = None
    ) -> ChatOpenAI:
        if self._profile_client:
            return self._profile_client

        base_url = profile["base_url"]
        token = profile["api_key"]
        model = profile.get("model")
        timeout = profile.get("timeout", 60)
        proxy = proxies["http"] if proxies else None

        assert token is not None, f"API token missing from profile: {profile['type']}"
        assert base_url is not None, f"Base URL missing from profile: {profile['type']}"
        assert model is not None, f"Model name is missing from profile: {profile['type']}"

        http_client = None
        if proxy:
            transport = httpx.HTTPTransport(
                proxy=httpx.Proxy(url=proxy), verify=False)
            http_client = httpx.Client(transport=transport)

        self._profile_client = ChatOpenAI(
            api_key=token, base_url=base_url, model=model, timeout=timeout, http_client=http_client
        )

        return self._profile_client

    def warm_up_clients(self) -> None:
        """
        Simple calls to get_markdown_content() to warm up
        the openai inference endpoints using dummy data
        """
        tp = ThreadPoolExecutor(max_workers=len(self.llm_clients))
        for i in range(len(self.llm_clients)):
            tp.submit(
                self.get_markdown_content,
                url=f"warmup://{i}",
                html_content="<p>warm‑up</p>",
                force_openai=True,
            )

    def get_markdown_content_batched(
        self,
        *,
        json: bool = False,
        schema: Optional[Dict] = None,
        max_workers: int = 8,
    ) -> Generator[List[Dict[str, Any]], List[str], None]:
        """
        Synchronous generator that yields a list of Jina responses for each
        batch of HTML strings sent via ``send()``.

        Usage example:

            gen = jina.get_markdown_content_batched(json=True)
            next(gen)                     # prime the generator
            batch1 = ["<html>…</html>", "<html>…</html>"]
            results1 = gen.send(batch1)   # -> list of 2 dicts (ordered)

            batch2 = ["<html>…</html>"]
            results2 = gen.send(batch2)

            gen.close()   # shuts down the internal executor
        """
        if len(self.llm_clients) == 0:
            raise ValueError("Batching needs predefined llm_clients")

        executor = ThreadPoolExecutor(max_workers=max_workers)

        # self.warm_up_clients()

        # Prime the generator – the first ``send`` will deliver the first batch
        batch: List[str] = yield []  # caller must call ``next(gen)`` first

        while True:
            if batch is None:  # if caller wants to stop
                break

            # Submit the current batch to the executor
            future_to_id = {
                executor.submit(
                    self.get_markdown_content,
                    html["html_content"],
                    url=html["url"],
                    json=json,
                    schema=schema,
                    custom_css_selectors=html["custom_css_selectors"],
                    force_openai=True,
                ): idx
                for idx, html in enumerate(batch)
            }

            # Collect results preserving the original order
            results: List[Dict[str, Any]] = [None] * len(batch)  # type: ignore
            for fut in as_completed(future_to_id):
                idx = future_to_id[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    results[idx] = {
                        "request_id": idx,
                        "error": str(exc),
                        "status": "error",
                    }

            # Yield the ordered list back to the caller and wait for the next batch
            batch = yield results

        # Clean‑up
        executor.shutdown(wait=True)

    def get_markdown_content(
        self,
        html_content: str,
        url: Optional[str] = None,  # for jina.ai
        json: bool = False,
        custom_css_selectors: List[str] = None,
        schema: Dict[str, Any] = None,
        max_retries: int = 2,  # max_retries on the LLM client multiplies this!
        force_openai: bool = False,
    ):
        extract_schema = (
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "News thread title"},
                    "date": {"type": "string", "description": "Date of publication"},
                    "summary": {"type": "string", "description": "Article summary"},
                    "keywords": {"type": "list", "description": "Descriptive keywords"},
                },
                "required": ["title", "date", "summary", "keywords"],
            }
            if schema is None
            else schema
        )

        if not force_openai and self.profile["type"] == "jina.ai":
            assert url is not None, "Missing URL for jina.ai extractor"
            assert self.profile["token"] is not None, "Missing jina.ai token"

            headers = {
                "Authorization": "Bearer " + self.profile["token"],
                "Accept": "application/json",
            }
            if json:
                headers["Accept"] = "application/json"

            if schema:
                self.logger.warning("jina.ai API only uses it's own schema")

            self.logger.info("Requesting jina.ai API for markdown content...")

            response = requests.get(
                "https://r.jina.ai/" + url, headers=headers, timeout=10, proxies=self.proxies
            )

            response.raise_for_status()
            if response.ok:
                json_body = js.loads(response.text)
                data = json_body["data"]
                content = data.get("content")
                warning = data.get("warning")
                if warning:
                    self.logger.warning(warning + f", {url}")
                    raise RetrievalError(warning)
                else:
                    return {"type": "jina.ai", "content": content, "status": "success"}
            else:
                return {"type": "jina.ai", "status": "error", "error": response.text}

        elif force_openai or self.profile["type"] == "openai":
            clean_html = self.clean_html(
                html_content,
                clean_svg=True,
                clean_base64=True,
                custom_css_selectors=custom_css_selectors,
            )

            self.logger.info(
                f"Extracting from cleaned HTML of size: {len(clean_html)}")

            schema = js.dumps(extract_schema, indent=2) if json else None
            prompt = self._create_prompt(schema=schema, lc_template=True)

            errors = []
            client = None
            for attempt in range(max_retries):
                if len(self.llm_clients) == 0:
                    client = self._get_openai_client_from_profile(
                        self.profile, proxies=self.proxies
                    )
                else:
                    client = self.get_next_client()

                try:
                    template = ChatPromptTemplate.from_messages(prompt)
                    chain = template | client
                    response = chain.invoke({"clean_html": clean_html})
                    return {
                        "type": "openai",
                        "client": client.openai_api_base,
                        "content": response.content,
                        "status": "success",
                        "attempt": attempt + 1,
                        "prev_errors": errors,
                        "usage_metadata": response.usage_metadata,
                    }
                except Exception as e:
                    self.logger.error(
                        f"Error using llm client {client.openai_api_base}: "
                        + str(e)
                        + f" ({str(url)})"
                    )
                    errors.append({
                        "type": "openai",
                        "client": client.openai_api_base,
                        "status": "error",
                        "attempt": attempt + 1,
                        "error": str(e),
                    })

                    if attempt < max_retries - 1:
                        time.sleep(1)
                    continue

            return {
                "type": "openai",
                "error": f"Failed after {max_retries} attempts",
                "status": "error",
                "prev_errors": errors,
            }

        else:
            raise ValueError("Unknown profile type: " +
                             str(self.profile["type"]))

    @staticmethod
    def strip_markdown(content: str) -> str:
        """Remove ```markdown\n<content>\n``` frame from content"""
        if content.strip().startswith("```markdown"):
            content = re.sub(r"^```markdown\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())

        # remove ```json\n<content>\n``` frame
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        return content

    @staticmethod
    def get_profile_from_env(profile_type: str = None) -> Dict:
        """Load profile from env based on profile_type OR JINA_PROFILE env var"""
        profile_type = profile_type or os.getenv("JINA_PROFILE", None)

        if profile_type is None:
            raise ValueError(
                "No profile is defined as argument OR JINA_PROFILE env")

        if profile_type == "jina.ai":
            return {"type": "jina.ai", "token": os.getenv("JINA_TOKEN")}

        elif profile_type == "openai":
            return {
                "type": "openai",
                "base_url": os.getenv("OPENAI_ENDPOINT"),
                "api_key": os.getenv("OPENAI_TOKEN"),
                "model": os.getenv("OPENAI_MODEL"),
            }

        else:
            raise ValueError("Unknown profile: " + str(profile_type))
