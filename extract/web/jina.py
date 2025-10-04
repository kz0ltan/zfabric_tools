#!/usr/bin/env python3

# https://huggingface.co/jinaai/ReaderLM-v2
# https://colab.research.google.com/drive/1FfPjZwkMSocOLsEYH45B3B4NxDryKLGI?usp=sharing#scrollTo=yDR0dRNrwlB8

# https://github.com/BerriAI/litellm/issues/12070

import argparse
import base64
import itertools
import json as js
import os
import re
import sys
import time
import threading
from typing import Dict, Optional, Any, List

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
            proxies: Dict[str, str] = None
    ):
        """
        - profile: Used if llm_clients are NOT set
        """
        self.profile = profile
        self.logger = set_up_logging(loglevel)
        self._profile_client = None
        self.proxies = proxies

        # thread-based round-robin load balancer for multiple OpenAI endpoints
        self.llm_clients = [{"id": i, "client": client}
                            for i, client in enumerate(llm_clients)]
        self.client_cycle = itertools.cycle(self.llm_clients)
        self.lock = threading.Lock()  # not sure if itertools.cycle is thread-safe

    def get_next_client(self) -> Dict[str, Any]:
        with self.lock:
            return next(self.client_cycle)

    def get_html_content(
            self,
            url: str,
            proxies: Optional[Dict[str, str]] = None
    ):
        try:
            self.logger.info(
                f"Using Jina Reader to fetch **raw HTML** from: {url}")
            headers = {"X-Return-Format": "html"}
            response = requests.get(
                f"https://r.jina.ai/{url}",
                proxies=proxies,
                verify=False if proxies else True,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()
            self.logger.info(
                f"Retrieved raw HTML content: {len(response.text)}")
            return response.text
        except requests.exceptions.RequestException as e:
            error = f"Error during HTML retrieveal: {str(e)}"
            self.logger.error(error + f", {url}")
            raise RetrievalError(error)

    def _create_prompt(
            self,
            text: str,
            instruction: str = None,
            schema: str = None,
            lc_template: bool = False
    ) -> str:
        """
        Create a prompt for the model with optional instruction and JSON schema.

        Args:
            text (str): The input HTML text
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
                f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json{schema}```"
            )
        else:
            prompt = f"{instruction}\n```html\n{text}\n```"

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

    def clean_html(self, html: str, clean_svg: bool = False, clean_base64: bool = False):
        html = re.sub(JinaAI.SCRIPT_PATTERN, "", html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))
        html = re.sub(JinaAI.STYLE_PATTERN, "", html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))
        html = re.sub(JinaAI.META_PATTERN, "", html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))
        html = re.sub(JinaAI.COMMENT_PATTERN, "", html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))
        html = re.sub(JinaAI.LINK_PATTERN, "", html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))

        if clean_svg:
            html = self._replace_svg(html)

        if clean_base64:
            html = self._replace_base64_images(html)

        return html

    @staticmethod
    def _basic_auth(username: str, password: str):
        token = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _get_openai_client_from_profile(
            self,
            profile: Dict[str, str],
            proxies: Optional[Dict[str, str]] = None
    ) -> ChatOpenAI:

        if self._profile_client:
            return self._profile_client

        base_url = profile["base_url"]
        token = profile["api_key"]
        model = profile.get("model")
        timeout = profile.get("timeout", 60)
        proxy = proxies["http"] if proxies else None

        assert token is not None, f"API token missing from profile: {profile["type"]}"
        assert base_url is not None, f"Base URL missing from profile: {profile["type"]}"
        assert model is not None, f"Model name is missing from profile: {profile["type"]}"

        http_client = None
        if proxy:
            transport = httpx.HTTPTransport(
                proxy=httpx.Proxy(url=proxy), verify=False)
            http_client = httpx.Client(transport=transport)

        self._profile_client = ChatOpenAI(
            api_key=token,
            base_url=url,
            model=model,
            timeout=timeout,
            http_client=http_client
        )

        return self._profile_client

    def get_markdown_content_batched(
        self,
        html_contents: List[str],
        json: bool = False,
        schema: Dict[str, Any] = None,
        max_workers: int = 16  # total workers for all clients
    ):
        if len(self.llm_clients) == 0:
            raise ValueError("Batching needs predefined llm_clients")

        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    self.get_markdown_content,
                    html_content,
                    {"type": "openai"},
                    json=json,
                    schema=schema,
                    proxies=self.proxies
                ): i for i, content in enumerate(html_contents)
            }

            for future in as_completed(future_to_id):
                request_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "request_id": request_id,
                        "error": str(e),
                        "status": "error"
                    })

        results.sort(key=lambda x: x["request_id"])

        end_time = time.time()
        elapsed = end_time - start_time

        self.logger.info(f"Batch finished in {elapsed}")

        return results

    def get_markdown_content(
        self,
        html_content: str,
        url: Optional[str] = None,  # for jina.ai
        json: bool = False,
        schema: Dict[str, Any] = None,
        max_retries=3
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

        if self.profile["type"] == "jina.ai":
            assert url is not None, "Missing URL for jina.ai extractor"
            assert self.profile["token"] is not None, "Missing jina.ai token"

            headers = {"Authorization": "Bearer " + self.profile["token"]}
            if json:
                headers["Accept"] = "application/json"

            if schema:
                self.logger.warning("jina.ai API only uses it's own schema")

            self.logger.info("Requesting jina.ai API for markdown content...")

            response = requests.get(
                "https://r.jina.ai/" + url,
                headers=headers,
                timeout=10,
                proxies=self.proxies
            )

            response.raise_for_status()
            return {
                "type": "jina.ai",
                "content": response.text,
                "status": "success",
                "attempt": attempt + 1
            }

        elif self.profile["type"] == "openai":

            clean_html = self.clean_html(
                html_content,
                clean_svg=True,
                clean_base64=True
            )

            self.logger.info(
                f"Extracting from cleaned HTML of size: {len(clean_html)}")

            schema = js.dumps(extract_schema, indent=2) if json else None
            prompt = self._create_prompt(clean_html, schema=schema)

            client = None
            for attempt in range(max_retries):

                if len(self.llm_clients) == 0:
                    client = self._get_openai_client_from_profile(
                        self.profile,
                        proxies=self.proxies
                    )
                else:
                    client = self.get_next_client()

                try:
                    template = ChatPromptTemplate.from_messages(prompt)
                    chain = template | client["client"]
                    response = chain.invoke({})
                    return {
                        "type": "openai",
                        "client": client["client"].openai_api_base,
                        "content": response.content,
                        "status": "success",
                        "attempt": attempt + 1
                    }
                except Exception as e:
                    self.logger.error(
                        f"Error on llm client {client["client"].openai_api_base}: " + str(e))

                    if attempt < max_retries - 1:
                        time_sleep(1)
                    continue

            return {
                "type": "openai",
                "error": f"Failed after {max_retries} attempts",
                "status": "error"
            }

        else:
            raise ValueError("Unknown profile type: " +
                             str(self.profile["type"]))

    @ staticmethod
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

    @ staticmethod
    def get_profile_from_env(profile_type: str = None) -> Dict:
        """Load profile from env based on profile_type OR JINA_PROFILE env var"""
        profile_type = profile_type or os.getenv("JINA_PROFILE", None)

        if profile_type is None:
            raise ValueError(
                "No profile is defined as argument OR JINA_PROFILE env")

        if profile_type == "jina.ai":
            return {
                "type": "jina.ai",
                "token": os.getenv("JINA_TOKEN")
            }

        elif profile_type == "openai":
            return {
                "type": "openai",
                "base_url": os.getenv("OPENAI_ENDPOINT"),
                "api_key": os.getenv("OPENAI_TOKEN"),
                "model": os.getenv("OPENAI_MODEL")
            }

        else:
            raise ValueError("Unknown profile: " + str(profile_type))
