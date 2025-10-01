#!/usr/bin/env python3

# https://huggingface.co/jinaai/ReaderLM-v2
# https://colab.research.google.com/drive/1FfPjZwkMSocOLsEYH45B3B4NxDryKLGI?usp=sharing#scrollTo=yDR0dRNrwlB8

# https://github.com/BerriAI/litellm/issues/12070

import argparse
import base64
import json as js
import os
import re
import sys
from typing import Dict, Optional, Any

import requests
import ollama
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

    def __init__(self, loglevel: int = 20):
        self.logger = set_up_logging(loglevel)

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
            self.logger.error(error)
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

    def _get_ollama_client(username: str, password: str, host: str):
        print("Deprecated: use ChatOllama from langchain_ollama package instead!")

        headers = {"Authorization": JinaAI._basic_auth(
            username, password)} if username and password else None

        if host:
            return ollama.Client(host=host, headers=headers)

    @staticmethod
    def _get_openai_client(url: str, token: str, max_retries: int = 5, proxy: str = None):
        http_client = None
        if proxy:
            transport = httpx.HTTPTransport(
                proxy=httpx.Proxy(url=proxy), verify=False)
            http_client = httpx.Client(transport=transport)
        return ChatOpenAI(
            api_key=token,
            base_url=url,
            max_retries=max_retries,
            http_client=http_client
        )

    def get_markdown_content(
        self,
        html_content: str,
        profile: Dict[str, str],
        llm_client: ChatOpenAI = None,
        url: Optional[str] = None,  # for jina.ai
        json: bool = False,
        schema: Dict[str, Any] = None,
        retriever: str = "requests",
        proxies: Optional[Dict[str, str]] = None
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

        if profile["type"] == "ollama":
            clean_html = clean_html(
                html_content, clean_svg=True, clean_base64=True)
            schema = js.dumps(extract_schema, indent=2) if json else None

            self.logger.info(
                f"Extracting from cleaned HTML of size: {len(clean_html)}")
            prompt = self._create_prompt(clean_html, schema=schema)

            username = profile["username"]
            password = profile["password"]
            ollama_host = profile["url"]
            model = profile["model"]

            assert username is not None, f"Username missing from profile: {profile["name"]}"
            assert password is not None, f"Password missing from profile: {profile["name"]}"
            assert ollama_url is not None, f"URL missing from profile: {profile["name"]}"

            # TODO: replace with langchain
            # TODO: proxy
            client = self._get_ollama_client(username, password, ollama_url)

            try:
                response = client.chat(
                    model=model or "ReaderLM-v2-Q8_0",
                    messages=prompt,
                    stream=False,
                )
                return response.message.content
            except Exception as e:
                self.logger.error("Error: " + str(e))
                sys.exit(1)

        elif profile["type"] == "jina.ai":
            assert url is not None, "Missing URL for jina.ai extractor"
            assert profile["token"] is not None, "Missing jina.ai token"

            headers = {"Authorization": "Bearer " + profile["token"]}
            if json:
                headers["Accept"] = "application/json"

            response = requests.get(
                "https://r.jina.ai/" + url,
                headers=headers,
                proxies=proxies,
                timeout=10
            )

            return response.text

        elif profile["type"] == "openai":
            token = profile["api_key"]
            base_url = profile["base_url"]
            model = profile.get("model")
            max_retries = profile.get("max_retries", 5)
            timeout = profile.get("timeout", 60)
            proxy = proxies["https"],

            assert token is not None, f"API token missing from profile: {profile["name"]}"
            assert base_url is not None, f"Base URL missing from profile: {profile["name"]}"

            clean_html = clean_html(
                html_content, clean_svg=True, clean_base64=True)
            self.logger.info(
                f"Extracting from cleaned HTML of size: {len(clean_html)}")

            schema = js.dumps(extract_schema, indent=2) if json else None
            prompt = self._create_prompt(clean_html, schema=schema)

            if llm_client is None:
                llm_client = self._get_openai_client(
                    base_url,
                    token,
                    max_retries,
                    proxy=proxy
                )

            try:
                template = ChatPromptTemplate.from_messages(prompt)
                chain = template | llm_client
                response = chain.invoke({})
                return html, response.content
            except Exception as e:
                self.logger.error("Error: " + str(e))
                sys.exit(1)

        else:
            raise ValueError("Unknown profile type: " + str(profile["type"]))

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
    def get_profile(profile_name: str = None) -> Dict:
        """Load profile from env based on profile_name OR JINA_PROFILE env var"""
        profile_name = profile_name or os.getenv("JINA_PROFILE", None)

        if profile_name is None:
            raise ValueError(
                "No profile is defined as argument OR JINA_PROFILE env")

        if profile_name == "jina.ai":
            return {"type": "jina.ai", "token": os.getenv("JINA_TOKEN")}
        elif profile_name == "openai":
            return {
                "name": profile_name,
                "type": "openai",
                "base_url": os.getenv("OPENAI_ENDPOINT"),
                "api_key": os.getenv("OPENAI_TOKEN")
            }
        elif profile_name == "ollama":
            return {
                "name": profile_name,
                "type": "ollama",
                "username": os.getenv("OLLAMA_USERNAME"),
                "password": os.getenv("OLLAMA_PASSWORD"),
                "url": os.getenv("OLLAMA_URL"),
            }
        else:
            raise ValueError("Unknown profile: " + str(profile_name))
