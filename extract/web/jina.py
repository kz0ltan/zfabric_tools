#!/usr/bin/env python3

# https://huggingface.co/jinaai/ReaderLM-v2
# https://colab.research.google.com/drive/1FfPjZwkMSocOLsEYH45B3B4NxDryKLGI?usp=sharing#scrollTo=yDR0dRNrwlB8

import argparse
import base64
from dotenv import load_dotenv
import json as js
import logging
import os
import re
import sys
from typing import Dict

import requests
import ollama
import openai

ENV_PATH = "~/.config/zfabric/.env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def get_html_content(url: str, use_jina_html_api: bool = False):
    api_url = f"https://r.jina.ai/{url}"
    headers = {"X-Return-Format": "html"}
    try:
        if use_jina_html_api:
            logger.info(
                f"We will use Jina Reader to fetch the **raw HTML** from: {url}")
            response = requests.get(api_url, headers=headers, timeout=10)
        else:
            response = requests.get(url, timeout=10)

        response.raise_for_status()
        logger.info(f"Retrieved raw HTML content: {len(response.text)}")
        return response.text
    except requests.exceptions.RequestException as e:
        return f"error: {str(e)}"


def create_prompt(text: str, instruction: str = None, schema: str = None) -> str:
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

    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


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


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def has_base64_images(text: str) -> bool:
    base64_content_pattern = r'data:image/[^;]+;base64,[^"]+'
    return bool(re.search(base64_content_pattern, text, flags=re.DOTALL))


def has_svg_components(text: str) -> bool:
    return bool(re.search(SVG_PATTERN, text, flags=re.DOTALL))


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(SCRIPT_PATTERN, "", html, flags=(
        re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(STYLE_PATTERN, "", html, flags=(
        re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(META_PATTERN, "", html, flags=(
        re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(COMMENT_PATTERN, "", html, flags=(
        re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(LINK_PATTERN, "", html, flags=(
        re.IGNORECASE | re.MULTILINE | re.DOTALL))

    if clean_svg:
        html = replace_svg(html)

    if clean_base64:
        html = replace_base64_images(html)

    return html


def _basic_auth(username: str, password: str):
    token = base64.b64encode(
        f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _get_ollama_client(username: str, password: str, host: str):
    headers = {"Authorization": _basic_auth(
        username, password)} if username and password else None

    if host:
        return ollama.Client(host=host, headers=headers)


def _get_openai_client(url: str, token: str, max_retries=5):
    return openai.OpenAI(api_key=token, base_url=url, max_retries=max_retries)


def get_markdown_content(
    url: str,
    profile: Dict,
    json: bool = False,
    schema: Dict = None,
    use_jina_html_api: bool = False,
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
        html = get_html_content(url, use_jina_html_api=use_jina_html_api)
        html = clean_html(html, clean_svg=True, clean_base64=True)
        prompt = create_prompt(html, schema=js.dumps(
            extract_schema, indent=2) if json else None)

        username = profile["username"]
        password = profile["password"]
        host = profile["url"]

        client = _get_ollama_client(username, password, host)

        try:
            response = client.chat(
                model="ReaderLM-v2-bf16_kzoltan:latest",
                messages=prompt,
                stream=False,
            )
            return response.message.content
        except Exception as e:
            logger.error("Error: " + str(e))
            sys.exit(1)

    elif profile["type"] == "jina.ai":
        token = profile["token"]
        headers = {"Authorization": "Bearer " + token}
        if json:
            headers["Accept"] = "application/json"

        response = requests.get("https://r.jina.ai/" +
                                url, headers=headers, timeout=10)

        return response.text

    elif profile["type"] == "openai":
        token = profile["api_key"]
        base_url = profile["base_url"]
        model = profile.get("model")
        max_retries = profile.get("max_retries")

        html = get_html_content(url, use_jina_html_api=use_jina_html_api)
        html = clean_html(html, clean_svg=True, clean_base64=True)
        logger.info(
            f"Extracting content from cleaned HTML of size: {len(html)}")
        prompt = create_prompt(html, schema=js.dumps(
            extract_schema, indent=2) if json else None)

        client = _get_openai_client(base_url, token, max_retries)

        try:
            response = client.chat.completions.create(
                model=model or "ReaderLM-v2-BF16",
                messages=prompt,
                stream=False,
                timeout=300,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error: " + str(e))
            sys.exit(1)

    else:
        raise ValueError("Unknown profile type: " + str(profile["type"]))


def strip_markdown(content: str):
    # remove ```markdown\n<content>\n``` frame
    if content.strip().startswith("```markdown"):
        content = re.sub(r"^```markdown\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())

    # remove ```json\n<content>\n``` frame
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)

    return content


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and convert content using Jina.AI ReaderLM-v2"
    )
    parser.add_argument("url", help="URL of the content to download")
    parser.add_argument(
        "-r", "--profile", default="jina.ai", help="Select profile (jina.ai/ollama/openai; default: jina.ai API)"
    )
    parser.add_argument("--json", default=False,
                        action="store_true", help="Use predefined schema")
    parser.add_argument(
        "--jina_html_api",
        default=False,
        action="store_true",
        help="Use Jina.ai API to get raw html content",
    )

    return parser.parse_args()


def get_profile(profile_name: str = None):
    """Load profile from env based on profile_name OR JINA_PROFILE env var"""
    load_dotenv(os.path.expanduser(ENV_PATH))
    profile_name = profile_name or os.getenv("JINA_PROFILE", None)

    if profile_name is None:
        raise ValueError(
            "No profile is defined as argument OR JINA_PROFILE env")
    if profile_name == "jina.ai":
        return {"type": "jina.ai", "token": os.getenv("JINA_TOKEN")}
    elif profile_name == "openai":
        return {
            "type": "openai",
            "url": os.getenv("OPENAI_ENDPOINT"),
            "token": os.getenv("OPENAI_TOKEN")
        }
    elif profile_name == "ollama":
        return {
            "type": "ollama",
            "username": os.getenv("OLLAMA_USERNAME"),
            "password": os.getenv("OLLAMA_PASSWORD"),
            "url": os.getenv("OLLAMA_URL"),
        }
    else:
        raise ValueError("Unknown profile: " + str(profile_name))


def main():
    args = parse_args()
    profile = get_profile(args.profile)
    content = get_markdown_content(
        args.url, profile=profile, json=args.json, use_jina_html_api=args.jina_html_api
    )
    print(content)


if __name__ == "__main__":
    main()
