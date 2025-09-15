#!/usr/bin/env python3

import argparse
import json as js
from urllib.parse import urlparse
from typing import List, Optional, Any, Union, Dict

import requests

from . import jina

ENV_PATH = "~/.config/zfabric/.env"

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark
# https://github.com/goose3/goose3


class WebExtractor:
    def __init__(self, lib: str = "jina.ai", profile: Optional[Union[str, Dict[str, Any]]] = None):
        self.lib = lib
        if self.lib == "jina.ai":
            if isinstance(profile, Dict):
                self.profile = profile
            else:
                self.profile = jina.get_profile(profile)

    def extract(self, url: str, json: bool = False) -> str:
        """Extract content from url using lib"""
        if self.lib == "newspaper3k":
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()
            return article.text

        elif self.lib == "readability-lxml":
            from readability import Document

            response = requests.get(url, timeout=10)
            doc = Document(response.content)
            return doc.summary()

        elif self.lib == "jina.ai":
            content = jina.get_markdown_content(
                url, profile=self.profile, json=json)

            if self.profile["type"] in ("ollama", "openai"):
                content = jina.strip_markdown(content)

            if json:
                return js.loads(content)
            else:
                return content

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
            url_data = self.extract(parsed_line["url"], json=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI tool to extract content using libraries")
    parser.add_argument("url", help="Content URL")
    parser.add_argument(
        "-l",
        "--library",
        required=False,
        choices=["newspaper3k", "readability-lxml", "jina.ai"],
        default="jina.ai",
    )
    parser.add_argument(
        "-r",
        "--profile",
        default="jina.ai",
        required=False,
        help="Profile to use for Jina.ai (default jina.ai API, ollama is the alternative)",
    )
    parser.add_argument("-j", "--json", default=False,
                        action="store_true", help="JSON output")

    args = parser.parse_args()
    extractor = WebExtractor(args.library, args.profile)
    print(extractor.extract(args.url, json=args.json))
