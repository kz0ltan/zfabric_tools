import argparse
import httpx
import json
import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from web.extractor import WebExtractor
from web.jina import JinaAI

ENV_PATH = "~/.config/zfabric/.env"


def get_llm_clients(config_path: str, proxy: str = None):
    if not config_path:
        return []

    with open(config_path, "r") as f:
        config = json.load(f)

    clients = []
    for c in config["clients"]:
        http_client = None
        if proxy:
            transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy), verify=False)
            http_client = httpx.Client(transport=transport)

        clients.append(ChatOpenAI(**c, http_client=http_client))

    return clients


def load_urls(url_path: str):
    lines = []
    with open(url_path, encoding="utf-8") as f:
        for line in f:
            sline = line.strip()
            if sline:
                lines.append(sline)
    return lines


def main():
    parser = argparse.ArgumentParser(description="CLI tool to extract content using libraries")
    parser.add_argument("-u", "--url", required=False, help="Content URL")
    parser.add_argument(
        "-e",
        "--extractor",
        required=False,
        choices=["preferred", "newspaper4k", "trafilatura", "jina.ai"],
        default=None,
    )
    parser.add_argument(
        "-r",
        "--retrievers",
        default=[],
        required=False,
        action="append",
        choices=["preferred", "requests", "jina_api", "playwright"],
        help="What retrievers to use to get raw HTML content",
    )

    parser.add_argument(
        "-j",
        "--jina-profile",
        default="openai",
        required=False,
        choices=["jina.ai", "openai"],
        help="Profile to load from env for Jina.ai, default: openai",
    )
    parser.add_argument("--json", default=False, action="store_true", help="JSON output")
    parser.add_argument("--config-path", default=None, help="llm config file path")
    parser.add_argument("--bulk-url-path", default=None, help="URLs to extract, one per line")
    parser.add_argument("--proxy", default=None, help="http proxy for debugging")
    parser.add_argument(
        "--css",
        action="store_true",
        default=False,
        help="Run CSS selectors after retrieval and print result",
    )

    args = parser.parse_args()
    load_dotenv(os.path.expanduser(ENV_PATH))
    llm_clients = get_llm_clients(args.config_path, args.proxy)

    we_args = {}
    if args.extractor:
        we_args["extractor"] = args.extractor
    if len(args.retrievers):
        we_args["retrievers"] = args.retrievers

    extractor = WebExtractor(
        **we_args, jina_profile=args.jina_profile, llm_clients=llm_clients, proxy=args.proxy
    )
    if args.css and args.url:
        custom_css_selectors = extractor._get_custom_css_selector(args.url)
        print(f"\nSelectors:\n{custom_css_selectors}\n")
        html_content = extractor.retrieve(args.url, we_args["retrievers"][0])
        selected_content = JinaAI._apply_selector(html_content, custom_css_selectors)
        print(f"\nSelected content:\n{selected_content}\n\n")
    elif args.bulk_url_path:
        content_generator = extractor.bulk_retrieve(
            load_urls(args.bulk_url_path), retrievers=args.retrievers
        )
        start = time.time()
        contents = extractor.bulk_extract(content_generator)
        contents = list(contents)
        for content in contents:
            r_status = content["retrieval"]["status"] == "success"
            e_status = content["retrieval"]["status"] == "success"
            print("------")
            print(content["url"])
            print(f"{content['retrieval']['status']} ({content['retrieval']['retriever']})")
            print(f"{content['extraction']['status']} ({content['extraction']['extractor']})")
            if r_status and e_status:
                print(str(len(content["html_content"])) + " // " + str(len(content["content"])))
        end = time.time()
        print(f"Bulk extraction time: {end - start}")
        breakpoint()
    else:
        _, content = extractor.extract(args.url, json=args.json)
        print(content)


if __name__ == "__main__":
    main()
