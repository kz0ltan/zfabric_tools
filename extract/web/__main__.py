import argparse
import os

from dotenv import load_dotenv

from web.extractor import WebExtractor

ENV_PATH = "~/.config/zfabric/.env"


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to extract content using libraries")
    parser.add_argument("url", help="Content URL")
    parser.add_argument(
        "-l",
        "--library",
        required=False,
        choices=["newspaper3k", "trafilatura", "jina.ai"],
        default="jina.ai",
    )
    parser.add_argument(
        "-r",
        "--profile",
        default="jina.ai",
        required=False,
        help="Profile to use for Jina.ai (default jina.ai API, ollama/openai are alternatives)",
    )
    parser.add_argument("-j", "--json", default=False,
                        action="store_true", help="JSON output")
    parser.add_argument(
        "--retriever",
        default="requests",
        help="What retriever to use to get raw HTML content: requests/jina_api"
    )

    args = parser.parse_args()
    load_dotenv(os.path.expanduser(ENV_PATH))
    extractor = WebExtractor(args.library, args.profile)
    _, content = extractor.extract(args.url, json=args.json)
    print(content)


if __name__ == '__main__':
    main()
