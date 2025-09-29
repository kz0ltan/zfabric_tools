import argparse

from web.extractor import WebExtractor


def main():
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
    _, content = extractor.extract(args.url, json=args.json)
    print(content)


if __name__ == '__main__':
    main()
