import argparse
import json
import os

from dotenv import load_dotenv

from .reddit import Reddit

ENV_PATH = "~/.config/zfabric/.env"


def parse_parameters():
    parser = argparse.ArgumentParser(description="Download saved Reddit posts and comments")
    parser.add_argument(
        "--client_id", type=str, default=None, required=False, help="Reddit API client ID"
    )
    parser.add_argument(
        "--client_secret", type=str, default=None, required=False, help="Reddit API client secret"
    )
    parser.add_argument(
        "--username", type=str, default=None, required=False, help="Reddit username"
    )
    parser.add_argument(
        "--password", type=str, default=None, required=False, help="Reddit password"
    )
    parser.add_argument(
        "--user_agent",
        type=str,
        required=False,
        default="zFabric Reddit client",
        help="User agent for Reddit API requests",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url", type=str, required=False, help="Reddit URL to download submission/comments"
    )
    group.add_argument(
        "--saved",
        required=False,
        action="store_true",
        help="Download user's saved posts/comments",
    )

    return parser.parse_args()


def main():
    args = parse_parameters()
    load_dotenv(os.path.expanduser(ENV_PATH))
    r = Reddit(
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent=args.user_agent,
        username=args.username,
        password=args.password,
    )
    if args.saved:
        results = r.get_saved_posts()
        filtered_results = []
        for r in results:
            if r["type"] == "comment":
                continue
            filtered_results.append(
                {
                    "title": r["title"],
                    "text": r["selftext"],
                    "subreddit": r["subreddit"],
                }
            )
        print(json.dumps(filtered_results, indent=2))
    elif args.url:
        submission = r.get_submission(args.url)
        r.print_post(submission)
        r.print_comment_tree(submission.comments)


if __name__ == "__main__":
    main()
