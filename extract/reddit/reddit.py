#!/usr/bin/env python3

"""Download public/user information from Reddit"""

import argparse
import datetime
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv
import praw

__all__ = ["Reddit"]

ENV_PATH = "~/.config/zfabric/.env"


class Reddit:
    """Reddit download object"""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = "Reddit client",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.credentials = {
            "client_id": client_id if client_id else os.getenv("REDDIT_CLIENT_ID"),
            "client_secret": client_secret if client_secret else os.getenv("REDDIT_CLIENT_SECRET"),
            "username": username if username else os.getenv("REDDIT_USERNAME"),
            "password": password if password else os.getenv("REDDIT_PASSWORD"),
        }
        for k, v in self.credentials.items():
            if v is None:
                print(f"Missing {k}")
                sys.exit(1)

        self.reddit = praw.Reddit(**self.credentials, user_agent=user_agent)

    def get_saved_posts(self, limit=None):
        """Get user's saved posts and comments."""
        saved_items = []
        count = 0

        # https://praw.readthedocs.io/en/stable/code_overview/models/submission.html#praw.models.Submission
        for item in self.reddit.user.me().saved(limit=limit):
            item_type = "post" if isinstance(item, praw.models.Submission) else "comment"

            saved_item = {
                "id": item.id,
                "type": item_type,
                "created_utc": datetime.datetime.fromtimestamp(item.created_utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "subreddit": item.subreddit.display_name,
                "permalink": f"https://www.reddit.com{item.permalink}"
                if hasattr(item, "permalink")
                else None,
                "url": item.url if hasattr(item, "url") else None,
            }

            # Add type-specific attributes
            if item_type == "post":
                saved_item.update({
                    "title": item.title,
                    "author": item.author.name if item.author else "[deleted]",
                    "selftext": item.selftext if hasattr(item, "selftext") else "",
                    "is_self": item.is_self if hasattr(item, "is_self") else None,
                    "score": item.score,
                })
            elif item_type == "comment":
                saved_item.update({
                    "body": item.body if hasattr(item, "body") else "",
                    "author": item.author.name if item.author else "[deleted]",
                    "parent_id": item.parent_id if hasattr(item, "parent_id") else None,
                    "score": item.score,
                })
            else:
                raise ValueError(f"Unknown item type: '{item_type}'")

            saved_items.append(saved_item)
            count += 1
            if count % 10 == 0:
                print(f"Fetched {count} items so far...")

        return saved_items

    def get_documents(self, limit: Optional[int] = None):
        documents = self.get_saved_posts(limit=limit)
        ndata = []
        for item in documents:
            if item["type"] == "post":
                url = item["permalink"]
                date = item["created_utc"]
                title = item["title"]
                text = item["selftext"]
                tags = ["reddit", item["subreddit"].lower()]
            elif item["type"] == "comment":
                url = item["permalink"]
                date = item["created_utc"]
                title = "comment"
                text = item["body"]
                tags = ["reddit", item["subreddit"].lower()]

            ndata.append({
                "url": url,
                "date": date,
                "title": title,
                "summary": text,
                "tags": tags,
                "source": "reddit",
            })

        return ndata

    def get_all_comments_of_post(self, post_url: str):
        submission = self.reddit.submission(url=post_url)
        submission.comments.replace_more(limit=None)  # Load all nested comments

        comments = []
        for comment in submission.comments.list():
            comments.append(comment.body)

        return comments

    def __call__(self):
        return self.get_saved_posts(self.reddit, limit=None)


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


if __name__ == "__main__":
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
            filtered_results.append({
                "title": r["title"],
                "text": r["selftext"],
                "subreddit": r["subreddit"],
            })
        print(json.dumps(filtered_results, indent=2))
    elif args.url:
        results = r.get_all_comments_of_post(args.url)
        for i, comment in enumerate(results, 1):
            print(f"{i}. {comment}\n")
