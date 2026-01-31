#!/usr/bin/env python3

"""Download public/user information from Reddit"""

import datetime
import os
import sys
from typing import Optional

# from dotenv import load_dotenv
import praw

__all__ = ["Reddit"]

# ENV_PATH = "~/.config/zfabric/.env"


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
                saved_item.update(
                    {
                        "title": item.title,
                        "author": item.author.name if item.author else "[deleted]",
                        "selftext": item.selftext if hasattr(item, "selftext") else "",
                        "is_self": item.is_self if hasattr(item, "is_self") else None,
                        "score": item.score,
                    }
                )
            elif item_type == "comment":
                saved_item.update(
                    {
                        "body": item.body if hasattr(item, "body") else "",
                        "author": item.author.name if item.author else "[deleted]",
                        "parent_id": item.parent_id if hasattr(item, "parent_id") else None,
                        "score": item.score,
                    }
                )
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

            ndata.append(
                {
                    "url": url,
                    "date": date,
                    "title": title,
                    "summary": text,
                    "tags": tags,
                    "source": "reddit",
                }
            )

        return ndata

    def get_submission(self, post_url: str, limit=200):
        submission = self.reddit.submission(url=post_url)
        submission.comments.replace_more(limit=limit)  # Load nested comments
        return submission

    def print_post(self, submission):
        print("# " + submission.title + "\n")
        print(submission.selftext)
        print("-" * 80)

    def print_comment_tree(self, comment_forest, indent=0, max_len: int = 80):
        """Recursively prints comments with indentation."""
        for comment in comment_forest:
            # `comment` can be a Comment or a MoreComments (should be gone after replace_more)
            if isinstance(comment, praw.models.Comment):
                # Clean up newlines & tabs for nicer output
                body = comment.body.replace("\n", " ").replace("\r", "")
                print(" " * indent + f"u/{comment.author} [{comment.score}] {body[:max_len]}...")
                # Recurse into replies (which is another CommentForest)
                if comment.replies:
                    self.print_comment_tree(comment.replies, indent + 4)
            else:
                # Should not happen after replace_more, but keep for safety
                print(" " * indent + f"<{type(comment).__name__}>")

    def build_post_dict(self, s):
        """Collect the fields you care about from a Submission."""
        post = {
            "id": s.id,
            "title": s.title,
            "subreddit": s.subreddit.display_name,
            "author": str(s.author) if s.author else "[deleted]",
            "score": s.score,
            "upvote_ratio": s.upvote_ratio,
            "created_utc": s.created_utc,
            "over_18": s.over_18,
            "url": s.url,
            "is_self": s.is_self,
            "permalink": s.permalink,
            "selftext": s.selftext if s.is_self else None,
            "preview_images": None,
            "media": None,
        }

        # --- optional media handling -------------------------------------------------
        # if hasattr(s, "preview") and s.preview:
        #    # Grab the first preview image (if any)
        #    images = s.preview.get("images")
        #    if images:
        #        post["preview_images"] = [img["source"]["url"] for img in images]

        # if s.media:
        #    post["media"] = s.media  # raw dict – you can parse further if needed

        # For gallery posts (multiple images) – works on newer Reddit designs
        # if getattr(s, "is_gallery", False):
        #    post["gallery"] = {}
        #    for media_id, meta in s.media_metadata.items():
        #        # Most common: images
        #        if meta.get("e") == "Image":
        #            post["gallery"][media_id] = meta["s"]["u"]  # direct URL
        # ------------------------------------------------------------------------------

        return post

    def comment_forest_to_dict(self, comment_forest):
        """
        Turn a CommentForest into a list of dicts:
        [
            {
                "id": "...",
                "author": "...",
                "score": ...,
                "body": "...",
                "created_utc": ...,
                "replies": [ ...same structure... ]
            },
            ...
        ]
        """
        result = []
        for comment in comment_forest:
            if not isinstance(comment, praw.models.Comment):
                # Skip any stray MoreComments (should be none after replace_more)
                continue

            comment_dict = {
                "id": comment.id,
                "author": str(comment.author) if comment.author else "[deleted]",
                "score": comment.score,
                "body": comment.body,
                "created_utc": comment.created_utc,
                "permalink": comment.permalink,
                "replies": self.comment_forest_to_dict(comment.replies) if comment.replies else [],
            }
            result.append(comment_dict)

        return result

    def __call__(self):
        return self.get_saved_posts(self.reddit, limit=None)


# def parse_parameters():
#    parser = argparse.ArgumentParser(description="Download saved Reddit posts and comments")
#    parser.add_argument(
#        "--client_id", type=str, default=None, required=False, help="Reddit API client ID"
#    )
#    parser.add_argument(
#        "--client_secret", type=str, default=None, required=False, help="Reddit API client secret"
#    )
#    parser.add_argument(
#        "--username", type=str, default=None, required=False, help="Reddit username"
#    )
#    parser.add_argument(
#        "--password", type=str, default=None, required=False, help="Reddit password"
#    )
#    parser.add_argument(
#        "--user_agent",
#        type=str,
#        required=False,
#        default="zFabric Reddit client",
#        help="User agent for Reddit API requests",
#    )
#    group = parser.add_mutually_exclusive_group(required=True)
#    group.add_argument(
#        "--url", type=str, required=False, help="Reddit URL to download submission/comments"
#    )
#    group.add_argument(
#        "--saved",
#        required=False,
#        action="store_true",
#        help="Download user's saved posts/comments",
#    )
#
#    return parser.parse_args()
#
#
# if __name__ == "__main__":
#    args = parse_parameters()
#    load_dotenv(os.path.expanduser(ENV_PATH))
#    r = Reddit(
#        client_id=args.client_id,
#        client_secret=args.client_secret,
#        user_agent=args.user_agent,
#        username=args.username,
#        password=args.password,
#    )
#    if args.saved:
#        results = r.get_saved_posts()
#        filtered_results = []
#        for r in results:
#            if r["type"] == "comment":
#                continue
#            filtered_results.append({
#                "title": r["title"],
#                "text": r["selftext"],
#                "subreddit": r["subreddit"],
#            })
#        print(json.dumps(filtered_results, indent=2))
#    elif args.url:
#        submission = r.get_submission(args.url)
#        r.print_post(submission)
#        r.print_comment_tree(submission.comments)
