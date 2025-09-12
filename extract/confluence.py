#!/usr/bin/env python3

import argparse
import json
import os
from typing import Optional, List, Dict, Iterable

from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth


ENV_PATH = "~/.config/zfabric/.env"


class ConfluenceClient:
    """
    Simple client able to authenticate to Confluence using a personal access token and retrieve:
    * pages of a space
    * page contents
    """

    def __init__(self, space_id: Optional[int] = None):
        self.user = str(os.getenv("CONFLUENCE_USER"))
        self.token = str(os.getenv("CONFLUENCE_TOKEN"))

        self.instance_url = str(os.getenv("CONFLUENCE_URL"))
        self.space_id = int(space_id or os.getenv("CONFLUENCE_SPACE"))

        self._auth = False

    @property
    def auth(self) -> HTTPBasicAuth:
        """Returns a BasicAuth object for requests"""
        if not self._auth:
            self._auth = HTTPBasicAuth(self.user, self.token)
        return self._auth

    def _send_request(self, url) -> requests.models.Response:
        headers = {"Accept": "application/json"}
        return requests.request("GET", url, headers=headers, auth=self.auth, timeout=30)

    def _paginated_request(
        self,
        url: str,
        max_pages: int = 1,
        search_field: Optional[str] = None,
        search_value: Optional[str] = None,
    ) -> List:
        page_idx = 0
        objects = []
        while True:
            response = self._send_request(url)
            r_json = response.json()
            objects.extend(r_json["results"])
            print(len(objects), end="\r")
            page_idx += 1
            if search_field and search_value:
                for obj in r_json["results"]:
                    if obj[search_field] == search_value:
                        return [obj]
            if (
                "_links" in r_json
                and "next" in r_json["_links"]
                and page_idx < max_pages
            ):
                url = self.instance_url + r_json["_links"]["next"]
            else:
                print("\n", end="")
                break

        return objects

    def get_spaces(self, space_name: Optional[str] = None) -> List:
        """
        Returns all space's metadata OR
        Return one confluence space's metadata with the name space_name
        """
        url = self.instance_url + "/wiki/api/v2/spaces"
        spaces = self._paginated_request(
            url, max_pages=99999, search_field="name", search_value=space_name
        )
        return spaces

    def get_pages_of_space(
        self, space_id: Optional[int] = None, page_title: Optional[str] = None
    ) -> List:
        """Return all pages' metadata within a space"""
        if not space_id:
            space_id = self.space_id
        url = self.instance_url + f"/wiki/api/v2/spaces/{space_id}/pages"
        pages = self._paginated_request(
            url, max_pages=99999, search_field="title", search_value=page_title
        )
        return pages

    def get_page_by_id(
        self, page_id: int, recursive: bool = False, limit: int = 10
    ) -> List[Dict]:
        """Get page contents based on page_id"""
        url = self.instance_url + f"/wiki/api/v2/pages/{page_id}?body-format=view"
        response = self._send_request(url)

        responses = []
        if recursive:
            children = self.get_child_pages_by_id(page_id)
            child_ids = [list(child.keys())[0] for child in children]
            responses.extend(
                [self.get_page_by_id(child_id)[0] for child_id in child_ids[:limit]]
            )
        else:
            responses.append(response.json())

        return responses

    def get_child_pages_by_id(self, page_id: int) -> Dict:
        """Get child pages based on page_id"""
        url = self.instance_url + f"/wiki/api/v2/pages/{page_id}/children"
        response = self._send_request(url)
        return [{child["id"]: child["title"]} for child in response.json()["results"]]


def print_list(input_iter: Iterable) -> None:
    """Print a list of objects from an iterable"""
    for i in input_iter:
        print(i)


def main(args: argparse.Namespace) -> None:
    """Main function"""
    client = ConfluenceClient()
    if args.children and args.pahe_id:
        children = client.get_child_pages_by_id(args.page_id)
        print_list(children)
    elif args.page_id:
        page = client.get_page_by_id(args.page_id)[0]
        print(json.dumps(page, indent=2))
    elif args.pages:
        pages = client.get_pages_of_space()
        print(json.dumps(pages, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple tool to download information from Confluence"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--page-id",
        type=int,
        default=None,
        help="Page ID of a confluence page",
    )
    group.add_argument(
        "--pages", action="store_true", default=False, help="List pages in the space"
    )
    parser.add_argument(
        "--children",
        required=False,
        default=False,
        action="store_true",
        help="Enumerate children of the given page-id",
    )
    cli_args = parser.parse_args()
    load_dotenv(os.path.expanduser(ENV_PATH))
    main(cli_args)
