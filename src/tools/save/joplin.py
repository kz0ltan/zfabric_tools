#!/usr/bin/env python3

# https://joplinapp.org/help/api/references/rest_api#post-notes

import argparse
import datetime
import json
import os
import requests
import sys

from dotenv import load_dotenv

ENV_PATH = "~/.config/zfabric/.env"


def get_api_url():
    api_url = os.getenv("JOPLIN_API_URL", None)
    if not api_url:
        raise ValueError("Missing JOPLIN_API_URL from env file")
    return api_url


def create_note(parent_id: str, title: str, body: str, api_url: str = None):
    if not api_url:
        api_url = get_api_url()
    data = {"parent_id": parent_id, "title": title, "body_html": body}
    response = requests.post(f"{api_url}/notes?token={os.getenv('JOPLIN_TOKEN')}", json=data)
    if response.status_code == 200:
        print("Note created successfully")
    else:
        print(f"Failed to create note: {response.text}")


def list_folders(api_url: str = None):
    if not api_url:
        api_url = get_api_url()
    response = requests.get(f"{api_url}/folders?token={os.getenv('JOPLIN_TOKEN')}")
    if response.status_code == 200:
        folders = json.loads(response.text)
        for folder in folders["items"]:
            print(f"{folder['id']} : {folder['title']}")
    else:
        print(f"Failed to list notes: {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Joplin CLI tool")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--file", type=str, help="Path to input file")
    group.add_argument("-l", "--list", action="store_true", help="List folders with their IDs")

    parser.add_argument("-p", "--parent-id", type=str, required=False, default=None)
    parser.add_argument("-t", "--title", type=str, required=False, default=None)

    args = parser.parse_args()

    load_dotenv(os.path.expanduser(ENV_PATH))

    if args.list:
        list_folders()
    else:
        if args.file:
            try:
                with open(args.file, "r") as file:
                    body = file.read()
            except FileNotFoundError:
                print(f"File not found: {args.file}")
        elif not sys.stdin.isatty():
            body = sys.stdin.read().rstrip()
        else:
            print("No input, exiting")
            sys.exit(0)

        parent_id = args.parent_id or os.getenv("JOPLIN_DEFAULT_FOLDER")
        title = args.title or datetime.datetime.now().strftime("Imported - %y-%m-%d_%H-%M-%S")

        create_note(parent_id, title, body)


if __name__ == "__main__":
    main()
