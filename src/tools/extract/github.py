#!/usr/bin/env python3
"""
Download files from a GitHub repository using either API or raw URL method.
Defaults to README.md if no specific file is provided.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from web.utils import set_up_logging


def is_github_repo_url(owner: str, repo: str, token=None):
    """
    Validates if the given URL points to a valid GitHub repository.

    This function parses the owner and repository name from the URL and
    uses the GitHub API to check if the repository exists.

    Args:
        url (str): The GitHub URL to validate.
        token (str, optional): A GitHub API token for authentication.

    Returns:
        bool: True if the URL points to a valid repository, False otherwise.
    """

    # We only care about the owner and repo name for this check
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-GitHub-File-Downloader",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        # We just need the response headers to check the status code
        request = Request(api_url, headers=headers, method="HEAD")
        with urlopen(request, timeout=10) as response:
            # A 200 OK status code means the repo exists
            return response.status == 200
    except HTTPError as e:
        # 404 Not Found means the repo doesn't exist
        if e.code == 404:
            logging.debug(f"Repository {owner}/{repo} not found (404).")
        else:
            logging.debug(f"API check for {owner}/{repo} failed with HTTP error {e.code}.")
        return False
    except URLError as e:
        logging.debug(f"Network error during API check for {owner}/{repo}: {e.reason}")
        return False


def get_default_branch(owner: str, repo: str, token=None):
    """
    Get the default branch of a GitHub repository.

    Args:
        owner (str): Repository owner
        repo (str): Repository name
        token (str, optional): GitHub API token

    Returns:
        str: Default branch name or None if not found
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-GitHub-File-Downloader",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        request = Request(api_url, headers=headers)
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("default_branch")
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        logging.debug(f"Failed to get default branch: {e}")
        return None


def get_repo_branches(owner: str, repo: str, token=None):
    """
    Get all branches of a GitHub repository.

    Args:
        owner (str): Repository owner
        repo (str): Repository name
        token (str, optional): GitHub API token

    Returns:
        list: List of branch names, empty list if failed
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/branches"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-GitHub-File-Downloader",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        request = Request(api_url, headers=headers)
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            return [branch["name"] for branch in data]
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        logging.debug(f"Failed to get repository branches: {e}")
        return []


def parse_github_url(url):
    """
    Parse a GitHub URL and extract owner, repo name, and file path if present.

    Supports formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo/blob/branch/path/to/file.txt
    - https://github.com/owner/repo/tree/branch/path/to/dir
    - git@github.com:owner/repo.git

    Returns:
        tuple: (owner, repo, file_path, branch)
    """
    # Remove trailing slashes
    url = url.rstrip("/")

    # Pattern for HTTPS URLs with potential file path
    https_pattern = r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)(?:/(?:blob|tree)/(?P<branch>[^/]+)/(?P<path>.+))?"
    # Pattern for SSH URLs
    ssh_pattern = r"github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$"

    match = re.search(https_pattern, url)
    if match:
        owner = match.group("owner")
        repo = match.group("repo")
        branch = match.group("branch")
        file_path = match.group("path")
        return owner, repo, file_path, branch

    match = re.search(ssh_pattern, url)
    if match:
        owner = match.group("owner")
        repo = match.group("repo")
        return owner, repo, None, None

    raise ValueError(f"Invalid GitHub URL: {url}")


def download_via_api(owner, repo, file_path="README.md", token=None, branch=None):
    """
    Download a file using GitHub API.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"

    # Try specified branch first, then main/master fallback
    branches_to_try = [branch] if branch else ["main", "master"]

    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "Python-GitHub-File-Downloader",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    last_error = None
    for try_branch in branches_to_try:
        try:
            branch_url = f"{url}?ref={try_branch}"
            request = Request(branch_url, headers=headers)

            logging.info(f"Attempting to download '{file_path}' via API (branch: {try_branch})...")
            with urlopen(request, timeout=10) as response:
                content = response.read()

                logging.info(f"Successfully downloaded '{file_path}' via API")
                logging.info(f"Size: {len(content)} bytes")
                return content

        except HTTPError as e:
            last_error = e
            if e.code == 404 and try_branch != branches_to_try[-1]:
                logging.debug(f"Branch '{try_branch}' not found, trying next...")
                continue  # Try next branch
            elif e.code == 404:
                logging.error(f"File '{file_path}' not found in repository")
            elif e.code == 403:
                logging.error("Rate limit exceeded or access denied. Consider using --token")
            else:
                logging.error(f"HTTP Error {e.code}: {e.reason}")
        except URLError as e:
            last_error = e
            logging.error(f"Network error: {e.reason}")

    # If we've tried all branches and still haven't found the file,
    # try to get the default branch from the repo info
    if not branch and last_error and last_error.code == 404:
        logging.info("Standard branches not found, checking repository for default branch...")
        default_branch = get_default_branch(owner, repo, token)

        if default_branch and default_branch not in ["main", "master"]:
            logging.info(f"Found default branch: {default_branch}")
            try:
                branch_url = f"{url}?ref={default_branch}"
                request = Request(branch_url, headers=headers)

                logging.info(
                    f"Attempting to download '{file_path}' via API (branch: {default_branch})..."
                )
                with urlopen(request, timeout=10) as response:
                    content = response.read()

                    logging.info(f"Successfully downloaded '{file_path}' via API")
                    logging.info(f"Size: {len(content)} bytes")
                    return content
            except (HTTPError, URLError) as e:
                logging.debug(f"Failed to download with default branch: {e}")

        # If default branch doesn't work, try to get all branches and use the first one
        logging.info("Default branch failed, getting all available branches...")
        branches = get_repo_branches(owner, repo, token)

        if branches:
            # Filter out branches we've already tried
            remaining_branches = [
                b for b in branches if b not in ["main", "master", default_branch]
            ]

            if remaining_branches:
                first_branch = remaining_branches[0]
                logging.info(f"Trying first available branch: {first_branch}")
                try:
                    branch_url = f"{url}?ref={first_branch}"
                    request = Request(branch_url, headers=headers)

                    logging.info(
                        f"Attempting to download '{file_path}' via API (branch: {first_branch})..."
                    )
                    with urlopen(request, timeout=10) as response:
                        content = response.read()

                        logging.info(f"Successfully downloaded '{file_path}' via API")
                        logging.info(f"Size: {len(content)} bytes")
                        return content
                except (HTTPError, URLError) as e:
                    logging.debug(f"Failed to download with branch {first_branch}: {e}")

    if last_error:
        raise last_error


def download_via_raw(owner, repo, file_path="README.md", branch=None):
    """
    Download a file using raw GitHub URL.
    """
    # Try specified branch first, then main/master fallback
    branches_to_try = [branch] if branch else ["main", "master"]

    headers = {"User-Agent": "Python-GitHub-File-Downloader"}

    last_error = None
    for try_branch in branches_to_try:
        try:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{try_branch}/{file_path}"
            request = Request(url, headers=headers)

            logging.info(
                f"Attempting to download '{file_path}' via raw URL (branch: {try_branch})..."
            )
            with urlopen(request, timeout=10) as response:
                content = response.read()

                logging.info(f"Successfully downloaded '{file_path}' via raw URL")
                logging.info(f"Size: {len(content)} bytes")
                return content

        except HTTPError as e:
            last_error = e
            if e.code == 404 and try_branch != branches_to_try[-1]:
                logging.debug(f"Branch '{try_branch}' not found, trying next...")
                continue  # Try next branch
            elif e.code == 404:
                logging.error(f"File '{file_path}' not found in repository")
            else:
                logging.error(f"HTTP Error {e.code}: {e.reason}")
        except URLError as e:
            last_error = e
            logging.error(f"Network error: {e.reason}")

    # If we've tried all branches and still haven't found the file,
    # try to get the default branch from the repo info
    if not branch and last_error and last_error.code == 404:
        logging.info("Standard branches not found, checking repository for default branch...")
        default_branch = get_default_branch(owner, repo)

        if default_branch and default_branch not in ["main", "master"]:
            logging.info(f"Found default branch: {default_branch}")
            try:
                url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{file_path}"
                )
                request = Request(url, headers=headers)

                logging.info(
                    f"Attempting to download '{file_path}' via raw URL (branch: {default_branch})..."
                )
                with urlopen(request, timeout=10) as response:
                    content = response.read()

                    logging.info(f"Successfully downloaded '{file_path}' via raw URL")
                    logging.info(f"Size: {len(content)} bytes")
                    return content
            except (HTTPError, URLError) as e:
                logging.debug(f"Failed to download with default branch: {e}")

        # If default branch doesn't work, try to get all branches and use the first one
        logging.info("Default branch failed, getting all available branches...")
        branches = get_repo_branches(owner, repo)

        if branches:
            # Filter out branches we've already tried
            remaining_branches = [
                b for b in branches if b not in ["main", "master", default_branch]
            ]

            if remaining_branches:
                first_branch = remaining_branches[0]
                logging.info(f"Trying first available branch: {first_branch}")
                try:
                    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{first_branch}/{file_path}"
                    request = Request(url, headers=headers)

                    logging.info(
                        f"Attempting to download '{file_path}' via raw URL (branch: {first_branch})..."
                    )
                    with urlopen(request, timeout=10) as response:
                        content = response.read()

                        logging.info(f"Successfully downloaded '{file_path}' via raw URL")
                        logging.info(f"Size: {len(content)} bytes")
                        return content
                except (HTTPError, URLError) as e:
                    logging.debug(f"Failed to download with branch {first_branch}: {e}")

    if last_error:
        raise last_error


def get_file(url: str, method: str = "api", token: str = None):
    owner, repo, url_file_path, url_branch = parse_github_url(url)
    file_path = url_file_path or "README.md"

    # Determine branch: CLI flag takes precedence, then URL, then None (will auto-detect)
    branch = url_branch

    logging.info(f"Repository: {owner}/{repo}")
    logging.info(f"File: {file_path}")
    if branch:
        logging.info(f"Branch: {branch}")

    # Download using selected method
    if method == "api":
        return download_via_api(owner, repo, file_path, token, branch)
    else:
        return download_via_raw(owner, repo, file_path, branch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download files from a GitHub repository (defaults to README.md)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download README.md from a repo
  %(prog)s https://github.com/facebook/react

  # Download a specific file from URL
  %(prog)s https://github.com/python/cpython/blob/main/setup.py

  # Download a specific file with --file flag
  %(prog)s https://github.com/torvalds/linux --file kernel/sched.c

  # Use raw method and specify branch
  %(prog)s https://github.com/microsoft/vscode --file package.json --method raw --branch main

  # Custom output filename
  %(prog)s https://github.com/owner/repo --file src/config.yaml --output my-config.yaml

  # With authentication token
  %(prog)s https://github.com/owner/private-repo --file secret.txt --token YOUR_TOKEN
        """,
    )

    parser.add_argument("url", help="GitHub repository URL or direct file URL")

    parser.add_argument(
        "--file",
        "-f",
        help="Specific file path to download (e.g., src/main.py). Defaults to README.md",
    )

    parser.add_argument(
        "--method",
        choices=["api", "raw"],
        default="api",
        help="Download method: api (default) or raw URL",
    )

    parser.add_argument(
        "--token", help="GitHub API token (for higher rate limits and private repos)"
    )

    parser.add_argument(
        "--branch",
        help="Branch name (will auto-detect from URL or try main/master if not specified)",
    )

    parser.add_argument(
        "--output", "-o", help="Output filename (defaults to the original filename)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        set_up_logging(logging.DEBUG)

    try:
        # Parse GitHub URL
        owner, repo, url_file_path, url_branch = parse_github_url(args.url)

        # usr the api to check if this is a repo
        if args.method == "api" and not is_github_repo_url(owner, repo, args.token):
            logging.error("This is not a github repo URL")
            sys.exit(1)

        # Determine file path: CLI flag takes precedence, then URL, then default to README.md
        file_path = args.file or url_file_path or "README.md"

        # Determine branch: CLI flag takes precedence, then URL, then None (will auto-detect)
        branch = args.branch or url_branch

        logging.info(f"Repository: {owner}/{repo}")
        logging.info(f"File: {file_path}")
        if branch:
            logging.info(f"Branch: {branch}")

        # Download using selected method
        if args.method == "api":
            result = download_via_api(owner, repo, file_path, args.token, branch)
        else:
            result = download_via_raw(owner, repo, file_path, branch)

        if args.output is None:
            output = Path(file_path).name

        with open(output, "wb") as f:
            f.write(result)

        return 0

    except ValueError as e:
        logging.error(f"{e}")
        return 1
    except Exception as e:
        logging.error(f"Failed to download file: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
