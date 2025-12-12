#!/usr/bin/env python3
"""
Download files from a GitHub repository using either API or raw URL method.
Defaults to README.md if no specific file is provided.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    https_pattern = r"github\.com/([^/]+)/([^/]+)(?:/(?:blob|tree)/([^/]+)/(.+))?"
    # Pattern for SSH URLs
    ssh_pattern = r"github\.com:([^/]+)/([^/]+?)(?:\.git)?$"

    match = re.search(https_pattern, url)
    if match:
        owner = match.group(1)
        repo = match.group(2).rstrip(".git")
        branch = match.group(3)  # Could be None
        file_path = match.group(4)  # Could be None
        return owner, repo, file_path, branch

    match = re.search(ssh_pattern, url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
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
                return True

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
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Parse GitHub URL
        owner, repo, url_file_path, url_branch = parse_github_url(args.url)

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
