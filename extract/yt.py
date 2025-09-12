#!/usr/bin/env python3

import argparse
import isodate
import json
import os
import re
import time
from xml.etree.ElementTree import ParseError
from xml.parsers.expat import ExpatError

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

ENV_PATH = "~/.config/zfabric/.env"


def _retry_operation(func, attempts: int = 5, delay: int = 2):
    for attempt in range(attempts):
        try:
            return func()
        except (ParseError, ExpatError) as e:
            if attempt == attempts - 1:
                raise e
            time.sleep(delay)


def get_video_id(url):
    # Extract video ID from URL
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_comments(youtube, video_id):
    comments = []

    try:
        # Fetch top-level comments
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,  # Adjust based on needs
        )

        while request:
            response = request.execute()
            for item in response["items"]:
                # Top-level comment
                topLevelComment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(topLevelComment)

                # Check if there are replies in the thread
                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        replyText = reply["snippet"]["textDisplay"]
                        # Add incremental spacing and a dash for replies
                        comments.append("    - " + replyText)

            # Prepare the next page of comments, if available
            if "nextPageToken" in response:
                request = youtube.commentThreads().list_next(
                    previous_request=request, previous_response=response
                )
            else:
                request = None

    except HttpError as e:
        print(f"Failed to fetch comments: {e}")

    return comments


def main_function(url, options, return_only=False):
    # Load environment variables from .env file
    load_dotenv(os.path.expanduser(ENV_PATH))

    # Get YouTube API key from environment variable
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in {}".format(ENV_PATH))
        return

    # Extract video ID from URL
    video_id = get_video_id(url)
    if not video_id:
        print("Invalid YouTube URL")
        return

    try:
        # Initialize the YouTube API client
        youtube = build("youtube", "v3", developerKey=api_key)

        # Get video details
        video_response = youtube.videos().list(
            id=video_id, part="contentDetails,snippet").execute()

        # Extract video duration and convert to minutes
        duration_iso = video_response["items"][0]["contentDetails"]["duration"]
        duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
        duration_minutes = round(duration_seconds / 60)

        # Set up metadata
        metadata = {}
        metadata["id"] = video_response["items"][0]["id"]
        metadata["description"] = video_response["items"][0]["snippet"]["description"]
        metadata["title"] = video_response["items"][0]["snippet"]["title"]
        metadata["channel"] = video_response["items"][0]["snippet"]["channelTitle"]
        metadata["published_at"] = video_response["items"][0]["snippet"]["publishedAt"]

        # Get video transcript
        try:
            # This fails randomly ...
            # https://github.com/jdepoix/youtube-transcript-api/issues/429
            # a retry operation is used below as a workaround
            # - this does not work anymore... waiting for update
            # transcript_list = YouTubeTranscriptApi.get_transcript(
            #    video_id, languages=[options.lang]
            # )

            transcript_list = _retry_operation(
                lambda: YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[options.lang])
            )

            # transcript_list[x]["start"] stores the start time in seconds
            transcript_text = " ".join([item["text"]
                                       for item in transcript_list])
            transcript_text = transcript_text.replace("\n", " ")
        except Exception as e:
            transcript_text = f"Transcript not available in the selected language ({options.lang}). ({e})\nThis could also mean that a random error occured with the YouTubeTranscriptAPI, try again and see if it works!"

        # Get comments if the flag is set
        comments = []
        if options.comments:
            comments = get_comments(youtube, video_id)

        output = {}
        if options.duration:
            output["duration"] = duration_minutes
        if options.transcript:
            output["transcript"] = (
                transcript_list if options.transcript_as_list else transcript_text
            )
        if options.comments:
            output["comments"] = comments
        if options.metadata:
            output["metadata"] = metadata

        if return_only:
            return output

        # Output based on options
        if options.duration:
            print(output["duration"])

        if options.transcript:
            if not options.transcript_as_list:
                print(output["transcript"].encode(
                    "utf-8").decode("unicode-escape"))
            else:
                print(json.dumps({"items": output["transcript"]}, indent=2))

        if options.comments:
            print(json.dumps(output["comments"], indent=2))

        if options.metadata:
            print(json.dumps(output["metadata"], indent=2))

    except HttpError as e:
        print(
            f"Error: Failed to access YouTube API. Please check your YOUTUBE_API_KEY and ensure it is valid: {e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--duration", action="store_true",
                        help="Output the duration")
    parser.add_argument("--transcript", action="store_true",
                        help="Output the transcript")
    parser.add_argument("--comments", action="store_true",
                        help="Output the comments")
    parser.add_argument("--metadata", action="store_true",
                        help="Output the video metadata")
    parser.add_argument(
        "--transcript-as-list",
        action="store_true",
        default=False,
        help="Return transcript as a list",
    )
    parser.add_argument(
        "--lang", default="en", help="Language for the transcript (default: English)"
    )

    args = parser.parse_args()

    if args.url is None:
        print("Error: No URL provided.")
        return

    main_function(args.url, args)


if __name__ == "__main__":
    main()
