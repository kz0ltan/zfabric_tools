import json
import pathlib
import requests
from unittest.mock import MagicMock, patch

import pytest

from extract.web.extractor import WebExtractor


@pytest.fixture
def dummy_html():
    return "<html><body><h1>Test Page</h1></body></html>"


@pytest.fixture
def extractor():
    # Create a WebExtractor with minimal dependencies
    we = WebExtractor(
        lib="preferred",
        default_lib="jina.ai",
        profile=None,
        llm_client=None,
        retriever="requests",
        proxy=None,
        loglevel=20,
    )
    return we


def test_retrieve_requests_success(extractor, dummy_html):
    # Mock requests.get to return a successful response
    mock_response = MagicMock()
    mock_response.text = dummy_html
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = extractor.retrieve("http://example.com", retriever="requests")
        mock_get.assert_called_once()
        assert result == dummy_html


def test_retrieve_jina_api_success(extractor, dummy_html):
    # Mock JinaAI.get_html_content to return dummy HTML
    extractor.jina = MagicMock()
    extractor.jina.get_html_content.return_value = dummy_html

    result = extractor.retrieve("http://example.com", retriever="jina_api")
    extractor.jina.get_html_content.assert_called_once_with(
        "http://example.com", extractor.proxies
    )
    assert result == dummy_html


def test_retrieve_with_fallback_uses_requests_first(extractor, dummy_html):
    # requests succeeds, jina_api should not be called
    mock_response = MagicMock()
    mock_response.text = dummy_html
    mock_response.raise_for_status.return_value = None

    extractor.jina = MagicMock()
    with patch("requests.get", return_value=mock_response):
        result = extractor.retrieve_with_fallback(
            "http://example.com", retrievers=["requests", "jina_api"]
        )
        assert result == dummy_html
        extractor.jina.get_html_content.assert_not_called()


def test_retrieve_with_fallback_falls_back_to_jina(extractor, dummy_html):
    # requests raises, jina_api succeeds
    extractor.jina = MagicMock()
    extractor.jina.get_html_content.return_value = dummy_html

    with patch(
        "requests.get", side_effect=requests.exceptions.RequestException("fail")
    ):
        result = extractor.retrieve_with_fallback(
            "http://example.com", retrievers=["requests", "jina_api"]
        )
        assert result == dummy_html
        extractor.jina.get_html_content.assert_called_once()


def test_extract_jina_ai_path(extractor, dummy_html):
    # Mock the JinaAI methods used in extract
    extractor.jina = MagicMock()
    extractor.jina.get_markdown_content.return_value = "markdown content"
    extractor.jina.strip_markdown.return_value = "clean markdown"
    extractor._profile = {"type": "openai"}  # triggers strip_markdown

    # Mock retrieve_with_fallback to return dummy_html
    extractor.retrieve_with_fallback = MagicMock(return_value=dummy_html)

    html, content = extractor.extract(
        "http://example.com", lib="jina.ai", json=False, retriever=["requests"]
    )
    extractor.jina.get_markdown_content.assert_called_once()
    extractor.jina.strip_markdown.assert_called_once()
    assert html == dummy_html
    assert content == "clean markdown"
