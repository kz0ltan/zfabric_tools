import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from extract.web.exceptions import FailedRetrievalError, RetrievalError
from extract.web.extractor import WebExtractor


@pytest.fixture
def dummy_html():
    return "<html><body><h1>Test Page</h1></body></html>"


@pytest.fixture
def extractor():
    # Create a WebExtractor with minimal dependencies
    return WebExtractor()


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
    extractor.jina.get_html_content.assert_called_once_with("http://example.com", extractor.proxies)
    assert result == dummy_html


def test_extract_jina_ai_path(extractor, dummy_html):
    # Mock the JinaAI methods used in extract
    extractor.jina = MagicMock()
    extractor.jina.get_markdown_content.return_value = {"content": "markdown content"}
    extractor.jina.strip_markdown.return_value = "clean markdown"
    extractor._jina_profile = {"type": "openai"}  # triggers strip_markdown

    # Mock retrieve_with_fallback to return dummy_html
    extractor.retrieve_with_fallback = MagicMock(return_value=[{"html_content": dummy_html}])

    html, content = extractor.extract(
        "http://example.com", extractor="jina.ai", json=False, retrievers=["requests"]
    )
    extractor.jina.get_markdown_content.assert_called_once()
    extractor.jina.strip_markdown.assert_called_once()
    assert html == dummy_html
    assert content == "clean markdown"


def test_retrieve_with_fallback_all_fail(extractor):
    """All retrievers raise an exception – should raise RetrievalError."""
    # requests fails
    with patch("requests.get", side_effect=requests.exceptions.RequestException("fail")):
        # jina_api (or any other retriever) also fails
        extractor.jina = MagicMock()
        extractor.jina.get_html_content.side_effect = Exception("fail")
        with pytest.raises(FailedRetrievalError):
            extractor.retrieve_with_fallback(
                "http://example.com", retrievers=["requests", "jina_api"]
            )


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
        assert result[0]["html_content"] == dummy_html
        extractor.jina.get_html_content.assert_not_called()


def test_retrieve_with_fallback_falls_back_to_jina(extractor, dummy_html):
    # requests raises, jina_api succeeds
    extractor.jina = MagicMock()
    extractor.jina.get_html_content.return_value = dummy_html

    with patch("requests.get", side_effect=requests.exceptions.RequestException("fail")):
        result = extractor.retrieve_with_fallback(
            "http://example.com", retrievers=["requests", "jina_api"]
        )
        assert result[1]["html_content"] == dummy_html
        extractor.jina.get_html_content.assert_called_once()


# ----------------------------------------------------------------------
# Additional tests to increase coverage
# ----------------------------------------------------------------------


def test_get_preferred_extractor_and_retriever():
    DOMAIN_SETTINGS = {
        "www.testdomain.com": {
            "retriever": "jina_api",
            "extractor": "jina.ai",
        }
    }
    we = WebExtractor()
    we.DOMAIN_SETTINGS = DOMAIN_SETTINGS
    url = "https://www.testdomain.com/some-article"
    assert we._get_preferred(url, "extractor") == "jina.ai"
    assert we._get_preferred(url, "retriever") == "jina_api"


def test_get_preferred_fallback():
    we = WebExtractor()
    url = "https://unknown.example.com"
    assert we._get_preferred(url, "extractor") == we.fallback_extractor
    assert we._get_preferred(url, "retriever") == we.fallback_retriever


def test_profile_string_loads_via_jina(monkeypatch):
    we = WebExtractor(jina_profile="openai")
    dummy = {"type": "openai", "base_url": None, "api_key": None, "model": None}
    monkeypatch.setattr(we.jina, "get_profile_from_env", lambda name: dummy)

    # First access triggers JinaAI.get_profile
    assert we.jina_profile == dummy
    # Second access should use cached value
    with monkeypatch.context() as m:
    dummy = {"type": "openai", "base_url": None,
             "api_key": None, "model": None}
    monkeypatch.setattr(we.jina, "get_profile_from_env", lambda name: dummy)

    # First access triggers JinaAI.get_profile
    assert we.jina_profile == dummy
    # Second access should use cached value
    with monkeypatch.context() as m:
        m.setattr(we.jina, "get_profile_from_env",
                  lambda _: pytest.fail("should not be called"))
        assert we.jina_profile == dummy


def test_profile_dict_is_used_directly():
    we = WebExtractor(jina_profile={"type": "openai"})
    assert we.jina_profile == {"type": "openai"}


def test_select_extractor_preferred():
    DOMAIN_SETTINGS = {
        "www.testdomain.com": {
            "retriever": "jina_api",
        }
    }
    we = WebExtractor()
    we.DOMAIN_SETTINGS = DOMAIN_SETTINGS
    url = "https://www.testdomain.com/article"
    assert we._select_extractor("preferred", url) == we.fallback_extractor


def test_select_extractor_explicit():
    we = WebExtractor()
    assert we._select_extractor("trafilatura", "https://any") == "trafilatura"


def test_select_extractor_none_uses_instance_default():
    we = WebExtractor(extractor="jina.ai")
    assert we._select_extractor(None, "https://any") == "jina.ai"


def test_select_retrievers_empty_uses_instance_default():
    we = WebExtractor(retrievers=["requests", "jina_api"])
    assert we._select_retrievers([], "https://any") == ["requests", "jina_api"]


def test_select_retrievers_preferred_keyword():
    DOMAIN_SETTINGS = {
        "www.testdomain.com": {
            "retriever": "jina_api",
        }
    }
    we = WebExtractor()
    we.DOMAIN_SETTINGS = DOMAIN_SETTINGS
    url = "https://www.testdomain.com"
    assert we._select_retrievers(["preferred"], url) == ["jina_api"]


def test_retrieve_requests_raises_retrieval_error():
    we = WebExtractor()
    with patch("requests.get", side_effect=requests.exceptions.RequestException("boom")):
        with pytest.raises(RetrievalError) as exc:
            we.retrieve("http://example.com", retriever="requests")
        assert "Error during HTML retrieveal" in str(exc.value)


def test_retrieve_playwright_missing_module(monkeypatch):
    we = WebExtractor()
    # Simulate playwright not being installed
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(ModuleNotFoundError):
        we.retrieve("http://example.com", retriever="playwright")


def test_extract_unknown_extractor_raises():
    we = WebExtractor()
    we.retrieve_with_fallback = MagicMock(
        return_value=[{"html_content": "<html></html>"}])
    with pytest.raises(ValueError, match="Unknown extraction library"):
        we.extract("http://example.com", extractor="nonexistent")


def test_extract_newspaper4k(monkeypatch):
    we = WebExtractor()
    dummy_html = [{"html_content": "<html><body>...</body></html>"}]
    dummy_text = "Article body"
    we.retrieve_with_fallback = MagicMock(return_value=dummy_html)

    class DummyArticle:
        def __init__(self, url): pass
        def set_html(self, html): pass
        def parse(self): pass
        @ property
        def text(self): return dummy_text

    monkeypatch.setitem(sys.modules, "newspaper",
                        MagicMock(Article=DummyArticle))

    html, text = we.extract("http://example.com", extractor="newspaper4k")
    assert html == dummy_html[0]["html_content"]
    assert text == dummy_text


def test_extract_trafilatura_markdown(monkeypatch):
    we = WebExtractor()
    dummy_html = [{"html_content": "<html>...</html>"}]
    we.retrieve_with_fallback = MagicMock(return_value=dummy_html)

    dummy_md = "## Title\\nContent"
    monkeypatch.setitem(sys.modules, "trafilatura",
                        MagicMock(extract=lambda *a, **k: dummy_md))

    html, md = we.extract("http://example.com",
                          extractor="trafilatura", json=False)
    assert md == dummy_md


def test_extract_trafilatura_json(monkeypatch):
    we = WebExtractor()
    dummy_html = [{"html_content": "<html>...</html>"}]
    we.retrieve_with_fallback = MagicMock(return_value=dummy_html)

    dummy_json = {"title": "Title", "content": "Content"}
    import json as _json
    monkeypatch.setitem(sys.modules, "trafilatura", MagicMock(
        extract=lambda *a, **k: _json.dumps(dummy_json) if k.get(
            "output_format") == "json" else None
    ))

    html, data = we.extract("http://example.com",
                            extractor="trafilatura", json=True)
    assert data == dummy_json


def test_extract_jina_ai_json(monkeypatch):
    we = WebExtractor(jina_profile={"type": "openai"})
    dummy_html = [{"html_content": "<html></html>"}]
    we.retrieve_with_fallback = MagicMock(return_value=dummy_html)

    json_str = json.dumps({"summary": "test"})
    markdown_str = f"```markdown\n{json_str}\n```"
    we.jina.get_markdown_content.return_value = {"content": markdown_str}
    we.jina.strip_markdown.return_value = json_str

    html, data = we.extract("http://example.com", extractor="jina.ai", json=True)
    we.jina.get_markdown_content.assert_called_once()
    assert data == {"summary": "test"}


def test_extract_uses_fallback_extractor():
    we = WebExtractor(fallback_extractor="trafilatura")
    we.retrieve_with_fallback = MagicMock(return_value=[{"html_content": "<html>asdf</html>"}])
    we.jina = MagicMock()
    we.jina.get_markdown_content.return_value = "md"

    # Domain not in preferences → fallback_extractor should be used
    html, content = we.extract("https://unknown.example.com", extractor="preferred")
    # The trafilatura branch will be exercised; we mock it to return simple text
    we.jina.get_markdown_content.assert_not_called()


def test_retrieve_with_fallback_empty_list_uses_instance_default(monkeypatch):
    we = WebExtractor(retrievers=["requests"])
    dummy_html = "<html></html>"
    called = []

    def fake_retrieve(url, retriever):
        called.append(retriever)
        return dummy_html

    monkeypatch.setattr(we, "retrieve", fake_retrieve)

    result = we.retrieve_with_fallback("http://example.com", retrievers=[])
    result = result[0]["html_content"]
    assert result == dummy_html
    assert called == ["requests"]


def test_proxies_property():
    we = WebExtractor()
    assert we.proxies is None

    we = WebExtractor(proxy="http://proxy:8080")
    assert we.proxies == {"http": "http://proxy:8080", "https": "http://proxy:8080"}
