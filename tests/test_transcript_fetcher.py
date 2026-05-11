import requests

import pytest

from earningslens import config
from earningslens.transcript_fetcher import TranscriptFetchError, _get


def test_get_wraps_request_exception(monkeypatch):
    monkeypatch.setattr(config, "ALPHA_VANTAGE_API_KEY", "demo")

    def fake_get(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("earningslens.transcript_fetcher.requests.get", fake_get)

    with pytest.raises(TranscriptFetchError, match="Transcript request failed"):
        _get({"function": "SYMBOL_SEARCH", "keywords": "MSFT"})
