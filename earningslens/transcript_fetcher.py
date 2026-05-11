from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import threading
import time

import requests

from earningslens import config


class TranscriptFetchError(RuntimeError):
    """Raised when transcript retrieval fails."""


@dataclass(frozen=True)
class SearchResult:
    symbol: str
    name: str
    region: str
    currency: str
    match_score: float

    @property
    def label(self) -> str:
        return f"{self.name} ({self.symbol}) - {self.region}"


@dataclass(frozen=True)
class RetrievedTranscript:
    company: str
    symbol: str
    quarter_label: str
    transcript_text: str
    transcript_quarter: str


@dataclass(frozen=True)
class QuarterOption:
    code: str
    label: str


_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL_SECONDS = 1.1
_MAX_RATE_LIMIT_RETRIES = 3


def _require_api_key() -> str:
    if not config.ALPHA_VANTAGE_API_KEY:
        raise TranscriptFetchError("ALPHA_VANTAGE_API_KEY is missing. Add it to .env to use ticker search.")
    return config.ALPHA_VANTAGE_API_KEY


def _wait_for_rate_limit_window() -> None:
    global _LAST_REQUEST_AT
    with _REQUEST_LOCK:
        now = time.monotonic()
        wait_seconds = _MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_REQUEST_AT = time.monotonic()


def _format_alpha_vantage_limit_message(note: str) -> str:
    lowered = note.lower()
    if "25 requests per day" in lowered:
        return (
            "Alpha Vantage free-tier quota was hit. Their free key allows 25 requests per day. "
            "Wait for the quota window to reset or use a premium key."
        )
    return (
        "Alpha Vantage rate limit was hit. EarningsLens spaces requests automatically, "
        "but the free tier can still reject bursts. Wait a few seconds and try again."
    )


def _get(params: dict) -> dict:
    api_key = _require_api_key()
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        _wait_for_rate_limit_window()
        response = requests.get(
            config.ALPHA_VANTAGE_BASE_URL,
            params={**params, "apikey": api_key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if "Information" in payload:
            raise TranscriptFetchError(payload["Information"])
        if "Note" in payload:
            if attempt >= _MAX_RATE_LIMIT_RETRIES:
                raise TranscriptFetchError(_format_alpha_vantage_limit_message(payload["Note"]))
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS * (attempt + 1))
            continue
        if "Error Message" in payload:
            raise TranscriptFetchError(payload["Error Message"])
        return payload
    raise TranscriptFetchError("Alpha Vantage request failed after repeated rate-limit retries.")


def search_companies(query: str, limit: int = 8) -> list[SearchResult]:
    normalized_query = query.strip().upper()
    payload = _get({"function": "SYMBOL_SEARCH", "keywords": normalized_query})
    matches = payload.get("bestMatches", [])
    results = [
        SearchResult(
            symbol=match["1. symbol"],
            name=match["2. name"],
            region=match.get("4. region", ""),
            currency=match.get("8. currency", ""),
            match_score=float(match.get("9. matchScore", 0.0)),
        )
        for match in matches
        if str(match.get("1. symbol", "")).strip().upper() == normalized_query
    ]
    results.sort(key=lambda item: item.match_score, reverse=True)
    return results[:limit]


def _quarter_code_from_date(date_str: str) -> str:
    fiscal_date = datetime.strptime(date_str, "%Y-%m-%d")
    quarter = ((fiscal_date.month - 1) // 3) + 1
    return f"{fiscal_date.year}Q{quarter}"


def _quarter_label_from_code(code: str) -> str:
    year = code[:4]
    quarter = code[-2:]
    return f"{quarter} {year}"


def _recent_quarter_codes(symbol: str, max_candidates: int = 8) -> list[str]:
    payload = _get({"function": "EARNINGS", "symbol": symbol})
    quarterly = payload.get("quarterlyEarnings", [])
    codes: list[str] = []
    for item in quarterly:
        fiscal_date = item.get("fiscalDateEnding")
        if not fiscal_date:
            continue
        code = _quarter_code_from_date(fiscal_date)
        if code not in codes:
            codes.append(code)
        if len(codes) >= max_candidates:
            break
    if len(codes) < 2:
        raise TranscriptFetchError(f"Could not determine at least two recent quarters for {symbol}.")
    return codes


def list_recent_quarters(symbol: str, max_candidates: int = 8) -> list[QuarterOption]:
    return [QuarterOption(code=code, label=_quarter_label_from_code(code)) for code in _recent_quarter_codes(symbol, max_candidates)]


def _extract_transcript_text(payload: dict) -> str:
    if isinstance(payload, dict):
        if "transcript" in payload and isinstance(payload["transcript"], str):
            return payload["transcript"].strip()
        if "transcript" in payload and isinstance(payload["transcript"], list):
            lines: list[str] = []
            for item in payload["transcript"]:
                if not isinstance(item, dict):
                    continue
                speaker = str(item.get("speaker", "")).strip() or "Unknown"
                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                label = f"{speaker} - {title}" if title else speaker
                lines.append(f"{label}: {content}")
            return "\n\n".join(lines).strip()
        if "content" in payload and isinstance(payload["content"], str):
            return payload["content"].strip()
    raise TranscriptFetchError("Transcript payload did not contain transcript text.")


def _fetch_transcript(symbol: str, quarter_code: str, company_name: str) -> RetrievedTranscript:
    payload = _get({"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": symbol, "quarter": quarter_code})
    transcript_text = _extract_transcript_text(payload)
    return RetrievedTranscript(
        company=company_name,
        symbol=symbol,
        quarter_label=_quarter_label_from_code(quarter_code),
        transcript_text=transcript_text,
        transcript_quarter=quarter_code,
    )


def fetch_transcript_pair(
    symbol: str,
    company_name: str,
    current_quarter_code: str,
    prior_quarter_code: str,
) -> tuple[RetrievedTranscript, RetrievedTranscript]:
    if current_quarter_code == prior_quarter_code:
        raise TranscriptFetchError("Choose two different quarters.")
    current = _fetch_transcript(symbol, current_quarter_code, company_name)
    prior = _fetch_transcript(symbol, prior_quarter_code, company_name)
    return current, prior


def fetch_latest_two_transcripts(symbol: str, company_name: str) -> tuple[RetrievedTranscript, RetrievedTranscript]:
    quarter_codes = _recent_quarter_codes(symbol)
    current = _fetch_transcript(symbol, quarter_codes[0], company_name)
    prior = _fetch_transcript(symbol, quarter_codes[1], company_name)
    return current, prior
