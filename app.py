from __future__ import annotations

import io
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, SimpleQueue

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from earningslens import config
from earningslens.evasion import analyze_evasion
from earningslens.hedging import analyze_hedging
from earningslens.local_llm import warmup_local_llm
from earningslens.models import EvasionScore, HedgingAnalysis, RiskVocabItem, SentimentAnalysis, TopicDrift
from earningslens.parser import parse_transcript
from earningslens.risk_vocab import analyze_risk_vocab
from earningslens.sentiment import analyze_sentiment, warmup_sentiment_model
from earningslens.synthesizer import synthesize
from earningslens.transcript_fetcher import (
    TranscriptFetchError,
    fetch_latest_two_transcripts,
    fetch_transcript_pair,
    list_recent_quarters,
    search_companies,
)
from earningslens.topics import analyze_topics, warmup_topic_models

LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

SAMPLES = {
    "ACME Q2→Q3 2025": {
        "company": "ACME",
        "current_quarter": "Q3 2025",
        "prior_quarter": "Q2 2025",
        "current_path": Path("data/samples/ACME_Q3_2025.txt"),
        "prior_path": Path("data/samples/ACME_Q2_2025.txt"),
    },
    "BRAVO Q1→Q2 2025": {
        "company": "BRAVO",
        "current_quarter": "Q2 2025",
        "prior_quarter": "Q1 2025",
        "current_path": Path("data/samples/BRAVO_Q2_2025.txt"),
        "prior_path": Path("data/samples/BRAVO_Q1_2025.txt"),
    },
    "Microsoft Q4 2025→Q1 2026 (Real)": {
        "company": "Microsoft",
        "current_quarter": "Q1 2026",
        "prior_quarter": "Q4 2025",
        "current_path": Path("data/samples/MSFT_Q1_2026.txt"),
        "prior_path": Path("data/samples/MSFT_Q4_2025.txt"),
    },
}

SIGNAL_COPY = {
    "green": ("🟢", "GREEN — Quarter looks clean", "#d8f3dc", "#1b4332"),
    "amber": ("🟡", "AMBER — Watch list", "#fff3bf", "#8d6e00"),
    "red": ("🔴", "RED — Multiple stress signals", "#ffe3e3", "#9b2226"),
}

@st.cache_data(show_spinner=False)
def _read_text_file(path_str: str) -> str:
    return Path(path_str).read_text()


@st.cache_data(show_spinner=False)
def _search_companies_cached(query: str):
    return search_companies(query)


@st.cache_data(show_spinner=False)
def _list_recent_quarters_cached(symbol: str):
    return list_recent_quarters(symbol)


@st.cache_data(show_spinner=False)
def _fetch_transcript_pair_cached(symbol: str, company_name: str, current_quarter_code: str, prior_quarter_code: str):
    return fetch_transcript_pair(symbol, company_name, current_quarter_code, prior_quarter_code)


@st.cache_data(show_spinner=False)
def _fetch_latest_two_transcripts_cached(symbol: str, company_name: str):
    return fetch_latest_two_transcripts(symbol, company_name)


@st.cache_data(show_spinner=False)
def _parse_transcript_cached(text: str, company: str, quarter: str, ticker: str | None = None):
    return parse_transcript(text, company=company, quarter=quarter, ticker=ticker)


def _render_sidebar_options() -> None:
    st.session_state["show_debug"] = st.sidebar.toggle(
        "Show diagnostics",
        value=bool(st.session_state.get("show_debug", False)),
    )


def _fallback_sentiment(reason: str, current) -> SentimentAnalysis:
    qa_missing = not current.qa_text.strip()
    prepared = 0.0
    return SentimentAnalysis(
        prepared_score=prepared,
        qa_score=prepared,
        gap=0.0,
        flagged=False,
        source="fallback",
        notes=[reason] + (["Transcript had no parsed Q&A text."] if qa_missing else []),
    )


def _fallback_hedging(reason: str) -> HedgingAnalysis:
    return HedgingAnalysis(
        current_density=0.0,
        prior_density=0.0,
        delta_pct=0.0,
        flagged=False,
        top_hedges=[],
        source="fallback",
        notes=[reason],
    )


def _fallback_topics(reason: str) -> TopicDrift:
    return TopicDrift(
        current_themes=[],
        prior_themes=[],
        new_themes=[],
        dropped_themes=[],
        semantic_similarity=1.0,
        flagged=False,
        source="fallback",
        notes=[reason],
    )


def _fallback_risk_vocab() -> list[RiskVocabItem]:
    return []


def _fallback_evasions(reason: str) -> list[EvasionScore]:
    return [
        EvasionScore(
            question="",
            question_speaker="",
            answer="",
            answer_speaker="",
            responsiveness=10,
            reasoning=reason,
            flagged=False,
            pair_confidence=0.0,
            low_confidence=True,
            source="fallback",
            notes=[reason],
        )
    ]


def _load_inputs(mode: str):
    if mode == "Use sample":
        selection = st.sidebar.selectbox("Sample pair", list(SAMPLES.keys()))
        sample = SAMPLES[selection]
        return {
            "company": sample["company"],
            "current_quarter": sample["current_quarter"],
            "prior_quarter": sample["prior_quarter"],
            "current_text": _read_text_file(str(sample["current_path"])),
            "prior_text": _read_text_file(str(sample["prior_path"])),
        }

    if mode == "Search company":
        query = st.sidebar.text_input("Ticker", value=st.session_state.get("company_query", "MSFT"))
        st.session_state["company_query"] = query

        if st.sidebar.button("Find companies", key="find_companies"):
            if not query.strip():
                st.sidebar.error("Enter a ticker first.")
            else:
                st.session_state["search_results"] = _search_companies_cached(query)
                st.session_state.pop("quarter_options", None)
                st.session_state.pop("quarter_options_symbol", None)

        results = st.session_state.get("search_results", [])
        if not results:
            st.sidebar.caption("Enter an exact ticker, then choose the returned match.")
            return None

        labels = [result.label for result in results]
        selected_label = st.sidebar.selectbox("Best matches", labels, key="search_match")
        selected = next(result for result in results if result.label == selected_label)

        if st.sidebar.button("Load quarters", key="load_quarters"):
            st.session_state["quarter_options"] = _list_recent_quarters_cached(selected.symbol)
            st.session_state["quarter_options_symbol"] = selected.symbol

        quarter_options = st.session_state.get("quarter_options", [])
        current_symbol = st.session_state.get("quarter_options_symbol")
        if not quarter_options or current_symbol != selected.symbol:
            st.sidebar.caption("Load recent quarters, then choose the pair to compare.")
            return None

        quarter_labels = [option.label for option in quarter_options]
        current_label = st.sidebar.selectbox("Current quarter", quarter_labels, index=0, key="current_quarter_pick")
        remaining_prior_labels = [label for label in quarter_labels if label != current_label]
        if not remaining_prior_labels:
            st.sidebar.error("Not enough distinct quarters available for comparison.")
            return None
        prior_label = st.sidebar.selectbox(
            "Prior quarter",
            remaining_prior_labels,
            index=0,
            key="prior_quarter_pick",
        )
        current_option = next(option for option in quarter_options if option.label == current_label)
        prior_option = next(option for option in quarter_options if option.label == prior_label)
        return {
            "mode": "search",
            "company": selected.name,
            "symbol": selected.symbol,
            "current_quarter_code": current_option.code,
            "prior_quarter_code": prior_option.code,
            "current_quarter": current_option.label,
            "prior_quarter": prior_option.label,
        }
    return None


def _metric_bar(label: str, value: float, min_value: float, max_value: float):
    normalized = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0
    st.caption(f"{label}: {value:+.2f}")
    st.progress(max(0, min(100, int(normalized * 100))))


def _render_chip(text: str, color: str, background: str):
    st.markdown(
        f"<span style='display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;background:{background};color:{color};margin:0.15rem;'>{text}</span>",
        unsafe_allow_html=True,
    )


def _format_delta_pct(value: float) -> str:
    return "NEW" if value == 999.0 else f"{value:+.1f}%"


def _build_pdf(brief) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    text_width = width - 80

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, y, f"EarningsLens Brief — {brief.company} {brief.quarter}")
    y -= 28

    pdf.setFont("Helvetica", 11)
    for line in [
        f"Signal: {brief.overall_signal.upper()} | Flags: {brief.flag_count}",
        brief.executive_summary,
        "",
        "Bullet points:",
        *[f"- {bullet}" for bullet in brief.bullet_points],
        "",
        f"Sentiment gap: {brief.sentiment.gap:+.2f}",
        f"Hedging density delta: {_format_delta_pct(brief.hedging.delta_pct)}",
        f"Topic similarity: {brief.topics.semantic_similarity:.2f}",
    ]:
        wrapped_lines = simpleSplit(line, "Helvetica", 11, text_width) or [""]
        for wrapped_line in wrapped_lines:
            if y < 80:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 11)
            pdf.drawString(40, y, wrapped_line)
            y -= 16

    flagged_evasions = [item for item in brief.evasions if item.flagged][:3]
    if flagged_evasions:
        pdf.drawString(40, y, "Top evasions:")
        y -= 16
        for item in flagged_evasions:
            for wrapped_line in simpleSplit(f"- {item.answer_speaker}: {item.responsiveness}/10", "Helvetica", 11, text_width - 10):
                if y < 80:
                    pdf.showPage()
                    y = height - 50
                    pdf.setFont("Helvetica", 11)
                pdf.drawString(50, y, wrapped_line)
                y -= 16

    pdf.save()
    return buffer.getvalue()


def _run_analysis(input_payload: dict):
    progress = st.progress(0, text="Parsing transcripts")
    if input_payload.get("mode") == "search":
        if input_payload.get("current_quarter_code") and input_payload.get("prior_quarter_code"):
            progress.progress(5, text=f"Fetching transcripts for {input_payload['company']} ({input_payload['current_quarter']} vs {input_payload['prior_quarter']})")
            current_doc, prior_doc = _fetch_transcript_pair_cached(
                symbol=input_payload["symbol"],
                company_name=input_payload["company"],
                current_quarter_code=input_payload["current_quarter_code"],
                prior_quarter_code=input_payload["prior_quarter_code"],
            )
        else:
            progress.progress(5, text=f"Fetching recent transcripts for {input_payload['company']} ({input_payload['symbol']})")
            current_doc, prior_doc = _fetch_latest_two_transcripts_cached(
                symbol=input_payload["symbol"],
                company_name=input_payload["company"],
            )
        input_payload = {
            "company": current_doc.company,
            "symbol": current_doc.symbol,
            "current_quarter": current_doc.quarter_label,
            "prior_quarter": prior_doc.quarter_label,
            "current_text": current_doc.transcript_text,
            "prior_text": prior_doc.transcript_text,
        }

    current = _parse_transcript_cached(
        input_payload["current_text"],
        company=input_payload["company"],
        quarter=input_payload["current_quarter"],
        ticker=input_payload.get("symbol"),
    )
    prior = _parse_transcript_cached(
        input_payload["prior_text"],
        company=input_payload["company"],
        quarter=input_payload["prior_quarter"],
        ticker=input_payload.get("symbol"),
    )

    progress.progress(15, text="Loading local models")
    warmup_sentiment_model()
    warmup_topic_models()
    warmup_local_llm()

    progress.progress(20, text="Running sentiment / hedging / topics / risk vocab / evasion in parallel")

    progress_updates: SimpleQueue[tuple[float, str]] = SimpleQueue()

    def evasion_progress(fraction: float, label: str):
        progress_updates.put((fraction, label))

    results: dict[str, object] = {}
    analysis_notes: dict[str, list[str]] = {
        "parser": [*current.parser_notes, *prior.parser_notes],
    }
    analysis_provenance: dict[str, str] = {}
    fallback_builders = {
        "sentiment": lambda reason: _fallback_sentiment(reason, current),
        "hedging": lambda reason: _fallback_hedging(reason),
        "topics": lambda reason: _fallback_topics(reason),
        "risk_vocab": lambda reason: _fallback_risk_vocab(),
        "evasions": lambda reason: _fallback_evasions(reason),
    }

    executor = ThreadPoolExecutor(max_workers=5)
    started_at = time.monotonic()
    try:
        futures = {
            "sentiment": executor.submit(analyze_sentiment, current),
            "hedging": executor.submit(analyze_hedging, current, prior),
            "topics": executor.submit(analyze_topics, current, prior),
            "risk_vocab": executor.submit(analyze_risk_vocab, current, prior),
            "evasions": executor.submit(analyze_evasion, current, evasion_progress),
        }
        pending = {future: name for name, future in futures.items()}
        while pending:
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            while True:
                try:
                    fraction, label = progress_updates.get_nowait()
                except Empty:
                    break
                progress.progress(min(80, 20 + int(fraction * 50)), text=label)

            elapsed = time.monotonic() - started_at
            timed_out = elapsed >= config.ANALYSIS_TIMEOUT_SECONDS
            for future in done:
                name = pending.pop(future)
                try:
                    result = future.result()
                    results[name] = result
                    if name == "evasions":
                        sources = sorted({item.source for item in result if getattr(item, "source", "")})
                        analysis_provenance[name] = ", ".join(sources) if sources else "computed"
                        combined_notes: list[str] = []
                        for item in result:
                            combined_notes.extend(item.notes)
                        analysis_notes[name] = combined_notes
                    elif name == "risk_vocab":
                        analysis_provenance[name] = "normalized_lexicon"
                        analysis_notes[name] = []
                    else:
                        analysis_provenance[name] = getattr(result, "source", "computed")
                        analysis_notes[name] = list(getattr(result, "notes", []))
                except Exception as exc:
                    LOGGER.exception("%s analysis failed", name)
                    reason = f"{name} analysis failed: {exc}"
                    results[name] = fallback_builders[name](reason)
                    analysis_provenance[name] = "fallback_error"
                    analysis_notes[name] = [reason]

            if timed_out:
                for future, name in list(pending.items()):
                    future.cancel()
                    reason = f"{name} analysis timed out after {config.ANALYSIS_TIMEOUT_SECONDS} seconds."
                    results[name] = fallback_builders[name](reason)
                    analysis_provenance[name] = "fallback_timeout"
                    analysis_notes[name] = [reason]
                    pending.pop(future)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for name in ["sentiment", "hedging", "topics", "risk_vocab", "evasions"]:
        if name not in results:
            reason = f"{name} analysis did not return a result."
            results[name] = fallback_builders[name](reason)
            analysis_provenance[name] = "fallback_missing"
            analysis_notes[name] = [reason]

    if "risk_vocab" not in analysis_notes:
        analysis_notes["risk_vocab"] = []
    if not results["risk_vocab"]:
        analysis_notes["risk_vocab"].append("No tracked risk-vocabulary movers were detected in either quarter.")
    analysis_provenance.setdefault("risk_vocab", "normalized_lexicon")

    evasion_notes = analysis_notes.setdefault("evasions", [])
    low_conf_pairs = sum(1 for item in results["evasions"] if getattr(item, "low_confidence", False))
    if low_conf_pairs:
        evasion_notes.append(f"{low_conf_pairs} Q&A pairs were skipped or downweighted due to low parser confidence.")

    parser_overview = {
        "current_parse_confidence": round(current.parse_confidence, 2),
        "prior_parse_confidence": round(prior.parse_confidence, 2),
        "current_qa_boundary_detected": current.qa_boundary_detected,
        "prior_qa_boundary_detected": prior.qa_boundary_detected,
        "current_turns": len(current.turns),
        "prior_turns": len(prior.turns),
        "current_turn_preview": [
            {
                "idx": turn.idx,
                "section": turn.section,
                "speaker": turn.speaker,
                "title": turn.title or "",
                "role": turn.role,
                "confidence": round(turn.role_confidence, 2),
                "text": turn.text[:180],
            }
            for turn in current.turns[:40]
        ],
        "prior_turn_preview": [
            {
                "idx": turn.idx,
                "section": turn.section,
                "speaker": turn.speaker,
                "title": turn.title or "",
                "role": turn.role,
                "confidence": round(turn.role_confidence, 2),
                "text": turn.text[:180],
            }
            for turn in prior.turns[:40]
        ],
    }

    progress.progress(85, text="Synthesizing brief")
    brief = synthesize(
        company=input_payload["company"],
        quarter=input_payload["current_quarter"],
        prior_quarter=input_payload["prior_quarter"],
        sentiment=results["sentiment"],
        hedging=results["hedging"],
        topics=results["topics"],
        risk_vocab=results["risk_vocab"],
        evasions=results["evasions"],
        analysis_provenance=analysis_provenance,
        analysis_notes=analysis_notes,
        parser_overview=parser_overview,
    )
    progress.progress(100, text="Complete")
    return brief


def _render_results(brief):
    emoji, label, bg, fg = SIGNAL_COPY[brief.overall_signal]
    hero_left, hero_right = st.columns([1, 2])
    with hero_left:
        st.markdown(
            f"""
            <div style="background:{bg};color:{fg};padding:1.25rem;border-radius:18px;text-align:center;">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="font-weight:700;">{label}</div>
                <div style="margin-top:0.35rem;">{brief.flag_count} flagged dimensions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(f"> {brief.executive_summary}")
        for bullet in brief.bullet_points:
            st.write(f"- {bullet}")
        provenance_line = ", ".join(f"{name}: `{source}`" for name, source in brief.analysis_provenance.items())
        if provenance_line:
            st.caption(f"Analysis provenance: {provenance_line}")

    sentiment_tab, hedging_tab, topics_tab, risk_tab, diagnostics_tab = st.tabs(
        ["Sentiment", "Hedging", "Topics", "Risk vocab", "Diagnostics"]
    )

    with sentiment_tab:
        col1, col2 = st.columns(2)
        with col1:
            _metric_bar("Prepared remarks", brief.sentiment.prepared_score, -1, 1)
        with col2:
            _metric_bar("Q&A", brief.sentiment.qa_score, -1, 1)
        st.metric("Gap", f"{brief.sentiment.gap:+.2f}", help="Negative means the Q&A was more negative than prepared remarks.")
        st.caption("A negative gap means management sounded worse under questioning than in prepared remarks.")
        for note in brief.analysis_notes.get("sentiment", []):
            st.caption(note)

    with hedging_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prior density", f"{brief.hedging.prior_density:.1f} / 1000w")
            st.progress(min(100, int(brief.hedging.prior_density)))
        with col2:
            st.metric("Current density", f"{brief.hedging.current_density:.1f} / 1000w")
            st.progress(min(100, int(brief.hedging.current_density)))
        st.metric("Delta", _format_delta_pct(brief.hedging.delta_pct))
        st.table([{"hedge": word, "count": count} for word, count in brief.hedging.top_hedges])
        for note in brief.analysis_notes.get("hedging", []):
            st.caption(note)

    with topics_tab:
        st.metric("Semantic similarity", f"{brief.topics.semantic_similarity:.2f}")
        st.progress(max(0, min(100, int(brief.topics.semantic_similarity * 100))))
        left, right = st.columns(2)
        with left:
            st.subheader("New themes")
            for theme in brief.topics.new_themes or ["None"]:
                risk_flag = any(term in theme.lower() for item in brief.risk_vocab[:20] for term in [item.keyword.lower()])
                _render_chip(theme, "#9b2226" if risk_flag else "#7f5539", "#ffe3e3" if risk_flag else "#fefae0")
        with right:
            st.subheader("Dropped themes")
            for theme in brief.topics.dropped_themes or ["None"]:
                _render_chip(theme, "#495057", "#e9ecef")
        with st.expander("Full theme lists"):
            st.write("Current:", brief.topics.current_themes)
            st.write("Prior:", brief.topics.prior_themes)
        for note in brief.analysis_notes.get("topics", []):
            st.caption(note)

    with risk_tab:
        rows = []
        for item in brief.risk_vocab:
            delta_pct = "NEW" if item.delta_pct == 999.0 else f"{item.delta_pct:+.1f}%"
            rows.append(
                {
                    "keyword": item.keyword,
                    "category": item.category,
                    "prior": item.prior_count,
                    "current": item.current_count,
                    "Δ": item.delta,
                    "Δ%": delta_pct,
                }
            )
        st.dataframe(rows, use_container_width=True)
        st.caption("Rows with large positive deltas or NEW entries are the biggest vocabulary movers.")
        for note in brief.analysis_notes.get("risk_vocab", []):
            st.caption(note)

    flagged_only_default = any(item.flagged for item in brief.evasions)
    with st.expander("Q&A Evasion Inspector", expanded=flagged_only_default):
        show_all = st.toggle("Show all Q&A pairs, not just flagged", value=False)
        flagged_evasions = [item for item in brief.evasions if item.flagged]
        evasion_rows = brief.evasions if show_all else flagged_evasions
        st.caption(f"{len(flagged_evasions)} flagged of {len(brief.evasions)} Q&A pairs scored.")
        if not evasion_rows:
            if brief.evasions:
                st.write("No flagged evasions found. Turn on `Show all Q&A pairs, not just flagged` to review scored pairs.")
            else:
                st.write("No Q&A pairs were extracted for evasion scoring.")
        for item in evasion_rows:
            if not item.question and not item.answer:
                st.caption(item.reasoning)
                continue
            st.markdown(f"**{item.question_speaker}:** {item.question}")
            st.markdown(f"> *{item.answer_speaker}: {item.answer}*")
            confidence_label = f"Pair confidence: {item.pair_confidence:.2f}"
            source_label = f"Source: {item.source}"
            if item.low_confidence:
                st.caption(f"{confidence_label} | {source_label} | Low-confidence pair")
            else:
                st.caption(f"Responsiveness: {item.responsiveness}/10 | {confidence_label} | {source_label}")
            st.write(item.reasoning)
            for note in item.notes:
                st.caption(note)
            st.divider()

    with diagnostics_tab:
        parser_overview = brief.parser_overview
        cols = st.columns(4)
        cols[0].metric("Current parse confidence", f"{parser_overview.get('current_parse_confidence', 0):.2f}")
        cols[1].metric("Prior parse confidence", f"{parser_overview.get('prior_parse_confidence', 0):.2f}")
        cols[2].metric("Current turns", str(parser_overview.get("current_turns", 0)))
        cols[3].metric("Prior turns", str(parser_overview.get("prior_turns", 0)))
        st.caption(
            f"Q&A boundary detected: current={parser_overview.get('current_qa_boundary_detected')} | prior={parser_overview.get('prior_qa_boundary_detected')}"
        )
        for note in brief.analysis_notes.get("parser", []):
            st.caption(note)
        if st.session_state.get("show_debug", False):
            with st.expander("Parsed Turns Preview"):
                st.write("Current quarter")
                st.dataframe(parser_overview.get("current_turn_preview", []), use_container_width=True)
                st.write("Prior quarter")
                st.dataframe(parser_overview.get("prior_turn_preview", []), use_container_width=True)
        else:
            st.caption("Enable `Show diagnostics` in the sidebar to inspect parsed speaker roles and turn segmentation.")
        with st.expander("Analysis Notes"):
            for name, notes in brief.analysis_notes.items():
                if not notes:
                    continue
                st.write(f"**{name}**")
                for note in notes:
                    st.write(f"- {note}")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download analyst brief (PDF)",
            data=_build_pdf(brief),
            file_name=f"{brief.company}_{brief.quarter.replace(' ', '_')}_brief.pdf",
            mime="application/pdf",
        )
    with col2:
        if st.button("Re-run"):
            st.session_state.pop("brief", None)
            st.rerun()


def main():
    st.set_page_config(page_title="EarningsLens", layout="wide")

    if "show_disclaimer" not in st.session_state:
        st.session_state.show_disclaimer = True

    st.title("🔍 EarningsLens")
    st.caption("Read earnings calls like a senior analyst.")

    if st.session_state.show_disclaimer:
        cols = st.columns([8, 1])
        with cols[0]:
            st.warning(
                "EarningsLens is an analytical tool, not investment advice. Outputs are AI-generated and may contain errors. Do your own diligence."
            )
        with cols[1]:
            if st.button("Dismiss"):
                st.session_state.show_disclaimer = False
                st.rerun()

    st.sidebar.header("Inputs")
    _render_sidebar_options()
    mode = st.sidebar.radio("Mode", ["Search company", "Use sample"])
    payload = _load_inputs(mode)

    if st.sidebar.button("Analyze", type="primary"):
        if payload is None:
            if mode == "Search company":
                st.sidebar.error("Search by exact ticker and choose the returned match first.")
            else:
                st.sidebar.error("Choose a sample pair first.")
        else:
            try:
                st.session_state["brief"] = _run_analysis(payload)
            except TranscriptFetchError as exc:
                st.session_state.pop("brief", None)
                st.error(str(exc))
            except Exception as exc:
                LOGGER.exception("Analysis failed")
                st.session_state.pop("brief", None)
                st.error(f"Analysis failed: {exc}")

    if "brief" in st.session_state:
        _render_results(st.session_state["brief"])
    else:
        st.info("Search by exact ticker or select a sample pair, then click Analyze.")


if __name__ == "__main__":
    main()
