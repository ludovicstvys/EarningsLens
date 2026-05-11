from __future__ import annotations

import io
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, SimpleQueue

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from earningslens.evasion import analyze_evasion
from earningslens.hedging import analyze_hedging
from earningslens.local_llm import warmup_local_llm
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
}

SIGNAL_COPY = {
    "green": ("🟢", "GREEN — Quarter looks clean", "#d8f3dc", "#1b4332"),
    "amber": ("🟡", "AMBER — Watch list", "#fff3bf", "#8d6e00"),
    "red": ("🔴", "RED — Multiple stress signals", "#ffe3e3", "#9b2226"),
}


def _load_inputs(mode: str):
    if mode == "Use sample":
        selection = st.sidebar.selectbox("Sample pair", list(SAMPLES.keys()))
        sample = SAMPLES[selection]
        return {
            "company": sample["company"],
            "current_quarter": sample["current_quarter"],
            "prior_quarter": sample["prior_quarter"],
            "current_text": sample["current_path"].read_text(),
            "prior_text": sample["prior_path"].read_text(),
        }

    if mode == "Search company":
        query = st.sidebar.text_input("Ticker", value=st.session_state.get("company_query", "MSFT"))
        st.session_state["company_query"] = query

        if st.sidebar.button("Find companies", key="find_companies"):
            if not query.strip():
                st.sidebar.error("Enter a ticker first.")
            else:
                st.session_state["search_results"] = search_companies(query)
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
            st.session_state["quarter_options"] = list_recent_quarters(selected.symbol)
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


def _build_pdf(brief) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

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
        f"Hedging density delta: {brief.hedging.delta_pct:+.1f}%",
        f"Topic similarity: {brief.topics.semantic_similarity:.2f}",
    ]:
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)
        pdf.drawString(40, y, line[:120])
        y -= 16

    flagged_evasions = [item for item in brief.evasions if item.flagged][:3]
    if flagged_evasions:
        pdf.drawString(40, y, "Top evasions:")
        y -= 16
        for item in flagged_evasions:
            pdf.drawString(50, y, f"- {item.answer_speaker}: {item.responsiveness}/10")
            y -= 16

    pdf.save()
    return buffer.getvalue()


def _run_analysis(input_payload: dict):
    progress = st.progress(0, text="Parsing transcripts")
    if input_payload.get("mode") == "search":
        if input_payload.get("current_quarter_code") and input_payload.get("prior_quarter_code"):
            progress.progress(5, text=f"Fetching transcripts for {input_payload['company']} ({input_payload['current_quarter']} vs {input_payload['prior_quarter']})")
            current_doc, prior_doc = fetch_transcript_pair(
                symbol=input_payload["symbol"],
                company_name=input_payload["company"],
                current_quarter_code=input_payload["current_quarter_code"],
                prior_quarter_code=input_payload["prior_quarter_code"],
            )
        else:
            progress.progress(5, text=f"Fetching recent transcripts for {input_payload['company']} ({input_payload['symbol']})")
            current_doc, prior_doc = fetch_latest_two_transcripts(
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

    current = parse_transcript(
        input_payload["current_text"],
        company=input_payload["company"],
        quarter=input_payload["current_quarter"],
    )
    prior = parse_transcript(
        input_payload["prior_text"],
        company=input_payload["company"],
        quarter=input_payload["prior_quarter"],
    )

    progress.progress(15, text="Loading local models")
    warmup_sentiment_model()
    warmup_topic_models()
    warmup_local_llm()

    progress.progress(20, text="Running sentiment / hedging / topics / risk vocab / evasion in parallel")

    progress_updates: SimpleQueue[tuple[float, str]] = SimpleQueue()

    def evasion_progress(fraction: float, label: str):
        progress_updates.put((fraction, label))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "sentiment": executor.submit(analyze_sentiment, current),
            "hedging": executor.submit(analyze_hedging, current, prior),
            "topics": executor.submit(analyze_topics, current, prior),
            "risk_vocab": executor.submit(analyze_risk_vocab, current, prior),
            "evasions": executor.submit(analyze_evasion, current, evasion_progress),
        }
        results = {}
        pending = {future: name for name, future in futures.items()}
        while pending:
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            while True:
                try:
                    fraction, label = progress_updates.get_nowait()
                except Empty:
                    break
                progress.progress(min(80, 20 + int(fraction * 50)), text=label)
            for future in done:
                name = pending.pop(future)
                results[name] = future.result()

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

    sentiment_tab, hedging_tab, topics_tab, risk_tab = st.tabs(["Sentiment", "Hedging", "Topics", "Risk vocab"])

    with sentiment_tab:
        col1, col2 = st.columns(2)
        with col1:
            _metric_bar("Prepared remarks", brief.sentiment.prepared_score, -1, 1)
        with col2:
            _metric_bar("Q&A", brief.sentiment.qa_score, -1, 1)
        st.metric("Gap", f"{brief.sentiment.gap:+.2f}", help="Negative means the Q&A was more negative than prepared remarks.")
        st.caption("A negative gap means management sounded worse under questioning than in prepared remarks.")

    with hedging_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prior density", f"{brief.hedging.prior_density:.1f} / 1000w")
            st.progress(min(100, int(brief.hedging.prior_density)))
        with col2:
            st.metric("Current density", f"{brief.hedging.current_density:.1f} / 1000w")
            st.progress(min(100, int(brief.hedging.current_density)))
        st.metric("Delta", f"{brief.hedging.delta_pct:+.1f}%")
        st.table([{"hedge": word, "count": count} for word, count in brief.hedging.top_hedges])

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

    flagged_only_default = any(item.flagged for item in brief.evasions)
    with st.expander("Q&A Evasion Inspector", expanded=flagged_only_default):
        show_all = st.toggle("Show all Q&A pairs, not just flagged", value=False)
        evasion_rows = brief.evasions if show_all else [item for item in brief.evasions if item.flagged]
        if not evasion_rows:
            st.write("No flagged evasions found.")
        for item in evasion_rows:
            st.markdown(f"**{item.question_speaker}:** {item.question}")
            st.markdown(f"> *{item.answer_speaker}: {item.answer}*")
            st.caption(f"Responsiveness: {item.responsiveness}/10")
            st.write(item.reasoning)
            st.divider()

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

    if "brief" in st.session_state:
        _render_results(st.session_state["brief"])
    else:
        st.info("Search by exact ticker or select a sample pair, then click Analyze.")


if __name__ == "__main__":
    main()
