"""Generate a polished PDF user guide for EarningsLens."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "EarningsLens_User_Guide.pdf"

NAVY = colors.HexColor("#0B2545")
ACCENT = colors.HexColor("#C8553D")
SOFT = colors.HexColor("#13315C")
MUTED = colors.HexColor("#5C677D")
BG = colors.HexColor("#F5F7FA")
RULE = colors.HexColor("#D9DEE5")
GREEN = colors.HexColor("#1B998B")
AMBER = colors.HexColor("#E0A458")
RED = colors.HexColor("#C8553D")


def make_styles():
    styles = getSampleStyleSheet()
    base_font = "Helvetica"

    styles.add(ParagraphStyle(
        name="CoverTitle", fontName=f"{base_font}-Bold",
        fontSize=42, leading=48, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CoverSub", fontName=base_font,
        fontSize=15, leading=20, textColor=SOFT, alignment=TA_LEFT,
        spaceAfter=24,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta", fontName=base_font,
        fontSize=10, leading=14, textColor=MUTED, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="ELH1", fontName=f"{base_font}-Bold",
        fontSize=22, leading=28, textColor=NAVY,
        spaceBefore=6, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="ELH2", fontName=f"{base_font}-Bold",
        fontSize=14, leading=20, textColor=ACCENT,
        spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ELH3", fontName=f"{base_font}-Bold",
        fontSize=11.5, leading=16, textColor=SOFT,
        spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ELBody", fontName=base_font,
        fontSize=10.5, leading=15.5, textColor=colors.HexColor("#1A1A1A"),
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ELBullet", fontName=base_font,
        fontSize=10.5, leading=15, textColor=colors.HexColor("#1A1A1A"),
        leftIndent=14, bulletIndent=2, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ELCode", fontName="Courier",
        fontSize=9.5, leading=13, textColor=NAVY,
        backColor=BG, leftIndent=8, rightIndent=8,
        borderPadding=6, spaceBefore=4, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ELCallout", fontName=base_font,
        fontSize=10, leading=14, textColor=SOFT,
        leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=8,
        backColor=BG, borderPadding=8,
    ))
    styles.add(ParagraphStyle(
        name="ELFooter", fontName=base_font,
        fontSize=8.5, leading=10, textColor=MUTED, alignment=TA_CENTER,
    ))
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    # Header rule (skip on cover, page 1)
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, height - 1.3 * cm, "EarningsLens")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - 2 * cm, height - 1.3 * cm, "User Guide")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, height - 1.5 * cm, width - 2 * cm, height - 1.5 * cm)

        # Footer
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(2 * cm, 1.2 * cm, "EarningsLens — Analytical aid, not investment advice.")
        canvas.drawRightString(width - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def cover_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    # Left navy band
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, 2.2 * cm, height, fill=1, stroke=0)
    # Accent block
    canvas.setFillColor(ACCENT)
    canvas.rect(2.2 * cm, height - 5.2 * cm, 1.2 * cm, 1.2 * cm, fill=1, stroke=0)

    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 46)
    canvas.drawString(4.2 * cm, height - 6 * cm, "EarningsLens")

    canvas.setFillColor(SOFT)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(4.2 * cm, height - 7.2 * cm, "User Guide")

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 11)
    canvas.drawString(4.2 * cm, height - 8.4 * cm,
                      "An analyst-style read of two earnings calls,")
    canvas.drawString(4.2 * cm, height - 8.9 * cm,
                      "powered by local Hugging Face models.")

    # Bottom block
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.8)
    canvas.line(4.2 * cm, 4 * cm, width - 2 * cm, 4 * cm)

    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(4.2 * cm, 3.4 * cm, "Version 1.0")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(4.2 * cm, 2.9 * cm, "Document type: End-user guide")
    canvas.drawString(4.2 * cm, 2.4 * cm, "Audience: Analysts, students, and reviewers")

    canvas.restoreState()


def bullet(text, styles):
    return Paragraph(text, styles["ELBullet"], bulletText="•")


def signal_table():
    data = [
        [Paragraph("<b>GREEN</b>", ParagraphStyle("g", fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("No major issues flagged across sentiment, hedging, topics, and Q&amp;A responsiveness.",
                   ParagraphStyle("gd", fontName="Helvetica", fontSize=10, leading=14))],
        [Paragraph("<b>AMBER</b>", ParagraphStyle("a", fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("One analytical dimension moved enough to warrant monitoring next quarter.",
                   ParagraphStyle("ad", fontName="Helvetica", fontSize=10, leading=14))],
        [Paragraph("<b>RED</b>", ParagraphStyle("r", fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph("Multiple independent stress signals appeared at once — read the transcripts closely.",
                   ParagraphStyle("rd", fontName="Helvetica", fontSize=10, leading=14))],
    ]
    t = Table(data, colWidths=[2.4 * cm, 13.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), GREEN),
        ("BACKGROUND", (0, 1), (0, 1), AMBER),
        ("BACKGROUND", (0, 2), (0, 2), RED),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("LEFTPADDING", (1, 0), (1, -1), 10),
        ("RIGHTPADDING", (1, 0), (1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [BG, colors.white, BG]),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
    ]))
    return t


def pipeline_diagram(styles):
    box = ParagraphStyle("pbox", fontName="Helvetica-Bold", fontSize=10,
                         leading=12, textColor=colors.white, alignment=TA_CENTER)
    sub = ParagraphStyle("psub", fontName="Helvetica", fontSize=8.5,
                         leading=10, textColor=colors.HexColor("#E8ECF3"), alignment=TA_CENTER)
    arrow = ParagraphStyle("parr", fontName="Helvetica-Bold", fontSize=12,
                           leading=14, textColor=ACCENT, alignment=TA_CENTER)

    def cell(title, sub_text=None, color=NAVY):
        bits = [Paragraph(title, box)]
        if sub_text:
            bits.append(Paragraph(sub_text, sub))
        t = Table([[bits]], colWidths=[5.0 * cm], rowHeights=[1.4 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0, color),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    arr_down = Paragraph("&#9660;", arrow)
    arr_right = Paragraph("&#9654;", arrow)

    analyzers = [
        ("sentiment.py", "FinBERT tone gap"),
        ("hedging.py", "uncertainty lexicon"),
        ("topics.py", "MiniLM + LLM themes"),
        ("risk_vocab.py", "curated vocab"),
        ("evasion.py", "LLM-as-judge"),
    ]
    rows = [[cell(t, s, SOFT)] for t, s in analyzers]
    analyzers_tbl = Table(rows, colWidths=[5.0 * cm])
    analyzers_tbl.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    # Layout: parser -> [analyzers stacked] -> synthesizer -> brief
    middle_row = Table(
        [[cell("parser.py", "regex + state machine", NAVY), arr_right, analyzers_tbl,
          arr_right, cell("synthesizer.py", "local LLM brief", NAVY)]],
        colWidths=[4.5 * cm, 0.9 * cm, 5.2 * cm, 0.9 * cm, 4.5 * cm],
    )
    middle_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    top = cell("Two transcripts (.txt)", "current + prior quarter", ACCENT)
    bottom = cell("EarningsBrief  →  app.py", "headline signal + 5 tabs", ACCENT)

    outer = Table(
        [[top], [arr_down], [middle_row], [arr_down], [bottom]],
        colWidths=[16.0 * cm],
    )
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return outer


def env_table():
    hdr = ParagraphStyle("eh", fontName="Helvetica-Bold", fontSize=10,
                         leading=12, textColor=colors.white)
    key = ParagraphStyle("ek", fontName="Courier-Bold", fontSize=8.5,
                         leading=11, textColor=NAVY)
    body = ParagraphStyle("eb", fontName="Helvetica", fontSize=9,
                          leading=12, textColor=colors.HexColor("#1A1A1A"))
    req = ParagraphStyle("er", fontName="Helvetica-Bold", fontSize=8,
                         leading=11, textColor=colors.white, alignment=TA_CENTER)

    def chip(text, color):
        return Table([[Paragraph(text, req)]], colWidths=[1.8 * cm], rowHeights=[0.5 * cm],
                     style=TableStyle([
                         ("BACKGROUND", (0, 0), (-1, -1), color),
                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                         ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                         ("LEFTPADDING", (0, 0), (-1, -1), 2),
                         ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                         ("TOPPADDING", (0, 0), (-1, -1), 1),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                     ]))

    rows = [
        [Paragraph("Variable", hdr), Paragraph("Req?", hdr),
         Paragraph("What it does", hdr), Paragraph("Default", hdr)],
        [Paragraph("MODEL", key), chip("Opt", SOFT),
         Paragraph("Hugging Face model id for the local instruct LLM used by "
                   "topic extraction, Q&amp;A evasion scoring, and brief synthesis.", body),
         Paragraph("SmolLM2-1.7B-<br/>Instruct", body)],
        [Paragraph("ALPHA_VANTAGE_<br/>API_KEY", key), chip("If search", ACCENT),
         Paragraph("Free Alpha Vantage key used by <b>Search company</b> mode to fetch "
                   "real ticker transcripts. Not needed for <b>Use sample</b> mode.", body),
         Paragraph("—", body)],
        [Paragraph("EVASION_<br/>BATCH_SIZE", key), chip("Opt", SOFT),
         Paragraph("Number of Q&amp;A pairs to score in parallel when evaluating evasion. "
                   "Reduce if memory-constrained.", body),
         Paragraph("6", body)],
        [Paragraph("ANALYSIS_<br/>TIMEOUT_SECONDS", key), chip("Opt", SOFT),
         Paragraph("Maximum seconds per analysis before falling back to a simpler method. "
                   "Set via <b>.env</b>, not sidebar.", body),
         Paragraph("300", body)],
    ]
    t = Table(rows, colWidths=[3.2 * cm, 1.8 * cm, 5.8 * cm, 2.2 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("INNERGRID", (0, 1), (-1, -1), 0.3, RULE),
    ]))
    return t


def tab_table():
    hdr = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=10,
                         leading=12, textColor=colors.white)
    name = ParagraphStyle("tn", fontName="Helvetica-Bold", fontSize=10,
                          leading=13, textColor=ACCENT)
    body = ParagraphStyle("tb", fontName="Helvetica", fontSize=9.5,
                          leading=13, textColor=colors.HexColor("#1A1A1A"))

    def P(text, style):
        return Paragraph(text, style)

    rows = [
        [P("Tab", hdr), P("What it measures", hdr), P("What to look at", hdr)],
        [P("Sentiment", name),
         P("FinBERT tone of prepared remarks vs. tone during Q&amp;A.", body),
         P("A widening negative gap means the unscripted portion was darker than the script.", body)],
        [P("Hedging", name),
         P("Use of uncertainty words (may, could, pressure, headwind…) normalised per turn.", body),
         P("A sharp quarter-over-quarter rise often signals weaker conviction.", body)],
        [P("Topics", name),
         P("Semantic similarity of prepared remarks via MiniLM + themes extracted by the local LLM.", body),
         P("New, risk-heavy themes and dropped themes are highlighted.", body)],
        [P("Risk vocab", name),
         P("Curated finance vocabulary (inflation, liquidity, competition, backlog…).", body),
         P("Biggest movers surface first — useful for narrative shifts.", body)],
        [P("Q&amp;A Evasion", name),
         P("LLM-as-judge rubric scoring whether each answer addressed its question.", body),
         P("Low scores flag exchanges that a human should re-read, not proof of misconduct.", body)],
    ]
    t = Table(rows, colWidths=[2.8 * cm, 6.6 * cm, 6.6 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("INNERGRID", (0, 1), (-1, -1), 0.3, RULE),
    ]))
    return t


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="EarningsLens — User Guide",
        author="EarningsLens",
    )
    frame_cover = Frame(0, 0, A4[0], A4[1], id="cover",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_body = Frame(doc.leftMargin, doc.bottomMargin,
                       doc.width, doc.height, id="body")
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_cover], onPage=cover_page),
        PageTemplate(id="Body", frames=[frame_body], onPage=header_footer),
    ])

    story = []
    # Cover page is drawn by canvas; jump to body template
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    story.append(Paragraph("Welcome to EarningsLens", styles["ELH1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=10))
    story.append(Paragraph(
        "EarningsLens reads two consecutive earnings call transcripts from the same company and "
        "produces a concise, analyst-style brief. It runs five independent analyses in parallel — "
        "sentiment, hedging, topic drift, risk vocabulary, and Q&amp;A evasion — then synthesises "
        "the structured outputs into a short narrative.",
        styles["ELBody"]))
    story.append(Paragraph(
        "Everything runs locally on your machine using open-source Hugging Face models. Nothing is "
        "sent to a remote LLM. The output is an analytical aid: it is <b>not</b> investment advice.",
        styles["ELBody"]))

    story.append(Paragraph("What you will get", styles["ELH2"]))
    story.append(bullet("A headline <b>GREEN / AMBER / RED</b> signal summarising the quarter-over-quarter read.", styles))
    story.append(bullet("Five diagnostic tabs you can drill into independently.", styles))
    story.append(bullet("A short, written analyst brief that grounds itself in the structured outputs.", styles))
    story.append(bullet("Provenance for every analysis so you know whether it came from the main path or a fallback.", styles))

    story.append(Paragraph("The headline signal", styles["ELH2"]))
    story.append(signal_table())
    story.append(Paragraph(
        "Think of the signal as a triage layer. GREEN does not mean the quarter was strong — it "
        "means nothing in the language warranted special attention. RED does not mean the company "
        "is in trouble — it means several independent linguistic stress markers moved together "
        "and a human should look closely.",
        styles["ELCallout"]))

    # Setup
    story.append(PageBreak())
    story.append(Paragraph("Getting started", styles["ELH1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=10))
    story.append(Paragraph("1. Requirements", styles["ELH2"]))
    story.append(bullet("Python 3.11", styles))
    story.append(bullet("Around 1&nbsp;GB of disk for the first run (FinBERT ~440&nbsp;MB, MiniLM ~80&nbsp;MB, the local instruct model ~1&nbsp;GB).", styles))
    story.append(bullet("8&nbsp;GB of RAM is enough thanks to the lightweight default model (<i>SmolLM2 1.7B</i>).", styles))
    story.append(bullet("An Alpha Vantage API key, if you want to load real tickers directly inside the app.", styles))

    story.append(Paragraph("2. Configure your .env", styles["ELH2"]))
    story.append(Paragraph(
        "Copy <b>.env.example</b> to <b>.env</b> at the project root and fill in the two "
        "variables below. The file is read at startup; restart the app after editing it.",
        styles["ELBody"]))
    story.append(env_table())
    story.append(Paragraph(
        "MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct<br/>"
        "ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key",
        styles["ELCode"]))
    story.append(Paragraph(
        "Grab a free Alpha Vantage key at <i>alphavantage.co/support/#api-key</i>. Without it "
        "you can still use <b>Use sample</b> mode — only ticker search needs the key.",
        styles["ELCallout"]))

    story.append(Paragraph("3. Install and run", styles["ELH2"]))
    story.append(Paragraph(
        "python3.11 -m venv .venv<br/>"
        "source .venv/bin/activate&nbsp;&nbsp;&nbsp;<i># Windows: .venv\\Scripts\\activate</i><br/>"
        "pip install -r requirements.txt<br/>"
        "cp .env.example .env<br/>"
        "python scripts/make_samples.py<br/>"
        "streamlit run app.py",
        styles["ELCode"]))
    story.append(Paragraph(
        "On macOS you can also double-click <b>launch.command</b> from Finder. The first launch "
        "downloads the three models — expect a few minutes — and subsequent launches start in "
        "seconds thanks to the on-disk model cache.",
        styles["ELBody"]))

    story.append(Paragraph("4. Optional health checks", styles["ELH2"]))
    story.append(Paragraph(
        "./.venv/bin/pytest -q<br/>"
        "./.venv/bin/python scripts/health_check.py",
        styles["ELCode"]))
    story.append(Paragraph(
        "These verify that parsing, scoring, and model loading behave as expected before you "
        "trust any output.",
        styles["ELBody"]))

    # Using the app
    story.append(PageBreak())
    story.append(Paragraph("Using the app", styles["ELH1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("Input modes", styles["ELH2"]))
    story.append(Paragraph(
        "EarningsLens supports two input modes, chosen from the sidebar:",
        styles["ELBody"]))
    story.append(bullet(
        "<b>Search company.</b> Type an exact ticker, select a match, load the available quarters, "
        "and pick the two you want to compare. Transcripts are fetched via Alpha Vantage.",
        styles))
    story.append(bullet(
        "<b>Use sample.</b> Run on built-in demo pairs — Microsoft and Goldman Sachs Q-over-Q comparisons. "
        "Perfect for trying the tool offline or for grading without API calls.",
        styles))

    story.append(Paragraph("Sidebar controls", styles["ELH2"]))
    story.append(bullet("<b>Mode</b> — choose <b>Search company</b> to fetch real transcripts by ticker, or <b>Use sample</b> for built-in demo pairs.", styles))
    story.append(bullet("<b>Input selection</b> — use sidebar dropdowns to pick or search companies and compare quarters.", styles))
    story.append(bullet("<b>Show diagnostics</b> — toggle to surface parsed speaker roles, turn segmentation, evasion confidence scores, and analysis provenance.", styles))
    story.append(bullet("<b>Configuration</b> — thresholds and timeouts are set in <b>.env</b>, not the sidebar. Adjust "
                   "<b>ANALYSIS_TIMEOUT_SECONDS</b> or <b>EVASION_BATCH_SIZE</b> if needed.", styles))

    story.append(Paragraph("The five analyses", styles["ELH2"]))
    story.append(tab_table())

    story.append(Paragraph("Reading the analyst brief", styles["ELH2"]))
    story.append(Paragraph(
        "The brief at the top of the page is generated by the local instruct model from the "
        "structured outputs above — not from the raw transcripts. That keeps the synthesis "
        "anchored to explicit evidence and reduces the risk of hallucinated quotes. If a section "
        "fell back, the brief will say so via the provenance badge.",
        styles["ELBody"]))

    # Under the hood
    story.append(PageBreak())
    story.append(Paragraph("Under the hood", styles["ELH1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=10))
    story.append(Paragraph(
        "EarningsLens is intentionally layered so each signal can be inspected and trusted "
        "independently. Rule-based components are transparent; ML components are scoped narrowly.",
        styles["ELBody"]))

    story.append(Paragraph("Pipeline", styles["ELH2"]))
    story.append(pipeline_diagram(styles))

    story.append(Paragraph("Why this mix of methods", styles["ELH2"]))
    story.append(bullet("<b>Parsing</b> is regex + a small state machine. The transcript format is repetitive enough that rules are more reliable than an LLM.", styles))
    story.append(bullet("<b>Sentiment</b> uses FinBERT because general-purpose sentiment misreads financial language (e.g. <i>“flat margins”</i>).", styles))
    story.append(bullet("<b>Hedging &amp; risk vocab</b> are explicit lexicons. The signal is fully auditable and runs instantly on CPU.", styles))
    story.append(bullet("<b>Topics</b> mix MiniLM embeddings (broad drift) with the local LLM (human-readable themes).", styles))
    story.append(bullet("<b>Evasion</b> uses an LLM-as-judge rubric on a local model so no transcript ever leaves your machine.", styles))

    story.append(Paragraph("Local model", styles["ELH2"]))
    story.append(Paragraph(
        "The default local instruct model is <b>HuggingFaceTB/SmolLM2-1.7B-Instruct</b>. It is "
        "lightweight enough to run on an 8&nbsp;GB Mac while still capable on short structured "
        "tasks: theme extraction, Q&amp;A rubric scoring, and brief synthesis. You can swap it "
        "by editing the model id in <b>.env</b>.",
        styles["ELBody"]))

    # Troubleshooting
    story.append(PageBreak())
    story.append(Paragraph("Troubleshooting &amp; tips", styles["ELH1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("First run is slow", styles["ELH3"]))
    story.append(Paragraph(
        "That is expected — three models are being downloaded and cached. Subsequent runs reuse "
        "the cache and are much faster.",
        styles["ELBody"]))

    story.append(Paragraph(“An analysis shows a \”fallback\” badge”, styles[“ELH3”]))
    story.append(Paragraph(
        “The main model path exceeded the per-analysis timeout or raised an error, so a simpler “
        “fallback path was used. The headline signal still works; treat the fallback section as “
        “lower-confidence. To increase the timeout, set <b>ANALYSIS_TIMEOUT_SECONDS</b> in <b>.env</b> “
        “(default 300) and restart the app.”,
        styles[“ELBody”]))

    story.append(Paragraph("Q&amp;A pairs look mismatched", styles["ELH3"]))
    story.append(Paragraph(
        "Low-confidence pairs — moderator interjections, paste artefacts, or speaker role ambiguity — "
        "are automatically downweighted in the evasion scoring (minimum confidence threshold 0.55). "
        "Enable <b>Show diagnostics</b> to inspect pair confidence scores.",
        styles["ELBody"]))

    story.append(Paragraph("Alpha Vantage rate limits", styles["ELH3"]))
    story.append(Paragraph(
        "The free tier is limited. If a fetch fails, wait a minute and retry, or fall back to "
        "the sample mode while you iterate.",
        styles["ELBody"]))

    # Closing
    story.append(Paragraph("A note on responsible use", styles["ELH2"]))
    story.append(Paragraph(
        "EarningsLens surfaces <i>linguistic</i> signals — sentiment shifts, hedging creep, "
        "topic drift, low-responsiveness Q&amp;A exchanges. None of these prove anything on "
        "their own. They are starting points for human reading, not conclusions. The banner at "
        "the top of the app makes the same point: the output is an analytical aid, not "
        "investment advice.",
        styles["ELCallout"]))

    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "EarningsLens · User Guide · v1.0 · Generated locally — no transcript data leaves your machine.",
        styles["ELFooter"]))

    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
