# AI tools used during development

This file documents how we used third-party AI coding assistants (Claude Code, ChatGPT, GitHub Copilot) while building EarningsLens. The assignment requires us to show where and how such tools contributed to the project.

All prompts below were issued by a human team member. Generated code was reviewed, edited, and tested before being committed.

---

## 1. Scaffolding the transcript parser

**Tool**: Claude (Sonnet / Opus) via Claude Code.

**Prompt**
> "Earnings call transcripts use a 'Speaker Name: text' format, sometimes with a title after a dash. Help me write a Python module that splits a raw transcript into ordered speaker turns, separates the prepared-remarks section from the Q&A section, and infers each speaker's role (executive / analyst / operator / unknown) with a confidence score. Use the regex module only — no LLM at parse time."

**Outcome**: produced the first draft of `earningslens/parser.py`. We then iterated on edge cases (multi-line speaker prefixes, missing Q&A boundary markers, merged-line transcripts).

**Follow-up prompts**
> "Some transcripts repeat the speaker name across paragraphs. Add a post-processing pass that merges runs of consecutive turns from the same speaker so the downstream evasion scorer sees one continuous answer."

> "I have a sample where the analyst's question opens with 'A separate question.' followed by the actual question. Where in the pipeline should this preamble be stripped, and what regex would cover similar openers ('One quick one,' 'Just a follow-up,' 'My last question,')?"

---

## 2. Designing the LLM-as-judge rubric for Q&A evasion

**Tool**: ChatGPT (GPT-4 class) and Claude.

**Prompt**
> "I want to score how responsively an executive answered an analyst question on an earnings call. Draft a short rubric that asks a small local LLM to output a single integer score 0-10 plus a one-sentence justification. The rubric should penalize hedging and pivoting to adjacent topics but not penalize legitimate deferral to future guidance."

**Outcome**: produced the first draft of `EVASION_SCORING_PROMPT` in `earningslens/prompts.py`. We then constrained the output format to two lines (`SCORE: N` and `WHY: <sentence>`) so a regex parser could extract the score deterministically.

**Follow-up prompts**
> "The model sometimes echoes the answer back as the WHY line. Suggest detection heuristics so we can flag echoed reasoning and replace it with a deterministic explanation derived from the question's sub-parts."

> "Outline a fallback that, when the LLM response is unparseable, scores the pair from a content-overlap heuristic instead. The fallback must never falsely produce a perfect score."

---

## 3. Performance optimization

**Tool**: Claude Code.

**Prompt**
> "Profile the evasion scoring loop. Each Q&A pair triggers one `model.generate()` call locked by a global mutex, and we run ~25 pairs per transcript. Suggest concrete changes that would cut wall-clock time without losing scoring quality."

**Outcome**: recommendations included
1. Batch prompts and call `model.generate()` once per batch (now wired through `EVASION_BATCH_SIZE`).
2. Drop `max_new_tokens` from 160 to 64 since the expected output is two short lines.
3. Stop calling `torch.mps.empty_cache()` after every generation; defer to end of phase.
4. Switch `torch.no_grad()` to `torch.inference_mode()`.

We implemented all four.

---

## 4. Removing silent fallback data

**Prompt**
> "Audit the codebase for places where an analysis silently returns benign-looking defaults (zeros, similarity=1.0) on error. Replace these with explicit error surfacing so a failed analysis halts the app instead of producing a misleadingly clean dashboard."

**Outcome**: removed `_fallback_hedging`, `_fallback_topics`, `_fallback_risk_vocab` from `app.py` and rewired `record_analysis_failure` to always append to `analysis_errors` and stop execution with a visible error block.

---

## 5. User guide and launchers

**Prompt**
> "Generate a polished PDF user guide with a cover page, color-coded GREEN/AMBER/RED signal table, a pipeline diagram, an environment-variable reference table, and a troubleshooting section. Use ReportLab."

**Outcome**: `scripts/make_user_guide_pdf.py`. We then iterated on layout and content as the product evolved.

**Prompt**
> "Write a Windows-equivalent of the existing macOS `launch.command`. The script should: cd into its own directory, verify `.venv` exists, activate it, verify Streamlit is importable, warn (non-fatally) if the model cache is missing, then run `streamlit run app.py`. Keep the window open if any step fails so the user can read the error."

**Outcome**: `launch.bat`.

---

## 6. Documentation

**Prompts** (paraphrased)
> "Rewrite this README section so a reviewer can run the app in under five minutes."

> "Draft a one-page mapping between the project and the course concepts from Introduction to AI for Business."

> "Cite the academic and library sources we used: FinBERT, MiniLM, Sentence-Transformers, LLM-as-judge methodology, hedging lexicon origins, SmolLM2 model card."

Outputs were reviewed and edited.

---

## What the human team did, end-to-end

- Project framing and scope
- Choice of analyses (sentiment gap, hedging, topic drift, risk vocab, Q&A evasion) and threshold tuning on real transcripts
- Architectural decisions: local-only execution, low-memory mode, fallback isolation, batched generation
- All code review and test design
- Final wording in the user guide and README
- All data collection, sample preparation, and verification

The AI assistants were used to accelerate boilerplate (regex drafts, rubric drafts, PDF layout, performance audits, documentation drafts), not to make product decisions or replace verification.
