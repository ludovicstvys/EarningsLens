# Sample transcripts — sources and usage

This directory contains four publicly available earnings call transcripts used as built-in demo material for EarningsLens. They are included so reviewers can run the application end-to-end without an Alpha Vantage API key.

## Files

| File | Company | Period | Call date |
|---|---|---|---|
| `GS_Q4_2025.txt` | The Goldman Sachs Group, Inc. (NYSE: GS) | Fourth quarter 2025 | 15 January 2026 |
| `GS_Q1_2026.txt` | The Goldman Sachs Group, Inc. (NYSE: GS) | First quarter 2026 | 13 April 2026 |
| `MSFT_Q4_2025.txt` | Microsoft Corporation (NASDAQ: MSFT) | Fiscal year 2025 Q4 (Apr–Jun 2025) | 30 July 2025 |
| `MSFT_Q1_2026.txt` | Microsoft Corporation (NASDAQ: MSFT) | Fiscal year 2026 Q1 (Jul–Sep 2025) | 29 October 2025 |

Each file is a plain-text transcript of the public earnings conference call, with operator, executive and analyst turns preserved in the original speaker order.

## Where the transcripts come from

Both Goldman Sachs and Microsoft host their earnings call audio replays and prepared remarks on their respective Investor Relations websites:

- Goldman Sachs Investor Relations — `https://www.goldmansachs.com/investor-relations/`
- Microsoft Investor Relations — `https://www.microsoft.com/en-us/Investor`

Transcripts were assembled by listening to the public audiocast / reading the press releases linked from those pages, and cross-checked against secondary sources (Seeking Alpha, The Motley Fool) where helpful. No transcript here was obtained from behind a paywall.

## Copyright and intended use

The underlying conference calls are copyrighted by the issuing company. Each transcript file preserves the company's own disclaimer text, for example:

> *"This audiocast is copyrighted material of the Goldman Sachs Group, Inc. and may not be duplicated, reproduced, or rebroadcast without consent."*

These files are reproduced here for **non-commercial educational use** as part of a graduate course assignment, on the same fair-use basis that financial analysts routinely quote earnings call material in published research notes. The files are not redistributed for any commercial purpose.

If you reuse this repository outside an educational context, please remove the sample transcripts from `data/samples/` and pull fresh transcripts at runtime via the Alpha Vantage integration (`Search company` mode in the app).

## Reproducibility notes

- Transcripts were not edited for content; only minor whitespace normalisation was applied so the parser sees consistent line breaks.
- Speaker prefixes follow the formats `Speaker Name - Title:` (Microsoft) and `Speaker Name: ` or `Speaker Name - Title:` (Goldman Sachs).
- Numbers, dates and company names in the files are exactly as spoken on the call.

## Data citation

When citing this dataset in academic work, please cite the issuing company and the call date — for example:

> The Goldman Sachs Group, Inc. (2026, April 13). *First Quarter 2026 Earnings Conference Call* [transcript]. Investor Relations, goldmansachs.com.

> Microsoft Corporation. (2025, October 29). *FY26 Q1 Earnings Conference Call* [transcript]. Investor Relations, microsoft.com.
