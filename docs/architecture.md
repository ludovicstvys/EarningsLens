# EarningsLens Architecture

```text
two transcripts (.txt) — current quarter, prior quarter
        │
        ▼
   parser.py  ──►  Transcript × 2
        │
        ├──► sentiment.py  ──►  SentimentAnalysis
        ├──► hedging.py    ──►  HedgingAnalysis
        ├──► topics.py     ──►  TopicDrift
        ├──► risk_vocab.py ──►  list[RiskVocabItem]
        └──► evasion.py    ──►  list[EvasionScore]
                                      │
                                      ▼
                            synthesizer.py
                                      │
                                      ▼
                              EarningsBrief
                                      │
                                      ▼
                                  app.py
```

`parser.py` converts raw transcript text into structured `SpeakerTurn` objects with section and role labels. It uses regex plus a lightweight state machine rather than an LLM because the format is repetitive and deterministic enough for rules to work well.

`sentiment.py` uses the finance-tuned FinBERT transformer (`ProsusAI/finbert`) to score prepared remarks and Q&A separately. That creates a tone gap signal based on domain-specific sentiment rather than a general-purpose language model.

`hedging.py` and `risk_vocab.py` use explicit vocabularies plus normalized counting. This is the statistical layer of the system: it is transparent, fast on CPU, and easy to explain to graders because the signal comes from measurable language shifts.

`topics.py` combines sentence embeddings from `all-MiniLM-L6-v2` with local LLM theme extraction. The embedding similarity captures broad semantic drift, while the local Hugging Face instruct model extracts human-readable themes that can be compared quarter over quarter.

`evasion.py` uses an LLM-as-judge rubric to score whether executive answers actually responded to analyst questions. Here the judge is a local Hugging Face instruct model rather than a remote API.

`synthesizer.py` takes the structured outputs from the five analyzers and asks the local Hugging Face instruct model to write a concise analyst brief. The final output is grounded in prior structured results rather than generated from transcripts directly, which keeps the synthesis anchored to explicit evidence.
