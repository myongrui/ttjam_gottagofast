# Exa evidence protocol

Use the installed Exa Search app's `web_search_exa` and `web_fetch_exa` tools for every external web search and source fetch in this workflow. If either tool is unavailable, unauthenticated, or rate-limited, stop the external investigation and report the connection problem; do not silently switch connectors.

Before searching, translate each workstream into a distinct search angle. Avoid synonym-only queries. Researchers search for supporting mechanisms and primary technical evidence; skeptics independently search for counterexamples, limitations, and benchmark mismatches.

For every retained source:

- fetch the source page rather than relying on a search snippet;
- record the canonical HTTP(S) URL, title, UTC retrieval time, and workstream;
- classify quality as `primary`, `official`, `paper`, `practitioner`, or `secondary`;
- state the exact claim it supports and do not extend that claim beyond the source;
- deduplicate repeated URLs before building the packet.

Prefer official documentation, source code, papers, and first-party engineering reports. Search relevance is discovery evidence, not validation. A source about similar hardware or a different benchmark may motivate a hypothesis, but must be identified as indirect.

Never put scores, timing measurements, correctness results, promotion decisions, or hardware-performance conclusions from web sources into benchmark fields. External sources remain `research_evidence`; only repository harness output can become `live_runtime_evidence`.
