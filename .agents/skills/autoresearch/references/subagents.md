# Native Codex research orchestration

Use native Codex subagents for bounded, parallel evidence work. The lead remains responsible for the research direction and must not delegate candidate promotion or benchmark interpretation.

## Roles

- `lead`: defines the objective and workstreams, checks coverage, and chooses whether a synthesis is actionable.
- `researcher`: finds and extracts evidence for one distinct search angle.
- `skeptic`: searches for counterevidence, missing assumptions, and source-quality problems. It must not share a workstream with the researcher whose claims it reviews.
- `synthesizer`: reconciles findings and critiques into a falsifiable hypothesis with explicit uncertainty. It cannot promote a candidate.

Keep workstreams independent and bounded. Use no more than `orchestration.max_parallel_subagents` from `research.yaml`. Each delegation must specify a question, role, acceptance criteria, and the JSON fields to return. Subagents may read repository evidence but must not edit candidates, run benchmarks, change infrastructure, or append canonical events.

## Research packet

The lead assembles subagent outputs into one JSON object and passes it to `controller.py ingest-research`:

```json
{
  "schema_version": 1,
  "plan": {
    "plan_id": "plan-example",
    "objective": "Identify a falsifiable optimization direction",
    "lead_role": "lead",
    "contract_hash": "<confirmed research.yaml sha256>",
    "workstreams": [
      {
        "workstream_id": "ws-kernels",
        "role": "researcher",
        "question": "Which implementation mechanisms apply to the official shapes?",
        "acceptance_criteria": ["primary source", "mechanism maps to an official shape"]
      },
      {
        "workstream_id": "ws-counter",
        "role": "skeptic",
        "question": "What evidence would invalidate the proposed mechanisms?",
        "acceptance_criteria": ["independent source", "explicit failure mode"]
      }
    ]
  },
  "sources": [
    {
      "source_id": "src-1",
      "workstream_id": "ws-kernels",
      "agent_role": "researcher",
      "connector": "exa",
      "url": "https://example.com/primary-source",
      "title": "Primary source",
      "retrieved_at": "2026-08-31T00:00:00Z",
      "quality": "primary"
    }
  ],
  "findings": [
    {
      "finding_id": "finding-1",
      "workstream_id": "ws-kernels",
      "agent_role": "researcher",
      "claim": "A concrete mechanism may improve a named shape class.",
      "source_ids": ["src-1"]
    }
  ],
  "critiques": [
    {
      "critique_id": "critique-1",
      "workstream_id": "ws-counter",
      "agent_role": "skeptic",
      "finding_ids": ["finding-1"],
      "verdict": "needs_more_evidence",
      "rationale": "The source does not measure this benchmark contract.",
      "source_ids": []
    }
  ],
  "synthesis": {
    "synthesis_id": "synthesis-1",
    "agent_role": "synthesizer",
    "hypothesis": "A falsifiable candidate hypothesis, not a promotion claim.",
    "finding_ids": ["finding-1"],
    "critique_ids": ["critique-1"],
    "source_ids": ["src-1"],
    "uncertainties": ["No live benchmark evidence yet"]
  }
}
```

Identifiers must be unique inside the packet. Findings need at least one source. Critiques may cite no additional source when they challenge inference rather than introduce external evidence. The synthesis must cite at least one finding and one critique.

After deterministic ingestion, the lead reads the recorded synthesis and decides whether to state a candidate hypothesis. Continue to the normal research cycle; do not treat the packet as runtime proof.
