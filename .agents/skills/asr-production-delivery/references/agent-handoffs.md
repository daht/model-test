# ASR Agent Handoffs

Pass compact task-local context. Never include runtime credentials, full chat
history, irrelevant diffs, or complete successful logs.

## Development Agent handoff

Provide:

```text
Role: sole Development Agent and owner through review corrections
Repository/worktree: <absolute path>
Base SHA: <sha>
Objective: <one concrete behavior>
Non-goals: <explicit exclusions>
Relevant paths: <small list>
Required invariants: <IDs or exact statements from project-contract.md>
Required RED evidence: <reproduction or regression test>
Required gates: focused tests; git add -- <intended paths>; commit runner while staged; then commit
Boundaries: preserve user changes; no secrets/audio/models/superpowers; commit result
Return: status, root cause, files, tests with counts, commit SHA, external gaps
```

Require one of `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`.
Do not accept a completion claim without a commit and fresh command evidence.

## Test Agent handoff

Dispatch only after the Development Agent commits a candidate.

```text
Role: independent read-only ASR Test Agent
Repository/worktree: <absolute path>
Candidate SHA: <exact sha>
Base SHA: <sha>
User-visible failure: <original symptom>
Required invariants and acceptance criteria: <exact list>
Required probes: <focused/adversarial tests plus commit runner>
External prerequisites expected here: <present/missing>
Restrictions: do not edit, stage, commit, install into repo, or expose secrets
Return first line: ACCEPTED or REJECTED
Then: findings by severity, commands/results, external gaps, final SHA/status
```

`REJECTED` returns to the same Development Agent with the exact findings. Use a
follow-up task so its implementation context remains available. After a new
commit, dispatch the Test Agent again against the new SHA.

## Primary product acceptance

Do not begin before the independent Test Agent returns `ACCEPTED`.

Checklist:

```text
[ ] Candidate SHA matches the accepted SHA
[ ] Original user workflow is exercised
[ ] Relevant protocol and failure paths are exercised
[ ] scripts/verify_asr_release.sh commit passes freshly
[ ] Required release/live gates pass, or are explicitly unexecuted
[ ] Git status and artifact/secret boundaries are unchanged
[ ] No required process or temporary verification directory remains
[ ] Final report distinguishes local, release, live, and production status
```

The primary agent must not repeat an agent's success claim without running fresh
evidence. It may reject the candidate and return it to the same Development
Agent if product behavior still fails.

## Checkpoints and escalation

- 30 minutes without supported root cause: Development Agent returns a compact
  evidence/hypothesis summary before more edits.
- 60 minutes without a testable commit: primary agent pauses and reduces scope or
  corrects the contract.
- Two rejected candidate cycles: primary agent performs an architecture, scope,
  and budget review before another implementation turn.
- Do not silently swap the Development Agent merely because work is difficult.
  Replace it only after an explicit re-plan caused by a real blocker or invalid
  architecture, while retaining the original agent for factual handoff.
- Prefer deterministic event/barrier control for concurrency. Set stress counts
  from observed failure rate and runtime; do not invent fixed large counts.
- Set numeric token budgets only when the user explicitly requests one and the
  platform exposes metering. Otherwise report elapsed checkpoints and context
  growth without fabricating token usage or enforcement.
