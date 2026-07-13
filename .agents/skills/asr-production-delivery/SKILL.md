---
name: asr-production-delivery
description: Use when working in this repository on ASR, VAD, streaming transcription, sentence_final/final protocol behavior, Qwen ASR runtime, ASR configuration, release verification, production rollout, or ASR incident fixes.
---

# ASR Production Delivery

## Principle

Require deterministic regression, a committed candidate, independent acceptance,
product regression, and applicable deployment gates. Missing means gap, not pass.

## Start Here

1. Confirm the repository contains:
   `scripts/verify_asr_release.sh` and `docs/asr-release-verification.md`.
   Stop otherwise.
2. Preserve pre-existing Git changes. Use an isolated feature worktree when the
   current workspace is dirty.
3. Read `references/project-contract.md` for every task.
4. Read `references/acceptance-gates.md` when code, configuration, deployment,
   verification, or rollout is in scope.
5. Read `references/agent-handoffs.md` before delegating implementation or
   acceptance.

## Route The Request

- Explanation, review, or plan only: do not edit or start delivery.
- Implementation or bug fix: run the complete workflow below.
- Verification only: remain read-only; report the SHA and unavailable gates.
- Production release or incident: require real release/live evidence. Mock
  evidence cannot establish production readiness.

## Implementation Workflow

1. Publish an execution contract: objective, non-goals, risk, paths, tests,
   gates, prerequisites, and 30/60-minute checkpoints.
2. Dispatch exactly one Development Agent to implement with test-first evidence
   and focused tests. It must explicitly stage intended files, run
   `scripts/verify_asr_release.sh commit` while they are staged, then commit. It
   owns the change through all review corrections.
3. Dispatch a separate read-only Test Agent against the committed SHA. It must
   return `ACCEPTED` or `REJECTED` with reproducible evidence.
4. On rejection, send the findings back to the same Development Agent. Do not
   silently replace it or let the primary agent patch the code. Re-run the
   independent Test Agent after the new commit.
5. After two rejected cycles, stop edits for a primary-agent architecture,
   scope, and budget review. Continue only with a corrected contract.
6. Only after acceptance, the primary agent runs fresh product regression using
   repository commands, not agent claims or stale output.
7. Report SHA, evidence, missing gates, risk, and integration state. Never say
   production-ready until required release and live gates pass.

## Control Cost Without Weakening Evidence

- At 30 minutes without a supported root cause, restate evidence and hypotheses.
  At 60 minutes without a testable commit, split or rescope.
- Use deterministic event/barrier tests before stress loops. Never substitute
  sleeps or repeated green runs for a synchronization proof.
- Run focused tests during development and one full gate per final candidate.
  Save long logs outside the repository and return summaries.
- Give subagents only the contract, worktree, SHA, paths, and failures.
- Time or token exhaustion is never completion evidence.
- Enforce a token budget only when the user set one and the platform exposes
  metering. Otherwise use time/context checkpoints; never invent token counts.

## Non-Negotiable Boundaries

Never put runtime keys in prompts, argv, code, docs, configs, evidence names, or
reports. Never track audio, model weights, manifests generated from untrusted
targets, verification output, caches, or `superpowers/`. Never revert unrelated
user changes. Never bypass, weaken, or retroactively raise a failed threshold.
