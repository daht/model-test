# ASR Acceptance Gates

## Select the gate by decision being made

| Decision | Required command/evidence | Workspace |
| --- | --- | --- |
| Inspect available gates | `scripts/verify_asr_release.sh --list-gates live` | Any repository checkout |
| Review an execution plan | `scripts/verify_asr_release.sh --dry-run <mode>` | Any repository checkout |
| Accept a code candidate | `scripts/verify_asr_release.sh commit` | Same worktree; fresh checkout not required |
| Promote an image | `scripts/verify_asr_release.sh release` | Clean release checkout with Docker, GPU, `.env`, approved model and manifest |
| Accept a deployment | `scripts/verify_asr_release.sh live` | Release-capable environment plus deployed URLs, runtime key, external zh/ja speech, thresholds and ffmpeg tools |

Read `docs/asr-release-verification.md` before release/live execution. It is the
authoritative environment-variable and evidence reference.

## Development gate

Before requesting independent review:

1. Stage only intended files with `git add -- <paths>`.
2. Run deterministic focused regression tests and observe the pre-fix failure
   when implementing a bug fix.
3. Run `scripts/verify_asr_release.sh commit`.
4. Only after that gate passes, commit the candidate and confirm the worktree is clean relative to the
   candidate. Never hide user-owned changes to manufacture cleanliness.

Running the development gate only after committing is invalid because staged
binary, size, and intended-delta checks would no longer see the candidate.

Do not invent generic `make test`, TSAN, ASAN, arbitrary loop counts, or alternate
verification commands unless the repository actually defines them or the root
cause specifically requires an additional probe.

For Qwen runtime, backend, dependency, model-ID, model-path, or deployment
changes, explicitly verify invariant 13 from `project-contract.md`. A passing
manifest hash check is not model/runtime compatibility evidence. The real R08
warmup is required before promotion, and an `audio_token` tokenizer failure
must be investigated as a model-export mismatch before changing GPU or
streaming settings.

## Independent acceptance

The Test Agent verifies the exact committed SHA and does not modify files. At a
minimum it must:

- reproduce the original failure or prove the regression test is sensitive to
  the pre-fix behavior;
- exercise relevant invariant and adversarial paths;
- run the applicable focused suite and the commit runner;
- check secrets, forbidden artifacts, background processes, temporary residue,
  and final Git status;
- distinguish missing external prerequisites from executed passes;
- return `ACCEPTED` only with no open Critical, High, or Medium finding.

The primary agent prepares any isolated/detached worktree before dispatch. The
read-only Test Agent does not create or remove worktrees.

Warnings and unexecuted external gates remain visible in the report.

## Product acceptance

After Test Agent acceptance, the primary agent independently:

1. Confirms the accepted SHA and clean candidate worktree.
2. Runs fresh relevant user workflow probes, not only unit tests.
3. Runs the commit runner again.
4. Confirms protocol terminal behavior, cleanup, repository boundaries, and no
   secret exposure for the affected surface.
5. Runs release/live only when every prerequisite exists. Missing prerequisites
   must fail closed and must be reported as unexecuted.

## Supplemental production evidence

The current live runner validates protocol, two-stream concurrency, completion
overhead, and sampled VRAM. It does not by itself establish higher concurrency,
CER/WER, sentence-boundary accuracy, multi-service GPU headroom, or long-duration
stability. Require separately approved corpus, capacity, soak, or canary evidence
when the rollout decision depends on those claims.

## Failure handling

- Stop at the earliest failed gate. Keep the failed evidence outside the repo.
- Do not bypass a gate, combine partial runs, regenerate approval data from a
  suspect target, or adjust a threshold after seeing a failure.
- Return implementation findings to the same Development Agent.
- A deployed live failure requires atomic rollback of code/image, configuration,
  model directory, and matching approved manifest, followed by readiness and live
  verification of the restored deployment.
