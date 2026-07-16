# ASR Full-Chain Observability Design

## Status

Approved design for the repository-native ASR observability implementation.

## Objective

Produce one self-contained evidence archive for every ASR test run so an
operator can determine, without reproducing the workload blindly:

- how client audio moved through Gateway buffering, scheduling, adapter
  grouping, faster-whisper inference, result publication, and cleanup;
- why scheduler batches fragmented and how scheduler batch size differed from
  the actual faster-whisper engine group size;
- which event preceded a BufferError, lag failure, timeout, transcript conflict,
  worker failure, or leaked session;
- whether HY-MT GPU residency or resource activity correlated with an ASR
  inference tail;
- whether every terminal session released PCM, reservations, backend state, and
  its worker lease.

The implementation remains repository-native: structured application logs,
the authenticated JSON metrics endpoint, the existing monitor script, and a
standard-library analyzer. It does not require Prometheus, Loki, Grafana, or an
external tracing service.

## Scope

The observed request path is:

```text
streaming client
  -> WebSocket/Gateway
  -> per-session PCM buffer and reservation
  -> GatewayScheduler
  -> FasterWhisperAdapter grouping
  -> CTranslate2 engine
  -> protocol/result publication
  -> terminal cleanup and lease release
```

The monitor also records HY-MT container and GPU-process resource use. It does
not collect HY-MT request or response content, and it does not instrument TTS.

## Non-goals

- Do not add a general-purpose logging framework or observability platform.
- Do not store audio, PCM, transcripts, prompts, model tokens, API keys, or
  exception messages from untrusted input.
- Do not expose per-session or per-job details through the metrics endpoint.
- Do not add a runtime API that changes logging configuration.
- Do not change scheduling, inference, buffering, or protocol behavior as part
  of this work.
- Do not claim a root cause from a threshold alone. Reports present correlated
  evidence and evidence-quality gaps.

## Operating modes

Observability has two levels.

### Always-on

Low-frequency lifecycle, failure, cleanup, worker-state, buffer-rejection, and
slow-engine events are always emitted. These events are suitable for normal
service operation.

### Diagnostic

`ASR_DIAGNOSTIC_LOGGING=true` enables high-frequency audio, job, batch, engine,
buffer, and result events. This setting is read at process startup. Enabling it
requires the normal environment update and service recreation; there is no
runtime mutation endpoint.

Diagnostic mode is intended for test and incident windows. Disabling it removes
high-frequency events without removing failure evidence.

## Structured event schema

Every application event is a single JSON object on one log line. The stable
base schema is:

```json
{
  "schema_version": 1,
  "timestamp": "2026-07-16T15:30:00.123Z",
  "event": "asr_engine_group_completed",
  "level": "INFO",
  "component": "faster_whisper_adapter",
  "process_id": "a process-start UUID",
  "worker_id": "local",
  "session_id": null,
  "generation": null,
  "job_id": null,
  "batch_id": "a stable scheduler-batch identity",
  "engine_call_id": "a per-engine-call identity"
}
```

Fields not relevant to an event are omitted rather than populated with invented
values. Numeric durations use seconds and end in `_seconds`. Sample counts end
in `_samples`. Byte counts end in `_bytes`.

### Correlation identities

- `process_id` is generated once at process startup.
- `session_id`, `generation`, and `job_id` retain their current runtime values.
- `batch_id` is deterministically derived from the ordered accepted job
  identities, so scheduler and adapter can calculate the same identity without
  changing the adapter protocol.
- `engine_call_id` combines `batch_id` with the adapter group ordinal. A
  scheduler batch that splits by language or beam therefore produces multiple
  correlated engine calls.

The batch identity helper belongs in the observability module and is used by
both scheduler and adapter. It does not mutate an `InferenceJob`.

## Event ownership

### Always-on events

| Event | Owner | Required fields |
| --- | --- | --- |
| `asr_process_started` | Gateway runtime | backend, stream mode, batch capacity, diagnostic enabled |
| `asr_session_opened` | Gateway runtime | session, generation, worker, language, active sessions |
| `asr_session_terminal` | Gateway runtime | state, controlled reason, close code |
| `asr_session_released` | Gateway runtime | remaining buffered/reserved/pending samples, lease/backend cleanup status |
| `asr_buffer_rejected` | rejection site | reason, limit/current/incoming samples or jobs |
| `asr_worker_state_changed` | backend registry/runtime | previous state, next state, controlled reason |
| `asr_engine_slow_call` | adapter | batch and engine identities, beam, group size, accumulated audio range, elapsed time |
| `asr_cleanup_conflict` | Gateway cleanup | job identity, expected/current generation, reservation presence |

### Diagnostic events

| Event | Owner | Required fields |
| --- | --- | --- |
| `asr_audio_ingested` | Gateway runtime | incoming, buffered, reserved, pending-VAD and accepted samples |
| `asr_job_enqueued` | Gateway runtime | job identity, chunk samples, final, length bucket, BatchKey fingerprint, queue gauges |
| `asr_batch_dispatched` | scheduler | batch identity, compatible depth, selected size, capacity, queue depth, audio cost, wait, dispatch reason |
| `asr_engine_group_started` | adapter | batch and engine identities, group ordinal/count, language identity, beam, group size, accumulated audio min/max/sum |
| `asr_engine_group_completed` | adapter | start fields plus elapsed time and output size/repetition summaries |
| `asr_job_cleaned` | Gateway cleanup | acknowledge/rollback/stale/cancel outcome and remaining accounting |
| `asr_result_published` | Gateway runtime | result mode, final, emitted event types, application/egress status |
| `asr_buffer_snapshot` | Gateway runtime | per-session held PCM plus aggregate queue state at a scheduling transition |

No event contains transcript text. Output summaries contain only token count when
available, character count, maximum same-token or same-character run, and a
numeric compression/repetition score.

## Buffer rejection taxonomy

Generic BufferError messages are insufficient for diagnosis. Capacity
exceptions remain subclasses of `BufferError` so current API handling remains
compatible, but each rejection carries one controlled reason:

- `session_pcm_limit`
- `scheduler_ready_job_limit`
- `scheduler_queued_audio_limit`
- `adapter_utterance_limit`

The rejection site records safe numeric fields before raising. The Gateway
terminal event repeats only the controlled reason, not the exception message.

## Application metrics

The existing `/v1/asr/metrics` response remains backward compatible. New data
uses bounded in-memory windows and low-cardinality keys.

### Scheduler metrics

- dispatched batch count;
- scheduler batch size count, min, p50, p95, and max;
- batch wait count, p50, p95, p99, and max;
- compatible depth and selected depth summaries;
- fragmentation counters for BatchKey, arrival window, session uniqueness,
  sample limit, language, partial/final identity, and beam grouping;
- ready, queued, accepted/in-flight, and cleanup-pending gauges.

### Engine metrics

- actual engine call count;
- actual engine group size count, min, p50, p95, and max;
- calls per scheduler batch;
- inference time p50, p95, p99, and max, separated into partial and final beam
  classes;
- accumulated audio min/max/sum distributions;
- output token/character count and repetition-score distributions;
- slow-call counter.

### Buffer and lifecycle metrics

- BufferError counters keyed by the four controlled reasons;
- aggregate and per-session held-PCM p50, p95, and max;
- session open, terminal, release, cancellation, and cleanup-conflict counters;
- current sessions with buffered PCM, active reservations, queued jobs,
  accepted/in-flight jobs, and pending cleanup;
- unreleased lease and backend-session gauges.

Percentiles use a deterministic nearest-rank implementation over a bounded
window. Empty windows return a count of zero and omit percentiles rather than
reporting fabricated zeros.

## Logging module

`app/asr_observability.py` owns:

- schema version, process identity, and UTC timestamp generation;
- diagnostic-mode evaluation;
- deterministic batch and engine-call identities;
- JSON serialization;
- field-name and value validation;
- sensitive-field rejection;
- bounded numeric summary helpers shared by metrics and the analyzer fixtures.

The module accepts only JSON scalars, bounded lists of controlled values, and
known identifiers. Field names containing `key`, `secret`, `token_text`,
`transcript`, `prompt`, `pcm`, `audio_path`, or `exception_message` are rejected.
Model token counts are allowed as numeric values; token values are not.

## Monitor collection

`scripts/monitor_asr_bottleneck.sh` remains the operator entry point. Each run
creates a new identity and never overwrites an earlier run.

```text
/tmp/asr-monitor/
  .asr-monitor-owned
  runs/
    20260716T153000Z-a1b2c3/
      metadata.json
      config-safe.json
      events.jsonl
      gateway-metrics.jsonl
      backends.jsonl
      readiness.jsonl
      gpu.csv
      gpu-processes.csv
      docker-stats.csv
      collector-errors.log
      report.json
      report.md
      manifest.sha256
    20260716T153000Z-a1b2c3.tar.gz
```

### Collectors

- ASR structured service log, continuously;
- `/ready`, `/v1/asr/metrics`, and `/v1/asr/backends`, every 0.5 seconds;
- GPU utilization, memory utilization, memory use, power, temperature, pstate,
  and clocks, every 0.5 seconds;
- GPU process PID and memory use, mapped to ASR and HY-MT containers;
- Docker CPU, memory, network, block I/O, and PID counts for ASR and HY-MT,
  every second;
- container/image identity and safe configuration at start and finish;
- collector failures and missed/late sampling intervals.

HY-MT business logs and content are not collected. Its container and GPU
resource use are collected only to establish contention.

### Coordination and finalization

- A lock prevents concurrent monitor processes from owning the same output
  root.
- Every collector writes only inside its run directory.
- Ctrl+C, a collector failure, or a test failure invokes the same finalize path.
- Finalization stops collectors, validates evidence, runs the analyzer, writes
  a SHA-256 manifest, scans for secrets, writes a temporary archive, and
  atomically renames it.
- A failed secret scan refuses to create an archive.

## Retention

Each run is retained independently. Defaults are:

- maximum 20 completed runs;
- maximum age 14 days.

Both limits are configurable. Cleanup removes only completed run directories
and archives under a marked output root whose names match the monitor's strict
UTC/run-ID pattern. It refuses symlinks, unmarked roots, partial paths, and broad
globs.

## Automated analyzer

`scripts/analyze_asr_bottleneck.py` uses only the Python standard library. It
reads one completed run and produces `report.json` plus a human-readable
`report.md`.

### Report sections

1. Evidence identity and quality
   - run/process/container/image identities;
   - observed interval and sample counts;
   - missing files, collector errors, time gaps, or timestamp disorder.
2. Workload and lifecycle
   - peak/open/terminal/released sessions;
   - incomplete lifecycles and unreleased ownership.
3. Scheduler behavior
   - scheduler batch distribution;
   - compatible versus selected depth;
   - fragmentation reasons and worst intervals.
4. Engine behavior
   - scheduler-to-engine batch expansion;
   - partial/final latency distributions;
   - slowest 20 engine calls and their audio/output summaries.
5. Buffer and failures
   - each rejection by controlled reason;
   - the preceding 30-second buffer, queue, scheduler, engine, GPU, and
     container timeline.
6. Resource correlation
   - ASR/HY-MT GPU process memory;
   - GPU utilization and power around engine tails;
   - ASR/HY-MT container resource changes.
7. Evidence-backed observations
   - exact correlated facts;
   - alternative explanations still compatible with the evidence;
   - missing evidence required to distinguish them.

The analyzer does not encode mutable production thresholds as root-cause
truth. For example, it reports that an engine call preceded buffer growth and
shows the timestamps; it does not claim the engine caused the growth when
sampling gaps prevent that conclusion.

## Example diagnosis

```text
Buffer rejection at 2026-07-16T15:32:41.220Z
  reason: session_pcm_limit
  held PCM: 5.992 / 6.000 seconds
  preceding scheduler batch: 5 items
  actual engine groups: 2 and 3 items
  split dimension: beam
  slowest preceding engine call: 4.830 seconds
  ASR GPU process memory: 13.1 GiB
  HY-MT GPU process memory: 8.2 GiB
  GPU utilization: 97 percent
  evidence gaps: none in the preceding 30-second window
```

## Security boundaries

- API credentials enter collectors through environment variables only.
- Credentials are never placed in argv, metadata, filenames, reports, or logs.
- Audio, transcripts, prompts, model token values, and exception messages are
  forbidden.
- Session/job identities are random runtime identifiers, not client-provided
  text.
- Configuration capture uses an explicit safe allowlist.
- The analyzer treats every collected string as data and never executes it.
- Evidence remains outside the repository and is never staged by release gates.

## Implementation boundaries

Expected implementation paths are limited to:

- `app/asr_observability.py`
- `app/asr_gateway.py`
- `app/asr_gateway_scheduler.py`
- `app/asr_gateway_metrics.py`
- `app/asr_gateway_chunking.py`
- `app/asr_faster_whisper.py`
- `app/config.py`
- `scripts/monitor_asr_bottleneck.sh`
- `scripts/analyze_asr_bottleneck.py`
- focused tests for the files above
- the operator documentation needed to run and interpret the monitor

No model, protocol, deployment topology, or scheduling behavior changes belong
in the observability candidate.

## Verification

### Deterministic tests

- JSON schema and UTC timestamp validation;
- sensitive field and value rejection;
- stable scheduler/adapter batch identity;
- always-on versus diagnostic event selection;
- all four controlled buffer rejection reasons;
- bounded-window percentile behavior, including empty and rollover cases;
- fake-clock scheduler-to-engine correlation;
- session open/terminal/release ownership conservation;
- archive ownership, locking, retention, symlink refusal, and atomic finalize;
- collector interruption and partial failure;
- secret scan refusal;
- analyzer fixtures for fragmentation, engine tail, GPU contention, buffer
  rejection, cleanup conflict, and missing evidence.

### Integration tests

- two compatible sessions produce one scheduler batch and one engine group;
- a scheduler batch split by beam produces two correlated engine calls;
- a controlled slow engine call is visible in logs, metrics, and the report;
- each buffer limit produces its exact controlled rejection reason;
- a terminal failure releases reservation, backend state, and lease;
- monitor finalization produces a valid report and manifest without credentials.

### Test-environment acceptance

1. Enable diagnostic logging and recreate only the ASR service.
2. Start the monitor on the test host.
3. Run the two-stream normal protocol test.
4. Run the five-stream workload that currently reproduces capacity failure.
5. Stop the monitor and analyze the archive.
6. Confirm every failure has an explicit source and a complete preceding
   scheduler/engine/buffer/resource timeline.
7. Confirm the evidence archive contains no credentials, audio, transcript, or
   prompt content.

## Acceptance criteria

The design is implemented when one archive can answer all of these questions
with exact timestamps and identifiers:

- Which session and incoming frame crossed which buffer limit?
- Which job owned the reservation at that time?
- Which scheduler batch accepted that job, and why did that batch dispatch at
  its observed size?
- How many actual engine calls did the adapter create from the scheduler batch?
- What were their beam, accumulated audio range, output-size summary, and
  inference duration?
- What were GPU and HY-MT resource levels during the inference tail?
- Did terminal cleanup release Gateway PCM, reservation, adapter state, and
  lease?
- Is any conclusion weakened by a collector gap or missing event?

If any answer requires transcript content, API credentials, raw PCM, or manual
guessing from aggregate averages, the implementation is incomplete.
