# MT Cloud Benchmark Monitor Design

## Objective

Add one self-contained cloud-host Shell script that monitors the existing HY-MT container while a separate local client runs `scripts/benchmark_mt.py`. The monitor must generate a complete, portable evidence archive without sending translation requests or changing service capacity.

The script is an observation tool only. It does not start, stop, restart, rebuild, or reconfigure the MT service; it does not run the benchmark; and it does not read an API key or endpoint.

## Operator Flow

```text
Cloud host: start scripts/monitor_mt_benchmark.sh
Local host: run scripts/benchmark_mt.py against the test service
Local host: wait for benchmark completion
Cloud host: press Ctrl+C
Monitor: stop collectors, summarize, checksum, archive, print artifact paths
```

The script runs until `SIGINT` or `SIGTERM`. Normal interruption is a successful monitored run if finalization completes. An unexpected collector failure is retained in evidence and reflected in the final report.

## Runtime Contract

Defaults:

- Docker Compose service: `hy-mt-api`
- GPU index: `0`
- output root: `/tmp/mt-monitor`
- GPU sampling interval: `0.5` seconds
- container sampling interval: `1` second
- retained completed runs: `20`
- maximum retention age: `14` days

Optional environment variables:

- `MT_MONITOR_SERVICE`
- `MT_MONITOR_GPU_INDEX`
- `MT_MONITOR_OUTPUT_ROOT`
- `MT_MONITOR_GPU_INTERVAL_SECONDS`
- `MT_MONITOR_CONTAINER_INTERVAL_SECONDS`
- `MT_MONITOR_KEEP_RUNS`
- `MT_MONITOR_KEEP_DAYS`

The script requires `bash`, `docker`, Docker Compose v2, `nvidia-smi`, `python3`, `tar`, and `sha256sum`. It must run from a repository checkout containing the Compose project. No endpoint, API key, model credential, or benchmark corpus is accepted by the script.

## Startup Safety

Before collectors start, the script must:

1. Resolve the repository root from the script location.
2. Validate required commands and numeric settings.
3. Require an absolute, non-root, non-symlink output root.
4. Refuse a pre-existing non-empty output root unless it has the monitor ownership marker.
5. Acquire an atomic lock directory and refuse a concurrent monitor.
6. Resolve exactly one running container for `MT_MONITOR_SERVICE`.
7. Confirm the selected GPU can be queried.
8. Create a unique UTC run directory only after validation succeeds.

The script never deletes arbitrary directories. Retention applies only beneath its marked output root and only to completed run directories and their matching archives.

## Evidence Layout

Each run is written to:

```text
/tmp/mt-monitor/runs/<run-id>/
```

Files:

- `metadata.json`: run ID, UTC start/finish, hostname, kernel, repository SHA, service name, container ID, image ID, container start state, GPU index and GPU name.
- `gpu.csv`: timestamp, GPU utilization, memory utilization, used/total MiB, power, temperature, P-state, SM clock, and memory clock.
- `gpu-processes.csv`: timestamp, PID, process name, and used GPU memory.
- `container.csv`: timestamp, CPU percent, memory usage/limit, memory percent, network I/O, block I/O, and PID count.
- `service.log`: `docker compose logs --follow --timestamps --since <monitor-start>` for the MT service.
- `collector-errors.log`: timestamped collector and finalization failures.
- `report.json`: deterministic machine-readable summary.
- `report.md`: human-readable summary and interpretation boundaries.
- `manifest.sha256`: checksums for every regular evidence file except the manifest itself.
- `.completed`: written only after reports and manifest succeed.

After finalization, create `/tmp/mt-monitor/runs/<run-id>.tar.gz` atomically through a `.partial` archive.

## Collectors

### GPU

Poll `nvidia-smi --query-gpu` at the configured interval. A failed sample appends a short category to `collector-errors.log`; it does not terminate other collectors.

### GPU Processes

Poll `nvidia-smi --query-compute-apps`. Empty output is valid when no compute process is visible. The query must not include command lines or environment values.

### Container Resources

Poll `docker stats --no-stream` for the resolved container. Use a delimiter-safe format and retain Docker's raw unit strings in CSV; the embedded report generator converts supported byte units for memory aggregation. A sample failure is recorded and the loop continues.

### Service Logs

Follow only the configured MT service from the monitor start time. The collector does not add request instrumentation or call the service. Logs are preserved verbatim because they are operational evidence; the runbook warns that the application must not log request/response bodies or secrets.

Collectors run as background child processes. Finalization sends termination, waits for children, and records any child that fails to stop. It never kills processes it did not start.

## Report Calculations

An embedded standard-library Python block reads completed CSV files and writes both reports. No separate analyzer file is required.

`report.json` and `report.md` include:

- UTC duration;
- sample counts and parsing error counts;
- GPU utilization average, P95, and maximum;
- GPU memory used average and maximum;
- memory utilization maximum;
- power average and maximum;
- temperature maximum;
- container CPU average, P95, and maximum;
- container memory used average and maximum where units are parseable;
- GPU process peak memory by PID/process;
- collector error count and categories;
- container final running/restart/OOM state when inspectable.

P95 uses deterministic nearest-rank selection. Missing samples produce `null` metrics and a visible warning, never a fabricated zero or pass.

The report does not determine MT RPS, token throughput, latency, or cost. Those values remain owned by `mt-benchmark.json` and `mt-benchmark.md` from the local benchmark client. The two evidence sets are correlated by UTC window and operator run notes.

## Secret and Content Boundaries

The script and reports must never record:

- environment dumps;
- API keys or authorization headers;
- endpoint URLs;
- benchmark corpus content;
- translation request or response bodies;
- Docker inspect environment variables;
- process command-line arguments.

Metadata includes only allowlisted infrastructure fields. Service logs are the only verbatim source; the script cannot redact arbitrary application output reliably, so the operator must verify logging policy before the run.

## Finalization and Retention

Finalization is idempotent and runs once through traps. It:

1. Stops and waits for owned collectors.
2. Captures finish time and final container state.
3. Generates JSON and Markdown reports.
4. Creates the checksum manifest.
5. Writes `.completed`.
6. Creates and atomically renames the archive.
7. Deletes only old completed artifacts according to count and age settings.
8. Releases the lock.
9. Prints the run directory, report, and archive paths.

If reports, manifest, or archive creation fails, the run remains incomplete, the error is visible, and the script exits nonzero. It must not label incomplete evidence as successful.

## Files and Scope

- Create `scripts/monitor_mt_benchmark.sh`.
- Create `tests/test_monitor_mt_benchmark.py`.
- Create `docs/mt-cloud-monitor.md` as the operator runbook.
- Add one README file-inventory entry.

No application, translation model, ASR protocol, TTS service, Compose, image, or existing monitor behavior changes.

## Automated Tests

Pytest launches the Shell script with temporary fake executables for `docker`, `nvidia-smi`, and `tar`, then controls it with signals. Deterministic tests cover:

- command and setting validation;
- refusal of unmarked output directories;
- lock exclusion;
- exact MT container requirement;
- concurrent collector startup;
- no HTTP/client invocation;
- Ctrl+C finalization;
- report metrics from fixed samples;
- missing/failed sample visibility;
- manifest and atomic archive;
- retention restricted to owned completed artifacts;
- endpoint/key/corpus absence from outputs;
- child cleanup and final Git status.

Implementation follows test-first evidence. The sole Development Agent stages only intended files, runs focused tests and `scripts/verify_asr_release.sh commit` against the staged candidate, then commits. A separate read-only Test Agent accepts or rejects the exact SHA. After acceptance, the primary agent reruns the operator workflow probe, focused tests, and commit gate.

## Acceptance Criteria

- Starting the script on the cloud host produces no MT requests and does not restart or reconfigure the service.
- Monitoring overhead is limited to configured `nvidia-smi`, `docker stats`, and log-follow sampling.
- Ctrl+C yields complete JSON, Markdown, checksum, completion marker, and archive artifacts.
- Reports distinguish missing data from zero load.
- No endpoint, key, environment dump, corpus, or translation content is introduced by the monitor.
- The script is locally commit-verified and independently accepted before handoff.
- Real cloud execution remains an external evidence gate until the operator returns the generated archive.
