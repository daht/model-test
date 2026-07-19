# SenseVoice terminal cancellation design

## Context

At 88 concurrent rolling streams, a completed result can exceed the connection
lag limit after scheduler cleanup has released its reservation. That release can
allow the same session to enqueue a continuation job before the lagged result is
applied. The result path then fails the session and aborts its adapter state, but
the continuation remains eligible for dispatch. SenseVoice rejects that stale
job before model inference, and scheduler fail-stop correctly fails the worker's
remaining queue.

## Objective

When any internal result path fails a session, cancel all queued work for that
session generation before invalidating the session or aborting adapter state.
No job from the failed generation may be submitted afterward.

## Design

`GatewayRuntime._fail_session()` is the common transition for connection lag,
undecoded age, adapter errors, and result conflicts. Before calling
`GatewaySession.fail()`, it records cancellation through the scheduler's existing
`cancel_session(session_id, generation=...)` interface. The scheduler already
filters cancelled generations before selecting its next batch and rolls back
their reservations through the normal reject path.

The result publication path must not call `wait_session_safe()`. Publication is
executed by the scheduler before it marks the current accepted job safe, so
waiting there would make the scheduler wait on its own completion barrier.
External error and abort paths retain their existing wait-before-cleanup
behavior.

No new configuration, retry, scheduler abstraction, or capacity change is
introduced.

## Deterministic regression

Use scheduler control rather than timing sleeps:

1. Complete and clean the current job.
2. Enqueue a continuation for the same generation before publishing a result
   whose recorded connection lag exceeds the limit.
3. Publish the lagged result and run the scheduler again.
4. Assert that the adapter never receives the continuation, the scheduler queue
   and queued-sample accounting reach zero, the reservation is released, no
   cleanup conflict is emitted, and the session emits exactly one terminal
   transition.

The focused gateway and scheduler suites plus the repository commit gate must
remain green.

## Acceptance boundary

This change fixes the lifecycle cascade observed in run
`20260719T101035Z-ef7e1b`. It does not establish that 88 streams fit within the
A10 latency target. A new monitored 88-stream run is required after deployment;
80 streams remains the last demonstrated stable lower bound until then.
