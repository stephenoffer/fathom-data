# 5. Shadow mode is the adoption mechanism, not a debugging aid

Status: accepted

## Context

The planner's value proposition asks for an unusual amount of trust: let a new tool
decide what *not* to rebuild. The failure mode is silent. A tool that skips a
partition it should have rebuilt does not crash, does not log an error, and does not
show up in a dashboard. It serves stale numbers until someone notices a report is
wrong, weeks later, and by then nobody connects it to the tool.

No amount of documentation or test coverage earns that trust from a stranger.

## Decision

Ship `shadow.run` before shipping `apply`, and make the honest metric the headline.

Shadow mode fingerprints each target partition, runs a full rebuild, fingerprints
again, and grades the plan against what actually changed:

- **savings** — partitions the plan would have skipped
- **missed** — partitions the plan called clean that the rebuild proved dirty

`missed` must be zero. It is the direct empirical test of the soundness invariant in
ADR 2, and it is reported per run and accumulated in the store. `fathom shadow`
exits non-zero and prints a blunt warning the moment it is not zero.

Ordering is load-bearing and documented: source data lands, then plan, then shadow,
then apply. Applying first destroys the pre-change state there is nothing left to
compare against.

Fingerprints sort rows inside the aggregate so they are order-independent. Without
that, every rebuild would report every partition as changed and the metric would be
meaningless.

## Consequences

- Users can run this alongside an existing full rebuild for weeks at zero risk,
  accumulating evidence, before anything writes.
- Publishing our own failure rate is a real commitment. If `missed` is ever
  non-zero in the wild, that is a bug report we cannot argue with — which is the
  point.
- The soundness checker must itself be tested for its ability to fail. A test feeds
  it a deliberately wrong plan and asserts it reports unsound; a checker that
  cannot fail is decoration.
- Shadow mode costs a full rebuild per run, so it is a rollout tool rather than a
  permanent fixture. That is the right trade while trust is being established.
