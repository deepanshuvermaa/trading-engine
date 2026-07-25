# LESSONS — compressed "tried X -> got Y -> don't retry"

Append-only, newest at the bottom. Each line is a durable, compressed lesson the
daily loop consults before re-proposing a hypothesis. Format:

`YYYY-MM-DD [exp-id] tried <change> -> <result on holdout> -> <don't retry / do>`

<!-- The daily outer loop appends REVERT/VETO outcomes here automatically. -->

- 2026-07-25 [0001-example] seeded the loop scaffold -> no live trades yet, holdout not scorable -> wait for real closed trades before trusting any metric delta.
