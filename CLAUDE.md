# Project instructions

## Research log

Maintain `research_log/details.md` as a running record of this project's testing and
decision-making process, and `research_log/index.md` as a title-only index of every entry in it.
This is a research project and the log is meant to be git-tracked and shared with collaborators
via GitHub, so write entries for that audience (self-contained, no assumed context from the
conversation) — but keep them skimmable, not exhaustive. See `research_log/schema.md` for the
exact file format (id assignment, YAML fields, index grouping).

Append a new dated entry:
- whenever the user explicitly asks (e.g. "add summary notes", "log this")
- proactively, after any substantive piece of work — a test/experiment run, a modeling or
  methodology decision, a result that changes direction, or a non-obvious tradeoff — even
  without being asked

Keep each entry to **3–6 sentences**, roughly: what was tried, what was decided (and the one
reason that mattered most), and what it means going forward — only if that's non-obvious or
requires a reader to act (e.g. "re-run X" or "old numbers are now stale"). Skip:
- alternatives considered and rejected, unless the rejected option is likely to resurface
- verification/testing narration ("checked via...", "confirmed with...") — that belongs in
  commit messages or PR descriptions, not the log
- trivial or purely mechanical changes (typos, file moves, formatting, dependency bumps)

If several small entries in one day are really steps toward one evolving decision, combine them
into a single entry rather than logging each step separately.

A new entry gets `id = max(existing ids) + 1` and goes at the top of both files, per
`research_log/schema.md`: the full YAML document (including its own `commits` list — usually just
whatever was committed in the same session, no reconstruction needed) in `details.md`, and the
`- id — Title` bullet under today's date in `index.md`.

When adding a new entry, also add a matching node (and any relevant edges) to the embedded data
in `research_log/resources/research_graph.html`, the interactive graph view of this log.

## Code formatting

CI runs `uv run ruff check .` and fails the build on any lint violation. Before committing,
run `uv run ruff check --fix .` and `uv run ruff format .` and confirm `uv run ruff check .`
passes clean. Do this for every commit that touches Python files, not just at the end of a task.

## Commit hygiene
Before starting any non-trivial code change, consult the `japr-commit-preferences` skill once. Its rules stay in effect for the rest of that task — no need to re-consult it before each individual commit.

## Writing style

Before producing any prose output (explanations, docs, comments, commit messages, summaries), consult the `japr-writing-skill` skill and apply it — even if this instruction doesn't look like a "writing" task on the surface.

