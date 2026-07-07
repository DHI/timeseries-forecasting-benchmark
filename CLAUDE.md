# Project instructions

## Research log

Maintain `RESEARCH_LOG.md` at the repo root as a running record of this project's testing and
decision-making process. This is a research project and the log is meant to be git-tracked and
shared with collaborators via GitHub, so write entries for that audience (self-contained,
no assumed context from the conversation).

Append a new dated entry:
- whenever the user explicitly asks (e.g. "add summary notes", "log this")
- proactively, after any substantive piece of work — a test/experiment run, a modeling or
  methodology decision, a result that changes direction, or a non-obvious tradeoff — even
  without being asked

Each entry should cover: what was tried/tested, what was decided and why, and what it means
going forward. Skip trivial or purely mechanical changes (typo fixes, file moves, formatting,
dependency bumps). Create the file if it doesn't exist yet. New entries go at the top, under
the "newest first" marker.

## Code formatting

CI runs `uv run ruff check .` and fails the build on any lint violation. Before committing,
run `uv run ruff check --fix .` and `uv run ruff format .` and confirm `uv run ruff check .`
passes clean. Do this for every commit that touches Python files, not just at the end of a task.

## Commit hygiene
Before implementing any non-trivial code change, or before running any git commit, consult the `jap-commit-preferences` skill.

## Writing style

Before producing any prose output (explanations, docs, comments, commit messages, summaries), consult the `japr-writing-skill` skill and apply it — even if this instruction doesn't look like a "writing" task on the surface.

