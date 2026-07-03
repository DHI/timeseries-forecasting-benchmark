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

## Commit hygiene

When implementing a non-trivial change, split the work into a sequence of small, focused
commits instead of one large commit at the end. Each commit should represent one logical step
(e.g. add a helper, wire it in, update call sites, add tests) and should leave the repo in a
working state.

- Commit as you go, once each logical step is complete, rather than batching everything into
  a single commit at the end of the task.
- Write succinct, informative commit messages: a short imperative summary line, plus a body
  only if the "why" isn't obvious from the diff.
- When planning a big/non-trivial change, ask for confirmation before implementing as usual —
  and as part of that confirmation, explicitly ask whether the user wants you to handle
  committing the work yourself as a sequence of small commits along the way. Only commit
  without asking again during that task if they say yes; otherwise leave committing to them
  (see the git safety rules above).
