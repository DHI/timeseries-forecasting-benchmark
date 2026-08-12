# Research Log Schema

Defines the structure `index.md` and `details.md` follow, so entries stay consistent as the log
grows. This file is the source of truth for the format — `CLAUDE.md` points here rather than
repeating it.

## Files

- `index.md` — title-only index, grouped by date, for scanning which entries are relevant.
- `details.md` — the full record, one entry per decision, as a multi-document YAML stream.
- `resources/research_graph.html` — interactive graph view of the same decisions.

## Ids

Every entry has a unique positive integer `id`. Ids increase with chronological order: the
oldest entry is `1`, and each entry logged after it gets the next integer up. A new entry always
gets `id = max(existing ids) + 1`.

Both files list entries **newest first** (unchanged convention), so ids *descend* reading top to
bottom — the most recent entry has the highest id. Entries logged on the same day keep their real
time-of-day order, so ids decrease strictly with no ties or exceptions anywhere in the file.

## `index.md` format

Entries are grouped under a heading per distinct date (or date range), newest date first; entries
within a group are also newest first:

```markdown
### 2026-07-30
- 88 — Fixed MLP's divergence at the source, and added a Forecast strategy card to every method

### 2026-07-29
- 87 — MLP+exo existed but was never in the pilot run
- 86 — Method cards get a Gaps capability, derived rather than hand-authored
```

A date range (work spanning more than one day) gets its own group under the literal range as the
heading, e.g. `### 2026-06-12 to 2026-06-13`.

## `details.md` format

A multi-document YAML stream: entries separated by a line containing only `---`, newest entry
first. Each document has these fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | integer | yes | Unique; see Ids above. |
| `date` | quoted string | yes | `"YYYY-MM-DD"`, or `"YYYY-MM-DD to YYYY-MM-DD"` for work spanning several days. Quoted so no YAML loader coerces it to a date type. |
| `title` | quoted string | yes | One line. Matches the entry's heading text in `index.md`. |
| `description` | literal block scalar (`\|`) | yes | The entry body: self-contained prose for a GitHub audience, 3–6 sentences per `CLAUDE.md`'s logging guidance. Markdown formatting (`` ` ``, `**bold**`) is fine — it's rendered wherever this gets displayed. |
| `commits` | list of one-key mappings | yes | `- <sha>: "subject"` (the key is the commit SHA), oldest commit first — the commit(s) that did the work this entry describes. Empty list (`commits: []`) only if the entry is a discussion/decision with no corresponding commit. |
| `commits_note` | quoted string | no | Only when the commit list is a best-effort reconstruction rather than commits that name the entry directly (e.g. work backfilled from docs/history after the fact) — value `"approximate — reconstructed after the fact"`. |

Quoting: `title` and every commit subject are double-quoted, with internal `"` escaped as `\"`
and `\` escaped as `\\` — several existing entries contain both `'` and `"`, so quoting
everything uniformly avoids picking a quote style per string.

### Worked example

```yaml
---
id: 88
date: "2026-07-30"
title: "Fixed MLP's divergence at the source, and added a Forecast strategy card to every method"
description: |
  Switched `MLPForecaster` from mlforecast's recursive strategy to its direct multi-horizon
  strategy (one model per horizon step, no feedback loop) — the `_BoundedRegressor` output clip
  added earlier treated the symptom, not the cause, and cost MLP the ability to extrapolate a
  trend. Existing MLP pilot results are stale pending a re-run.
commits:
  - 7107040: "Switch MLP to mlforecast's direct multi-horizon strategy"
  - 7d08804: "Add ForecastStrategy metadata field alongside Capabilities/Scorecard"
  - 4bd6535: "Log the MLP direct-horizon fix and forecast_strategy taxonomy"
---
id: 87
date: "2026-07-29"
title: "MLP+exo existed but was never in the pilot run"
description: |
  ...
commits:
  - 1d553b0: "Including exo to MLP"
```

## Adding a new entry

1. `details.md`: prepend a new YAML document at the top with `id = max(existing) + 1`, today's
   date, title, description, and the commit(s) from this session's work (usually no
   reconstruction needed — it's whatever you just committed).
2. `index.md`: prepend the `- id — Title` bullet under today's date heading (add a new date
   heading if today doesn't have one yet).
3. `resources/research_graph.html`: add a matching node (and any relevant edges), per
   `CLAUDE.md`.
