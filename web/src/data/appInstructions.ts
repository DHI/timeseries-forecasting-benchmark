/** Markdown shown in the instructions modal opened from the info icon next to the page title.
 * Moved here from the repo root's `APP_INSTRUCTIONS.md` so it ships with the app instead of
 * living in a file nobody browsing the site would find. */
export const APP_INSTRUCTIONS = `
## Selecting methods in the matrix

Click a cell to open a method card for it. You can have up to three method cards open
at once, all from the same row (case).

- The first cell you click becomes the **reference method**. When you open the snapshot
  panel, the reference method's score is what decides which forecast origins are shown —
  the best, median, and worst origins by the current metric, for that method. Every other
  selected method is then plotted at those same origins, so you're comparing them on the
  reference's terms.
- Click a selected cell again to remove it. If you remove the last one, the whole selection
  clears.
- Click a fourth method (in the same row): the oldest of your three selections drops off
  and the new one is added. Because the reference is always the oldest surviving selection,
  this can quietly change which method is the reference and can shift the colors used for
  the other selected methods, since colors are assigned by position, not by method.
- Click a cell in a different row: your whole selection resets to just that one method. You
  can't compare methods across different cases.

Each method card can be resized: drag the thin bar just under its title up or down to show
more or less of its content.

## Case details panel

The panel on the right can be resized too — drag its left edge to make it wider or
narrower, if the content doesn't fit.

## Control panel

The sidebar lets you filter the matrix by domain (e.g. hydrology, groundwater, water
quality) and by method family (baseline, ensemble, classical, ML, DL, foundation models).
Both lists show short abbreviations to save space — hover over one to see its full name as
a tooltip.

### Metric selection

The metric dropdown controls what the matrix scores and colors by. Most metrics
(MASE, MAE, RMSE, MAPE, sMAPE, Skill vs. Persistence, Skill vs. Mean) apply to every case.

Two metrics, NSE and KGE, are domain-restricted: they're standard goodness-of-fit measures
for flow and water-level data, and don't make sense for other domains (e.g. wave height).
Selecting one of them narrows the matrix down to only the cases whose configuration
explicitly declares that metric as meaningful — everything else disappears from the grid
until you switch back to a universal metric.

### "Score on full training history" checkbox

This toggle applies only to **global methods**: methods that fit one model and reuse it for
every forecast origin, rather than refitting at each origin. That covers the ML, DL, and
foundation-model families. Baseline, classical, and ensemble methods are refit at every
origin, so there's no separate "trained model" to extend, and this checkbox has no effect
on them.

Among global methods, only those that actually learn from training data (ML and DL) can
make use of more history — turning the checkbox on refits them on the case's full available
history instead of just the shared curated span everyone is scored on, and swaps in that
score. Foundation models are also "global" in the sense of reusing one fit, but they work
zero-shot from recent context rather than learning from a training window, so extra history
doesn't change their forecasts — the checkbox doesn't affect them either, even though they
sit in the same family group.

When a cell's value has been swapped to the full-history version, it's marked with a small
pink triangle in the corner, as a reminder that it isn't directly comparable to the rest of
its row (it was fit on more data than the others). The same swap also affects the case
details panel (training history plot and split counts) and the snapshot panel (forecasts
are drawn from the full-history fit and labeled accordingly).

### "Use exogenous inputs where available" checkbox

Exogenous inputs are external variables — like rainfall or a weather forecast — fed to a
model alongside the target series' own history, on top of what it would use otherwise.

Some methods (a subset of ML and foundation-model methods) have a second variant trained
to also use these external variables. Turning this checkbox on swaps a method's cell over
to that variant's score, but only where both sides support it: the method needs to have
that alternate variant, and the case needs to actually provide exogenous columns. Methods
without an exogenous-aware variant, and cases without any exogenous data, are unaffected.

Swapped cells get a small teal triangle in the opposite corner from the training-history
marker, so the two never collide on the same cell. If you have method cards or the snapshot
panel open, toggling this checkbox also swaps what they show — the same method now displays
the exogenous variant's description and forecasts instead of the base version.

### "Show training time" button

Press and hold this button, or hold the space bar, to see each cell's measured training time
instead of its score. The whole matrix — including any cells already swapped by the two
checkboxes above — switches to timing for as long as you hold it down; release to go back to
normal. You can still move your mouse around the matrix while holding, and the usual
row/column hover tooltips keep working, so you can hold with one hand (so to speak) and hover
to identify cells with the other.
`.trim();
