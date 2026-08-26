import { ClickCellDiagram } from './ClickCellDiagram';
import { PageOverviewDiagram } from './PageOverviewDiagram';
import { ResizeHandleDiagram } from './ResizeHandleDiagram';

/** Static usage writeup below the matrix: selecting methods, the case details panel, and the
 * control panel's filters, metric selection, and toggles. */
export function AppInstructions({ maxWidth }: { maxWidth?: number }) {
  return (
    <div style={{ color: 'var(--text-secondary)', ...(maxWidth !== undefined && { maxWidth }) }}>
      <p>
        This page compares forecasting methods across cases in a heatmap. Click a cell to dig into a
        case, drag panels to resize them, and use the sidebar to filter the matrix down to what you
        care about. The diagram below shows where everything sits.
      </p>
      <PageOverviewDiagram />
      <h3>Selecting methods in the matrix</h3>
      <p>
        Click a cell to open a method card for it. You can have up to three method cards open at
        once, all from the same row (case). Hovering a cell (without clicking) highlights its row
        and column and shows the case ID and method name, so you can orient yourself before you
        click.
      </p>
      <ClickCellDiagram />
      <ul>
        <li>
          The first cell you click becomes the <strong>reference method</strong>. When you open the
          snapshot panel, the reference method&apos;s score is what decides which forecast origins
          are shown — the best, median, and worst origins by the current metric, for that method.
          Every other selected method is then plotted at those same origins, so you&apos;re
          comparing them on the reference&apos;s terms.
        </li>
        <li>
          Click a selected cell again to remove it. If you remove the last one, the whole selection
          clears.
        </li>
        <li>
          Click a fourth method (in the same row): the oldest of your three selections drops off and
          the new one is added. Because the reference is always the oldest surviving selection, this
          can quietly change which method is the reference and can shift the colors used for the
          other selected methods, since colors are assigned by position, not by method.
        </li>
        <li>
          Click a cell in a different row: your whole selection resets to just that one method. You
          can&apos;t compare methods across different cases.
        </li>
      </ul>
      <h3>Method cards</h3>
      <p>
        Click a card&apos;s title bar to collapse it down to just that bar, or to expand it again.
        Drag the same bar up or down instead of clicking it to resize the card&apos;s content,
        showing more or less of it.
      </p>
      <ResizeHandleDiagram />
      <h3>Case details panel</h3>
      <p>
        The panel on the right can be resized too — drag its left edge to make it wider or narrower,
        if the content doesn&apos;t fit.
      </p>
      <h3>Control panel</h3>
      <p>
        The sidebar lets you filter the matrix by domain (e.g. hydrology, groundwater, water
        quality) and by method family (baseline, ensemble, classical, ML, DL, foundation models).
        Both lists show short abbreviations to save space — hover over one to see its full name as a
        tooltip.
      </p>
      <h4>Metric selection</h4>
      <p>
        The metric dropdown controls what the matrix scores and colors by. Most metrics (MASE, MAE,
        RMSE, MAPE, sMAPE, Skill vs. Persistence, Skill vs. Mean) apply to every case.
      </p>
      <p>
        Two metrics, NSE and KGE, are domain-restricted: they&apos;re standard goodness-of-fit
        measures for flow and water-level data, and don&apos;t make sense for other domains (e.g.
        wave height). Selecting one of them narrows the matrix down to only the cases whose
        configuration explicitly declares that metric as meaningful — everything else disappears
        from the grid until you switch back to a universal metric.
      </p>
      <h4>&quot;Score on full training history&quot; checkbox</h4>
      <p>
        This toggle applies only to <strong>global methods</strong>: methods that fit one model and
        reuse it for every forecast origin, rather than refitting at each origin. That covers the
        ML, DL, and foundation-model families, plus all of the classical ones (AutoARIMA, AutoETS,
        Theta, MSTL, TBATS) and the ensembles. The naive baselines are refit at every origin, so
        there&apos;s no separate &quot;trained model&quot; to extend, and this checkbox has no
        effect on them. It has none on the ensembles either: their members take a single
        contiguous series, so there is no full-history variant to fit.
      </p>
      <p>
        Among global methods, this benchmark currently only scores the full-history fit for those
        that actually learn from training data (ML and DL) — turning the checkbox on refits them on
        the case&apos;s full available history instead of just the shared curated span everyone is
        scored on, and swaps in that score. The fit-once classical methods roll forward like ML/DL,
        but a full-history variant isn&apos;t wired up for them yet, so the checkbox has no effect
        on them either. Foundation models are also &quot;global&quot; in the sense of reusing one
        fit, but they work zero-shot from recent context rather than learning from a training
        window, so extra history doesn&apos;t change their forecasts — the checkbox doesn&apos;t
        affect them either, even though they sit in the same family group.
      </p>
      <p>
        When a cell&apos;s value has been swapped to the full-history version, it&apos;s marked with
        a small pink triangle in the corner, as a reminder that it isn&apos;t directly comparable to
        the rest of its row (it was fit on more data than the others). The same swap also affects
        the case details panel (training history plot and split counts) and the snapshot panel
        (forecasts are drawn from the full-history fit and labeled accordingly).
      </p>
      <h4>&quot;Use exogenous inputs where available&quot; checkbox</h4>
      <p>
        Exogenous inputs are external variables — like rainfall or a weather forecast — fed to a
        model alongside the target series&apos; own history, on top of what it would use otherwise.
      </p>
      <p>
        Some methods (a subset of ML and foundation-model methods) have a second variant trained to
        also use these external variables. Turning this checkbox on swaps a method&apos;s cell over
        to that variant&apos;s score, but only where both sides support it: the method needs to have
        that alternate variant, and the case needs to actually provide exogenous columns. Methods
        without an exogenous-aware variant, and cases without any exogenous data, are unaffected.
      </p>
      <p>
        Swapped cells get a small teal triangle in the opposite corner from the training-history
        marker, so the two never collide on the same cell. If you have method cards or the snapshot
        panel open, toggling this checkbox also swaps what they show — the same method now displays
        the exogenous variant&apos;s description and forecasts instead of the base version.
      </p>
      <h4>&quot;Show training time&quot; button</h4>
      <p>
        Press and hold this button, or hold the space bar, to see each cell&apos;s cost to produce
        one forecast with that method, instead of its score. The whole matrix — including any cells
        already swapped by the two checkboxes above — switches to timing for as long as you hold it
        down; release to go back to normal. The row/column hover highlight described above still
        works while holding, so you can hold with one hand (so to speak) and hover to identify cells
        with the other.
      </p>
      <p>
        The number shown is fit time (if the method needs fitting) plus predict time. The naive
        baselines are refit at every rolling origin, so their fit time reflects one such
        refit; the fit-once methods (ML, DL, and the classical ones) report that single fit; an
        ensemble reports the one fit of its trained members, with its naive members&apos;
        per-origin cost falling under predict time instead;
        zero-shot foundation models have no case-specific training at all, so their number is
        essentially inference time. Predict time is not uniformly small either — Theta and MSTL
        re-derive a seasonal decomposition over the whole history at every origin, so on the longest
        records their predict cost dominates their fit cost by a wide margin. Hover a cell while
        holding to see the fit/predict split.
      </p>
    </div>
  );
}
