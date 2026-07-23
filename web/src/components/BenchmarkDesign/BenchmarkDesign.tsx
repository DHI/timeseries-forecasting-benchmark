const TEXT_STYLE = { color: 'var(--text-secondary)', maxWidth: 1000 };

/** Static methodology writeup below the matrix: splits, curation, model parameters, case
 * description, and exogenous-input handling. Port of app/app.py's "Benchmark Design" section. */
export function BenchmarkDesign() {
  return (
    <>
      <h2>Benchmark Design</h2>
      <div style={TEXT_STYLE}>
        <p>
          <strong>Splits. </strong>
          Each case declares its train/val/test split in case.toml, either as calendar dates
          (train_end, val_end, test_end) or as fractions (default 70/15/15) of the observed
          timestamps, so series with gaps still split cleanly. Splits are strictly chronological —
          no shuffling.
        </p>
        <p>
          <strong>Curation. </strong>
          Curation enforces minimum lengths: train must cover at least two full seasonal cycles, and
          val+test together must cover at least one forecast horizon. Evaluation is rolling-origin
          by default (errors pooled across many refit points in the test window), with a --fast
          single-holdout mode for quick checks. Missing data is kept as explicit NaN: imputed for
          training, masked out during scoring so only real observations count.
        </p>
        <p>
          <strong>Model parameters. </strong>
          There&apos;s no per-case tuning or grid search. Each method uses fixed, hard-coded
          defaults — e.g., LightGBM and CatBoost both use 200 iterations at a 0.05 learning rate, so
          the two gradient-boosting methods compete on equal footing. Classical methods (AutoARIMA,
          AutoETS, etc.) run their own internal search, but training size is capped per method to
          bound fit time, not to improve accuracy. Any deviation from a method&apos;s default is
          manual and explicit, not automated.
        </p>
        <p>
          <strong>Case description. </strong>
          Cases carry structured metadata in case.toml: domain, target, frequency, horizons,
          exogenous variables, source citation, and declared seasonalities. Seasonalities are the
          single source of truth for every downstream method — a candidate only qualifies with real
          evidence (enough training cycles, a clear periodogram peak, sufficient autocorrelation).
          There&apos;s no ground-truth seasonality to validate against; any override needs a written
          justification. The same evidence-based standard applies to marking exogenous variables as
          necessary, since that&apos;s a domain judgment, not something a model&apos;s error alone
          can tell you.
        </p>
        <p>
          <strong>Exogenous inputs. </strong>
          Each case declares its exogenous tier — none, optional, or necessary — a domain judgment,
          not something inferable from measured skill deltas. LinearLags, LightGBM, and CatBoost
          each have an exogenous-aware sibling; the matrix shows one column per algorithm, and the
          &quot;Use exogenous inputs where available&quot; checkbox swaps in that sibling&apos;s
          score for cases with real exogenous data, marking the cell with a teal corner. Toggling it
          changes which of two already-computed scores is shown per cell — it never adds columns,
          drops cases, or aggregates across them, matching the matrix&apos;s design elsewhere.
          Calendar and seasonal features (day-of-week, sin/cos-of-period) sit outside this toggle
          entirely: they&apos;re deterministic and free for any future timestamp, so every
          windowed-ML method always includes them. Genuine exogenous covariates don&apos;t have that
          guarantee — future values fed to the exogenous-aware methods here come from observed
          reanalysis data (a perfect-forecast proxy), so their scores are an upper bound on real
          operational skill, which depends on actual forecast quality for those covariates.
        </p>
      </div>
    </>
  );
}
