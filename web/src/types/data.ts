export type ScoreRow = {
  case_id: string;
  domain: string;
  case_exo_tier: string;
  method: string;
  family: string;
  uses_exogenous: boolean;
  is_global: boolean;
  used_extended_history: boolean;
  horizon: string;
  metric: string;
  value: number | null;
  fit_seconds: number | null;
  predict_seconds: number | null;
  frac_finite_test: number | null;
  status: string;
  n_origins: number;
};

export type ScorecardLevel = 'Low' | 'Medium' | 'High';

/** What structure a method can represent at all. Mirrors the Python `Capabilities` model in
 * src/tsforecastbench/bench/methods/scorecard.py; `exogenous` describes the method, not the
 * column, so plain LightGBM is true even though its column feeds no covariates.
 *
 * `gaps` is the one field with no counterpart in that Python model — it's re-exposed from
 * `Forecaster.pools_segments` at export time (see `webdata.py`'s `_method_registry()`)
 * rather than hand-declared per method, since it's mechanically derivable rather than a
 * judgment call. */
export type Capabilities = {
  trend: boolean;
  seasonality: boolean;
  multiple_seasonality: boolean;
  exogenous: boolean;
  gaps: boolean;
};

/** What a method costs to use. Mirrors the Python `Scorecard` model; hand-authored
 * judgement, not measured from the results. Field order here matches the model's. */
export type Scorecard = {
  history_requirements: ScorecardLevel;
  interpretability: ScorecardLevel;
  fit_cost: ScorecardLevel;
  inference_cost: ScorecardLevel;
  tuning_effort: ScorecardLevel;
  robustness_to_noise: ScorecardLevel;
};

export type MethodMeta = {
  family: string;
  description: string;
  uses_exogenous: boolean;
  library: string;
  compute_profile: string;
  limitations: string;
  code_snippet: string;
  capabilities: Capabilities;
  scorecard: Scorecard;
};

export type MethodRegistry = Record<string, MethodMeta>;

export type Series = {
  ds: string[];
  y: (number | null)[];
};

export type SplitCounts = {
  train: number;
  val: number;
  test: number;
};

export type Distribution = {
  x_grid: number[];
  train_density: (number | null)[] | null;
  test_density: (number | null)[] | null;
};

export type SeasonalPattern = {
  period: number;
  x: (number | null)[];
  y: (number | null)[];
  curve_x: number[];
  curve_y: (number | null)[];
};

export type CaseMeta = {
  domain: string;
  target: string;
  target_unit: string;
  freq: string;
  horizon_primary: string;
  exogenous: string[];
  exogenous_known_future: string[];
  necessary_exogenous: string[];
  exo_tier: 'none' | 'optional' | 'necessary';
  domain_metrics: string[];
  train_tail: Series;
  train_tail_extended: Series;
  test_split: Series;
  split_counts: SplitCounts;
  split_counts_extended: SplitCounts;
  distribution: Distribution;
  distribution_extended: Distribution;
  seasonal_patterns: SeasonalPattern[];
  seasonal_patterns_extended: SeasonalPattern[];
  source: { name: string; url: string | null };
  notes: string;
};

export type CaseRegistry = Record<string, CaseMeta>;

export type SnapshotOrigin = {
  origin_id: number;
  origin_ds: string;
};

export type CuratedOrigin = {
  origin_id: number;
  value: number;
  label: string;
};

export type PredictionVariant = 'local' | 'extended';

export type MethodPredictions = {
  local: Record<string, (number | null)[]>;
  extended?: Record<string, (number | null)[]>;
};

export type MethodCuration = {
  local: CuratedOrigin[];
  extended?: CuratedOrigin[];
};

export type SnapshotBundle = {
  case_id: string;
  n_history: number;
  horizon_steps: number;
  origins: SnapshotOrigin[];
  actual: Record<string, (number | null)[]>;
  predictions: Record<string, MethodPredictions>;
  curation: Record<string, Record<string, MethodCuration>>;
};
