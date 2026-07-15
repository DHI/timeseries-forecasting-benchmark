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
  seconds: number | null;
  frac_finite_test: number | null;
  status: string;
  n_origins: number;
};
