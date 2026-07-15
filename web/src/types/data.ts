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

export type MethodMeta = {
  family: string;
  description: string;
  uses_exogenous: boolean;
  library: string;
  compute_profile: string;
  limitations: string;
  code_snippet: string;
};

export type MethodRegistry = Record<string, MethodMeta>;

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
  source: { name: string; url: string | null };
  notes: string;
};

export type CaseRegistry = Record<string, CaseMeta>;
