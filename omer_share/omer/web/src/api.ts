export type Severity = 'major' | 'moderate' | 'minor' | 'none'

export interface DrugCard {
  idx: number
  id: string
  name: string
  class: string | null
  matched?: string | null
}

export interface Effect {
  name: string
  prob: number
  twosides: boolean
}

export type ChainStep = { label: string; type: string } | { arrow: string; rel: string }

export interface PairRow {
  a: DrugCard
  b: DrugCard
  status: 'known' | 'predicted' | 'none'
  score: number
  severity: Severity
  severity_source: string
  mechanism: string | null
  mechanism_source: 'DrugBank' | 'predicted'
  type_confidence: number | null
  effects: Effect[]
  sources: { drugbank_tdc: boolean; primekg: boolean; ddinter: boolean; ddinter_level: string | null }
  explanation: string
  path_chain: ChainStep[]
  context_flag?: boolean
}

export interface GraphNode {
  id: number
  label: string
  type: string
  focus: boolean
}
export interface GraphEdge {
  source: number
  target: number
  rel: string
  hl?: boolean
}

export interface PathItem {
  kind: string
  score: number
  text: string
  via: { id: number; name: string; type: string }[]
  matches_effect: boolean
}

export interface PairDetail extends PairRow {
  paths: PathItem[]
  effect_tail: { text: string; effect: string } | null
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
  all_effects: Effect[]
  twosides_count: number
  twosides_examples: string[]
  threshold: number
  drug_info: Record<'a' | 'b', DrugInfo>
  model_explanation: ModelExplanation | null
  disclaimer: string
}

export interface ModelExplanation {
  method: string
  score: number
  score_without_top: number
  score_only_top: number
  score_no_edges: number
  n_edges: number
  n_selected: number
  top_nodes: { id: number; name: string; type: string; score: number; drugs: number[]; rels: string[]; shared: boolean }[]
  top_edges: { source: number; source_name: string; drug: number; rel: string; weight: number }[]
}

export interface DrugInfo {
  mechanism_of_action: string | null
  categories: string | null
  atc: string | null
  n_targets: number
  n_enzymes: number
  n_side_effects: number
}

export interface PatientContext {
  age: number | null
  kidney: boolean
  liver: boolean
  conditions: string[]
  allergies: string[]
}

export interface Alert {
  drug: DrugCard
  reason: string
  message: string
  source: string
}

export interface AnalyzeResult {
  drugs: DrugCard[]
  unknown: string[]
  normalised?: { input: string; drugs: DrugCard[]; source: string; match: string | null }[]
  regimen: { level: 'HIGH' | 'MODERATE' | 'LOW'; score: number; pairs_checked: number; flagged: number }
  pairs: PairRow[]
  alerts: Alert[]
  disclaimer: string
  model: { name: string; threshold: number }
}

export interface Neighbourhood {
  nodes: GraphNode[]
  edges: GraphEdge[]
  counts: Record<string, number>
  drug: DrugCard & DrugInfo
  highlight?: GraphEdge[]
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      msg = typeof body.detail === 'string' ? body.detail : body.detail?.message ?? msg
    } catch {
      /* keep status text */
    }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export const api = {
  search: (q: string) => req<DrugCard[]>(`/api/drugs/search?q=${encodeURIComponent(q)}`),
  analyze: (drugs: string[], context: PatientContext) =>
    req<AnalyzeResult>('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drugs, context }),
    }),
  extract: (text: string) =>
    req<ExtractResult>('/api/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
  pair: (a: string, b: string) => req<PairDetail>(`/api/pair?a=${a}&b=${b}`),
  graph: (id: string, highlight?: string) =>
    req<Neighbourhood>(`/api/graph/drug/${id}${highlight ? `?highlight=${highlight}` : ''}`),
  metrics: () => req<Metrics>('/api/metrics'),
  eda: () => req<Eda>('/api/eda'),
  modelCard: () => req<{ disclaimer: string; card: string; model: Record<string, unknown>; threshold: number }>(
    '/api/model-card',
  ),
  alternatives: (a: string, b: string, meds: string[]) =>
    req<AlternativesResult>(`/api/alternatives?a=${a}&b=${b}&meds=${meds.join(',')}`),
  faersPair: (a: string, b: string) => req<FaersPair>(`/api/faers/pair?a=${a}&b=${b}`),
  faersOverview: (ids: string[]) => req<FaersOverview>(`/api/faers/overview?ids=${ids.join(',')}`),
  me: () => req<User>('/api/auth/me'),
  login: (username: string, password: string) =>
    req<User>('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }),
  logout: () => req<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  audit: (level?: string) => req<AuditLog>(`/api/audit?limit=200${level ? `&level=${level}` : ''}`),
}

/* ---------------------------------------------------------------- spec extras */
export interface Mention {
  text: string
  start: number
  end: number
  drugs: DrugCard[]
  source: string
  via: string | null
  score: number | null
  dose: string | null
  frequency: string | null
  route: string | null
  duration: string | null
}
export interface ExtractResult {
  drugs: DrugCard[]
  mentions: Mention[]
  unmatched: string[]
  models: string[]
}

export interface QuickPair {
  with: DrugCard
  status: 'known' | 'predicted' | 'none'
  severity: Severity
  score: number
}
export interface RiskProfile {
  worst: Severity
  risk: number
  pairs: QuickPair[]
}
export interface Alternative extends RiskProfile {
  drug: DrugCard
  reason: string
  risk_drop: number
}
export interface AlternativesResult {
  for: { replaces: DrugCard; current: RiskProfile; n_candidates: number; suggestions: Alternative[] }[]
  medications: DrugCard[]
  note: string
}

export interface FaersReaction {
  term: string
  count: number
  share: number
  share_alone: number
  ratio: number | null
}
export interface FaersPair {
  ok: boolean
  error?: string
  bias_note: string
  names?: Record<'a' | 'b', { query: string; how: string }>
  drugs?: Record<'a' | 'b', DrugCard>
  total_reports?: number
  reports_a?: number
  reports_b?: number
  reports_both?: number
  expected_both?: number
  co_report_ratio?: number | null
  serious?: number
  deaths?: number
  reactions?: FaersReaction[]
  by_year?: { year: string; count: number }[]
  source?: string
  query?: string
}
export interface FaersOverview {
  ok: boolean
  error?: string
  bias_note?: string
  pairs?: { a: DrugCard; b: DrugCard; reports: number }[]
}

export interface User {
  username: string | null
  role: 'admin' | 'clinician' | 'guest'
}
export interface AuditEntry {
  id: number
  ts: string
  role: string
  drugs: { id: string; name: string }[]
  n_drugs: number
  unknown: number
  pairs_checked: number
  flagged: number
  level: 'HIGH' | 'MODERATE' | 'LOW'
  score: number
  top_pairs: { a: string; b: string; severity: Severity; status: string; score: number }[]
  context_used: string[]
  model: string
}
export interface AuditLog {
  total: number
  levels: Record<string, number>
  entries: AuditEntry[]
}

export interface SplitMetrics {
  auroc: number
  auprc: number
  f1: number
  threshold: number
  n_pos: number
  n_unlabelled: number
  'hits@20': number
  'hits@50': number
  'hits@100': number
}
export interface RunResult {
  val: SplitMetrics
  test: SplitMetrics
  test_hard: SplitMetrics
  train_seconds?: number
  type?: { accuracy: number; macro_f1: number; n: number }
  side_effects?: { mean_auprc: number; mean_auroc: number; n_labels: number; n_pairs: number }
}
export type Metrics = Record<string, Record<string, Record<string, RunResult>>>

export interface HistBin {
  lo: number
  hi: number
  count: number
}
export interface Eda {
  graph: { nodes: Record<string, number>; edges: Record<string, number> }
  ddi: {
    n_drugs: number
    possible_pairs: number
    positives: Record<string, number>
    density: Record<string, number>
    source_overlap: Record<string, number>
    degree_hist: HistBin[]
    drugs_without_interactions: number
    top_drugs: { name: string; id: string; degree: number }[]
    type_counts: { type_id: number; template: string; count: number }[]
    ddinter_levels: Record<string, number>
  }
  cold_start: Record<string, number>
  twosides: {
    pairs: number
    drugs: number
    side_effects: number
    top_side_effects: { name: string; pairs: number }[]
    side_effect_freq_hist: HistBin[]
    positive_rate_median: number
    side_effects_per_pair_median: number
  }
}
