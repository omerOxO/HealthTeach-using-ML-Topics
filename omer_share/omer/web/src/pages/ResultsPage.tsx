import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { PairRow, Severity } from '../api'
import { Disclaimer } from '../components/Disclaimer'
import { PathChain } from '../components/PathChain'
import { SEV_VAR, SevIcon, SeverityBadge } from '../components/Severity'
import { useRegimen } from '../state'

const SEV_RANK: Record<Severity, number> = { major: 0, moderate: 1, minor: 2, none: 3 }
const STATUS_LABEL = { known: 'Known', predicted: 'Predicted', none: 'None found' }
const LEVEL_SEV: Record<string, Severity> = { HIGH: 'major', MODERATE: 'moderate', LOW: 'none' }

type SortKey = 'severity' | 'score' | 'pair'

function pairName(p: PairRow) {
  return `${p.a.name} + ${p.b.name}`
}

export default function ResultsPage() {
  const { result } = useRegimen()
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'severity', dir: 1 })
  const nav = useNavigate()

  const rows = useMemo(() => {
    if (!result) return []
    const r = [...result.pairs]
    const { key, dir } = sort
    r.sort((x, y) => {
      if (key === 'pair') return dir * pairName(x).localeCompare(pairName(y))
      if (key === 'score') return dir * (y.score - x.score)
      return dir * (SEV_RANK[x.severity] - SEV_RANK[y.severity]) || y.score - x.score
    })
    return key === 'severity' && dir === 1 ? result.pairs : r
  }, [result, sort])

  if (!result)
    return (
      <div className="card">
        <h1>No results yet</h1>
        <p>
          Add medications on the <Link to="/">Check</Link> page and press Analyse.
        </p>
      </div>
    )

  const { regimen } = result
  const levelSev = LEVEL_SEV[regimen.level]
  const top = result.pairs.find((p) => p.severity !== 'none') ?? result.pairs[0]
  const sortBtn = (key: SortKey, label: string) => (
    <button
      onClick={() => setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : 1 }))}
      aria-label={`Sort by ${label}`}
    >
      {label}
      {sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
    </button>
  )

  return (
    <div className="stack">
      <div>
        <div className="eyebrow" style={{ marginBottom: 6 }}>Medication list</div>
        <div className="chips" style={{ marginTop: 0 }}>
          {result.drugs.map((d) => (
            <span className="chip" key={d.id} style={{ paddingRight: 10 }}>{d.name}</span>
          ))}
        </div>
        {result.unknown.length > 0 && (
          <p className="error small" style={{ marginTop: 8 }}>Not recognised and skipped: {result.unknown.join(', ')}</p>
        )}
        {(result.normalised ?? []).length > 0 && (
          <p className="small muted" style={{ marginTop: 8 }}>
            Normalised with RxNorm:{' '}
            {result.normalised!.map((n) => `${n.input} → ${n.drugs.map((d) => d.name).join(' + ')}${n.match === 'approximate' ? ' (closest match, please check)' : ''}`).join('; ')}
          </p>
        )}
      </div>

      <div className="kpis">
        <div className="kpi accent">
          <div className="kpi-label">Regimen risk</div>
          <div className="hero" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <SevIcon severity={levelSev} size={28} />
            {regimen.level}
          </div>
          <div className="kpi-foot">Worst severity among flagged pairs</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Combined risk score</div>
          <div className="kpi-value num">{regimen.score.toFixed(2)}</div>
          <div
            className="meter"
            role="meter"
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={regimen.score}
            aria-label="Combined regimen risk score"
          >
            <span style={{ width: `${Math.max(2, regimen.score * 100)}%`, background: SEV_VAR[levelSev] }} />
          </div>
          <div className="kpi-foot">1 − Π(1 − severity weight × probability)</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Pairs checked</div>
          <div className="kpi-value">{regimen.pairs_checked}</div>
          <div className="kpi-foot">every combination of {result.drugs.length} drugs</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Flagged</div>
          <div className="kpi-value">{regimen.flagged}</div>
          <div className="kpi-foot">known or predicted interactions</div>
        </div>
      </div>

      {result.alerts.length > 0 && (
        <div className="card">
          <h2>Patient-context alerts</h2>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {result.alerts.map((a, i) => (
              <li key={i}>
                <strong>{a.drug.name}:</strong> {a.message} <span className="tag">{a.source}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {top && top.severity !== 'none' && (
        <div className="card">
          <div className="card-head">
            <h2>Why {pairName(top)} is the top concern</h2>
            <Link className="small" to={`/pair/${top.a.id}/${top.b.id}`}>
              Pair details →
            </Link>
          </div>
          <p style={{ marginBottom: 10 }}>
            <PathChain steps={top.path_chain} />
          </p>
          <p className="explain" style={{ marginBottom: 0 }}>
            {top.explanation}
          </p>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <h2>Drug pairs</h2>
          <span className="muted small">Ranked by severity, then evidence and score. Select a row for details.</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th aria-hidden />
                <th>{sortBtn('severity', 'Severity')}</th>
                <th>{sortBtn('pair', 'Pair')}</th>
                <th>Evidence</th>
                <th className="num">{sortBtn('score', 'Risk')}</th>
                <th>Top predicted effects</th>
                <th>Mechanism</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr
                  key={`${p.a.id}-${p.b.id}`}
                  className="row-link"
                  tabIndex={0}
                  onClick={() => nav(`/pair/${p.a.id}/${p.b.id}`)}
                  onKeyDown={(e) => e.key === 'Enter' && nav(`/pair/${p.a.id}/${p.b.id}`)}
                >
                  <td className="rowbar">
                    <span style={{ background: SEV_VAR[p.severity] }} />
                  </td>
                  <td>
                    <SeverityBadge severity={p.severity} />
                    <div className="muted small" style={{ marginTop: 4, maxWidth: 160 }}>
                      {p.severity_source}
                    </div>
                  </td>
                  <td style={{ fontWeight: 600, minWidth: 160 }}>
                    {pairName(p)}
                    {p.context_flag && (
                      <div>
                        <span className="tag">patient context</span>
                      </div>
                    )}
                  </td>
                  <td>
                    {STATUS_LABEL[p.status]}
                    <div className="muted small">
                      {[p.sources.drugbank_tdc && 'DrugBank', p.sources.primekg && 'PrimeKG', p.sources.ddinter && 'DDInter']
                        .filter(Boolean)
                        .join(', ') || 'model only'}
                    </div>
                  </td>
                  <td className="num" style={{ minWidth: 70 }}>
                    <strong>{p.score.toFixed(2)}</strong>
                    <div className="meter" style={{ height: 6, marginTop: 6, width: 56 }}>
                      <span style={{ width: `${p.score * 100}%`, background: 'var(--accent)' }} />
                    </div>
                  </td>
                  <td style={{ minWidth: 170 }}>
                    {p.effects
                      .slice(0, 3)
                      .map((e) => e.name)
                      .join(', ')}
                  </td>
                  <td className="small" style={{ minWidth: 220 }}>
                    {p.mechanism ?? <span className="muted">No interaction known or predicted (not proof of safety)</span>}
                    {p.mechanism && p.mechanism_source === 'predicted' && <span className="tag" style={{ marginLeft: 6 }}>predicted</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small" style={{ marginTop: 10 }}>
          Risk = model probability that the pair interacts ({result.model.name}, decision threshold{' '}
          {result.model.threshold.toFixed(2)}). Effects are polypharmacy side effects predicted for the pair, ranked by
          how far they exceed their usual rate; effects implied by the mechanism come first. They are learned from
          TWOSIDES, whose signals are statistical associations from spontaneous reports and partly reflect the
          conditions being treated.
        </p>
      </div>
      <Disclaimer text={result.disclaimer} />
    </div>
  )
}
