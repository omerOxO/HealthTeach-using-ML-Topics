import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type AlternativesResult, type FaersPair } from '../api'
import { compact } from './Charts'
import { SevIcon, severityLabel } from './Severity'

/** FAERS row of the pair evidence table. */
export function FaersEvidence({ a, b }: { a: string; b: string }) {
  const [f, setF] = useState<FaersPair | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setF(null)
    setError(null)
    api.faersPair(a, b).then(setF).catch((e) => setError(String(e.message)))
  }, [a, b])

  if (error) return <span className="muted">Could not reach openFDA: {error}</span>
  if (!f) return <span className="muted">Querying openFDA…</span>
  if (!f.ok) return <span className="muted">{f.error}</span>
  const signal = (f.reactions ?? []).filter((r) => r.ratio !== null && r.ratio >= 1.5).slice(0, 3)
  return (
    <div>
      <strong className="num">{(f.reports_both ?? 0).toLocaleString()}</strong> reports list both drugs
      {f.co_report_ratio != null && (
        <>
          {' '}
          (<span className="num">{f.co_report_ratio.toFixed(1)}×</span> the number expected if they were reported
          independently)
        </>
      )}
      ; <span className="num">{compact(f.serious ?? 0)}</span> serious,{' '}
      <span className="num">{compact(f.deaths ?? 0)}</span> with death as an outcome.
      {signal.length > 0 && (
        <div className="small" style={{ marginTop: 4 }}>
          Reported more often with the pair than with either drug alone:{' '}
          {signal.map((r) => `${r.term.toLowerCase()} (${r.ratio?.toFixed(1)}×)`).join(', ')}.
        </div>
      )}
      <div className="small muted" style={{ marginTop: 4 }}>
        Searched as “{f.names?.a.query}” + “{f.names?.b.query}”. Spontaneous reports: co-reporting is not causation,
        and counts reflect how widely drugs are used and reported.{' '}
        <Link to={`/faers?a=${a}&b=${b}`}>FAERS dashboard →</Link>
      </div>
    </div>
  )
}

/** Safer-alternatives row of the pair evidence table. */
export function SaferAlternatives({ a, b, meds, flagged }: { a: string; b: string; meds: string[]; flagged: boolean }) {
  const [r, setR] = useState<AlternativesResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const medKey = meds.join(',')
  useEffect(() => {
    if (!flagged) return
    setR(null)
    setError(null)
    api.alternatives(a, b, medKey ? medKey.split(',') : []).then(setR).catch((e) => setError(String(e.message)))
  }, [a, b, medKey, flagged])

  if (!flagged) return <span className="muted">Not needed: no interaction flagged for this pair.</span>
  if (error) return <span className="error">{error}</span>
  if (!r) return <span className="muted">Searching same-class and same-indication drugs…</span>
  const others = r.medications.length - 1
  return (
    <div className="alts">
      {r.for.map((f) => (
        <div key={f.replaces.id} className="alt-group">
          <div className="small">
            Instead of <strong>{f.replaces.name}</strong>{' '}
            <span className="muted">
              (now {severityLabel(f.current.worst).toLowerCase()}, risk {f.current.risk.toFixed(2)} with the other{' '}
              {others} drug{others === 1 ? '' : 's'})
            </span>
          </div>
          {f.suggestions.length === 0 ? (
            <div className="small muted">
              None of {f.n_candidates} same-class or same-indication drugs has a lower interaction risk with this list.
            </div>
          ) : (
            <ul className="alt-list">
              {f.suggestions.map((s) => (
                <li key={s.drug.id} title={s.reason}>
                  <span className="alt-name">
                    <SevIcon severity={s.worst} size={12} /> {s.drug.name}
                    <span className="small muted num" style={{ fontWeight: 400 }}>
                      risk {s.risk.toFixed(2)}
                    </span>
                  </span>
                  <span className="small muted">
                    {s.worst === 'none' ? 'no interaction found' : `worst ${severityLabel(s.worst).toLowerCase()}`} ·{' '}
                    {s.reason.split('; ').map((r) => r.split(':')[0]).join(', ')}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
      <details className="small muted">
        <summary>How these are chosen</summary>
        {r.note}
      </details>
    </div>
  )
}
