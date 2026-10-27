import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, type DrugCard, type FaersOverview, type FaersPair } from '../api'
import { ChartCard, Columns } from '../components/Charts'
import { Disclaimer } from '../components/Disclaimer'
import { DrugSearch } from '../components/DrugSearch'
import { useRegimen } from '../state'

function Picker({ label, value, onPick }: { label: string; value: DrugCard | null; onPick: (d: DrugCard) => void }) {
  return (
    <div className="field" style={{ flex: '1 1 240px' }}>
      <label>{label}</label>
      {value && (
        <div className="small">
          <strong>{value.name}</strong> <span className="muted">{value.class ?? ''}</span>
        </div>
      )}
      <DrugSearch onPick={onPick} placeholder={value ? 'Change drug…' : 'Search a medication'} />
    </div>
  )
}

function BiasNote({ text }: { text: string }) {
  return (
    <div className="disclaimer" role="note">
      <strong>Reporting bias.</strong> {text}
    </div>
  )
}

export default function FaersPage() {
  const [params, setParams] = useSearchParams()
  const { result } = useRegimen()
  const [a, setA] = useState<DrugCard | null>(null)
  const [b, setB] = useState<DrugCard | null>(null)
  const [f, setF] = useState<FaersPair | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [overview, setOverview] = useState<FaersOverview | null>(null)
  const aId = params.get('a')
  const bId = params.get('b')

  useEffect(() => {
    if (!aId || !bId) return
    setBusy(true)
    setError(null)
    setF(null)
    api
      .faersPair(aId, bId)
      .then((r) => {
        setF(r)
        if (r.drugs) {
          setA(r.drugs.a)
          setB(r.drugs.b)
        }
      })
      .catch((e) => setError(String(e.message)))
      .finally(() => setBusy(false))
  }, [aId, bId])

  const regimenIds = (result?.drugs ?? []).slice(0, 10).map((d) => d.id)
  const regimenKey = regimenIds.join(',')
  useEffect(() => {
    if (!regimenKey) return
    api.faersOverview(regimenKey.split(',')).then(setOverview).catch(() => setOverview(null))
  }, [regimenKey])

  const pick = (which: 'a' | 'b') => (d: DrugCard) => {
    if (which === 'a') setA(d)
    else setB(d)
    const na = which === 'a' ? d : a
    const nb = which === 'b' ? d : b
    if (na && nb && na.id !== nb.id) setParams({ a: na.id, b: nb.id })
  }

  const reactions = f?.reactions ?? []
  const maxShare = Math.max(...reactions.map((r) => Math.max(r.share, r.share_alone)), 0.01)

  return (
    <div className="stack">
      <p className="muted" style={{ maxWidth: 820 }}>
        Counts of US FDA Adverse Event Reporting System (FAERS) reports that list both drugs, from the openFDA API.
        Results are cached locally for 30 days. Use them as context for a prediction, not as proof of an interaction.
      </p>

      <div className="card">
        <h2>Choose a pair</h2>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Picker label="Drug A" value={a} onPick={pick('a')} />
          <Picker label="Drug B" value={b} onPick={pick('b')} />
        </div>
        {busy && <p className="muted small">Querying openFDA (about 5 seconds the first time)…</p>}
        {error && <p className="error small">{error}</p>}
      </div>

      {f && !f.ok && <p className="error">{f.error}</p>}
      {f?.ok && (
        <>
          <div className="kpis">
            <div className="kpi accent">
              <div className="kpi-label">Reports listing both</div>
              <div className="kpi-value num">{(f.reports_both ?? 0).toLocaleString()}</div>
              <div className="kpi-foot">
                {f.drugs?.a.name} + {f.drugs?.b.name}
              </div>
            </div>
            <div className="kpi">
              <div className="kpi-label">Co-report ratio</div>
              <div className="kpi-value num">{f.co_report_ratio != null ? `${f.co_report_ratio.toFixed(1)}×` : '–'}</div>
              <div className="kpi-foot">
                observed ÷ expected ({(f.expected_both ?? 0).toLocaleString()}) if reported independently
              </div>
            </div>
            <div className="kpi">
              <div className="kpi-label">Serious</div>
              <div className="kpi-value num">{(f.serious ?? 0).toLocaleString()}</div>
              <div className="kpi-foot">
                {f.reports_both ? `${Math.round(((f.serious ?? 0) / f.reports_both) * 100)}% of co-reports` : ''},{' '}
                {(f.deaths ?? 0).toLocaleString()} with death
              </div>
            </div>
            <div className="kpi">
              <div className="kpi-label">Each drug overall</div>
              <div className="kpi-value num" style={{ fontSize: '1.2rem' }}>
                {(f.reports_a ?? 0).toLocaleString()} / {(f.reports_b ?? 0).toLocaleString()}
              </div>
              <div className="kpi-foot">of {(f.total_reports ?? 0).toLocaleString()} FAERS reports</div>
            </div>
          </div>

          <BiasNote text={f.bias_note} />

          <div className="grid-2">
            <ChartCard
              title="Most reported reactions with both drugs"
              sub="Share of co-reports (red) vs. the higher share among reports of either drug (grey)"
              table={
                <table>
                  <thead>
                    <tr>
                      <th>Reaction (MedDRA)</th>
                      <th className="num">Reports</th>
                      <th className="num">Share with pair</th>
                      <th className="num">Share alone</th>
                      <th className="num">Ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reactions.map((r) => (
                      <tr key={r.term}>
                        <td>{r.term}</td>
                        <td className="num">{r.count.toLocaleString()}</td>
                        <td className="num">{(r.share * 100).toFixed(1)}%</td>
                        <td className="num">{(r.share_alone * 100).toFixed(1)}%</td>
                        <td className="num">{r.ratio != null ? `${r.ratio.toFixed(2)}×` : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              }
            >
              {reactions.map((r) => (
                <div className="faers-row" key={r.term}>
                  <span>
                    {r.term}
                    {r.ratio != null && r.ratio >= 1.5 && <span className="tag" style={{ marginLeft: 6 }}>{r.ratio.toFixed(1)}× alone</span>}
                  </span>
                  <div className="faers-bars" aria-hidden>
                    <div className="fill" style={{ width: `${(r.share / maxShare) * 100}%` }} />
                    <div className="fill alone" style={{ width: `${(r.share_alone / maxShare) * 100}%` }} />
                  </div>
                  <span className="num" style={{ textAlign: 'right' }}>
                    {r.count.toLocaleString()}
                  </span>
                </div>
              ))}
              <p className="small muted" style={{ marginTop: 8 }}>
                A ratio above 1 means the reaction makes up a larger share of the pair's reports than of either drug's
                own reports. It is a hint worth checking, not a signal test.
              </p>
            </ChartCard>

            <ChartCard
              title="Co-reports by year received"
              sub="Reporting volume changes with FAERS itself, so trends are hard to interpret"
              table={
                <table>
                  <tbody>
                    {(f.by_year ?? []).map((y) => (
                      <tr key={y.year}>
                        <td>{y.year}</td>
                        <td className="num">{y.count.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              }
            >
              <Columns
                labels={(f.by_year ?? []).map((y) => y.year.slice(2))}
                values={(f.by_year ?? []).map((y) => y.count)}
                tipTitle={(i) => (f.by_year ?? [])[i].year}
                valueLabel="Reports"
                ariaLabel="FAERS co-reports by year"
                color="var(--red)"
              />
            </ChartCard>
          </div>
          <p className="small muted">
            Query: <span className="mono">{f.query}</span>. Names: {f.drugs?.a.name} → “{f.names?.a.query}” ({f.names?.a.how}),{' '}
            {f.drugs?.b.name} → “{f.names?.b.query}” ({f.names?.b.how}). Products containing the drug with other
            ingredients are included.
          </p>
        </>
      )}

      {result && (
        <div className="card">
          <div className="card-head">
            <h2>Your current list</h2>
            <span className="small muted">Co-reports for every pair of the first 10 drugs analysed</span>
          </div>
          {!overview && <p className="muted small">Querying openFDA…</p>}
          {overview && !overview.ok && <p className="muted small">{overview.error}</p>}
          {overview?.ok && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th className="num">FAERS co-reports</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {overview.pairs?.map((p) => (
                    <tr key={p.a.id + p.b.id}>
                      <td>
                        {p.a.name} + {p.b.name}
                      </td>
                      <td className="num">{p.reports.toLocaleString()}</td>
                      <td>
                        <button className="link small" onClick={() => setParams({ a: p.a.id, b: p.b.id })}>
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
      {!result && !f && (
        <p className="small muted">
          Tip: after you <Link to="/">analyse a medication list</Link>, this page also lists FAERS co-reports for every
          pair on it.
        </p>
      )}
      <Disclaimer />
    </div>
  )
}
