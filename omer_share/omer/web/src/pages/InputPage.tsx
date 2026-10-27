import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type DrugCard, type ExtractResult } from '../api'
import { Disclaimer } from '../components/Disclaimer'
import { DrugSearch } from '../components/DrugSearch'
import { useRegimen } from '../state'

const EXAMPLE = ['Warfarin', 'Aspirin', 'Omeprazole', 'Simvastatin']

function splitList(s: string) {
  return s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)
}

export default function InputPage() {
  const { drugs, setDrugs, context, setContext, setResult } = useRegimen()
  const [tab, setTab] = useState<'search' | 'text'>('search')
  const [text, setText] = useState('')
  const [extractNote, setExtractNote] = useState<string | null>(null)
  const [extracted, setExtracted] = useState<{ text: string; r: ExtractResult } | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const nav = useNavigate()

  const add = (d: DrugCard) => setDrugs((cur) => (cur.some((x) => x.id === d.id) ? cur : [...cur, d]))
  const remove = (id: string) => setDrugs((cur) => cur.filter((d) => d.id !== id))

  const loadExample = async () => {
    setError(null)
    const found: DrugCard[] = []
    for (const name of EXAMPLE) {
      const r = await api.search(name)
      if (r[0]) found.push(r[0])
    }
    setDrugs(found)
  }

  const extract = async () => {
    setError(null)
    setExtracting(true)
    try {
      const r = await api.extract(text)
      const merged = [...drugs]
      r.drugs.forEach((d) => !merged.some((x) => x.id === d.id) && merged.push(d))
      setDrugs(merged)
      setExtracted({ text, r })
      setExtractNote(
        r.drugs.length
          ? `Found ${r.drugs.length}: ${r.drugs.map((d) => d.name).join(', ')}. Check the list before analysing.`
          : 'No known medication names found in the text.',
      )
    } catch (e) {
      setError(String((e as Error).message))
    } finally {
      setExtracting(false)
    }
  }

  const analyse = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await api.analyze(
        drugs.map((d) => d.id),
        context,
      )
      setResult(r)
      nav('/results')
    } catch (e) {
      setError(String((e as Error).message))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      <div>
        <p className="muted" style={{ maxWidth: 760 }}>
          Add two or more medications. Every pair is checked against known interactions (DrugBank, DDInter) and
          scored by a graph neural network, with a plain-language explanation of the likely mechanism.
        </p>
      </div>

      <div className="grid-2">
        <section className="card" aria-labelledby="meds-h">
          <div className="card-head">
            <h2 id="meds-h">Medications</h2>
            <button className="link small" onClick={loadExample}>
              Try the example: {EXAMPLE.join(', ')}
            </button>
          </div>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === 'search'} onClick={() => setTab('search')}>
              Search
            </button>
            <button role="tab" aria-selected={tab === 'text'} onClick={() => setTab('text')}>
              Paste prescription text
            </button>
          </div>
          {tab === 'search' ? (
            <>
              <DrugSearch onPick={add} exclude={drugs.map((d) => d.id)} autoFocus />
              <p className="muted small" style={{ marginTop: 6 }}>
                Generic and common brand names work (e.g. Coumadin, Zocor, Prilosec).
              </p>
            </>
          ) : (
            <div className="stack">
              <textarea
                rows={6}
                value={text}
                onChange={(e) => setText(e.target.value)}
                aria-label="Prescription or discharge text"
                placeholder={'e.g. Warfarin 5 mg once daily\nAspirin 81 mg daily\nOmeprazole 20 mg before breakfast'}
              />
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <button onClick={extract} disabled={!text.trim() || extracting}>
                  {extracting ? 'Reading…' : 'Find medications'}
                </button>
                <span className="muted small">
                  Biomedical NER finds medications, doses, frequencies and routes. Brand names are normalised with
                  RxNorm. The first run loads the models (up to 30 s).
                </span>
              </div>
              {extracted && <ExtractView text={extracted.text} r={extracted.r} />}
            </div>
          )}
          {extractNote && <p className="small" style={{ marginTop: 10 }}>{extractNote}</p>}

          <div className="chips" aria-label="Added medications">
            {drugs.length === 0 && <span className="muted small">No medications added yet.</span>}
            {drugs.map((d) => (
              <span className="chip" key={d.id}>
                {d.name}
                <button aria-label={`Remove ${d.name}`} onClick={() => remove(d.id)}>
                  ×
                </button>
              </span>
            ))}
          </div>

          <div style={{ marginTop: 16, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button className="primary" disabled={drugs.length < 2 || busy} onClick={analyse}>
              {busy ? 'Analysing…' : `Analyse ${drugs.length >= 2 ? (drugs.length * (drugs.length - 1)) / 2 : ''} pairs`}
            </button>
            {drugs.length > 0 && (
              <button className="link small" onClick={() => setDrugs([])}>
                Clear list
              </button>
            )}
          </div>
          {error && <p className="error" style={{ marginTop: 10 }}>{error}</p>}
        </section>

        <section className="card" aria-labelledby="ctx-h">
          <h2 id="ctx-h">Patient context</h2>
          <p className="muted small">Optional. Used to flag and re-rank warnings; nothing is stored on the server.</p>
          <div className="stack">
            <div className="field">
              <label htmlFor="age">Age</label>
              <input
                id="age"
                type="number"
                min={0}
                max={120}
                value={context.age ?? ''}
                onChange={(e) => setContext({ ...context, age: e.target.value === '' ? null : Number(e.target.value) })}
              />
            </div>
            <label className="check">
              <input type="checkbox" checked={context.kidney} onChange={(e) => setContext({ ...context, kidney: e.target.checked })} />
              Reduced kidney function
            </label>
            <label className="check">
              <input type="checkbox" checked={context.liver} onChange={(e) => setContext({ ...context, liver: e.target.checked })} />
              Reduced liver function
            </label>
            <div className="field">
              <label htmlFor="cond">Existing conditions (comma separated)</label>
              <input
                id="cond"
                type="text"
                placeholder="e.g. peptic ulcer, asthma"
                value={context.conditions.join(', ')}
                onChange={(e) => setContext({ ...context, conditions: splitList(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="allergy">Allergies (comma separated)</label>
              <input
                id="allergy"
                type="text"
                placeholder="e.g. penicillin, salicylate"
                value={context.allergies.join(', ')}
                onChange={(e) => setContext({ ...context, allergies: splitList(e.target.value) })}
              />
            </div>
          </div>
        </section>
      </div>
      <Disclaimer />
    </div>
  )
}

function Highlighted({ text, r }: { text: string; r: ExtractResult }) {
  const parts: ReactNode[] = []
  let at = 0
  r.mentions.forEach((m, i) => {
    if (m.start < at) return
    parts.push(text.slice(at, m.start))
    parts.push(
      <mark key={i} title={`${m.drugs.map((d) => d.name).join(' + ')} (${m.source})`}>
        {text.slice(m.start, m.end)}
      </mark>,
    )
    at = m.end
  })
  parts.push(text.slice(at))
  return <div className="ner-text">{parts}</div>
}

function ExtractView({ text, r }: { text: string; r: ExtractResult }) {
  return (
    <div className="stack">
      <Highlighted text={text} r={r} />
      {r.mentions.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>In the text</th>
                <th>Medication</th>
                <th>Dose</th>
                <th>Frequency</th>
                <th>Route</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {r.mentions.map((m, i) => (
                <tr key={i}>
                  <td>
                    {m.text}
                    <div className="small muted">
                      {m.source}
                      {m.via ? ` · ${m.via}` : ''}
                    </div>
                  </td>
                  <td>{m.drugs.map((d) => d.name).join(' + ')}</td>
                  <td className="num">{m.dose ?? '–'}</td>
                  <td>{m.frequency ?? '–'}</td>
                  <td>{m.route ?? '–'}</td>
                  <td>{m.duration ?? '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {r.unmatched.length > 0 && (
        <p className="small">
          Recognised as a medication but not matched to a drug in the database: <strong>{r.unmatched.join(', ')}</strong>.
          Add them by hand if needed.
        </p>
      )}
      <p className="small muted">
        Models: {r.models.join(', ')}, plus the local name index. Doses and frequencies are shown for checking only;
        the interaction analysis does not use them yet.
      </p>
    </div>
  )
}
