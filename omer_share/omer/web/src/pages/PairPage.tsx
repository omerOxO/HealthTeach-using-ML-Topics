import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type PairDetail } from '../api'
import { Disclaimer } from '../components/Disclaimer'
import { KGGraph } from '../components/KGGraph'
import { PathChain } from '../components/PathChain'
import { SeverityBadge } from '../components/Severity'
import { FaersEvidence, SaferAlternatives } from '../components/PairExtras'
import { useRegimen } from '../state'

const KIND_LABEL: Record<string, string> = {
  enzyme: 'Shared metabolising enzyme',
  target: 'Shared drug target',
  transporter: 'Shared transporter',
  carrier: 'Shared carrier protein',
  side_effect: 'Shared side effect',
  in_class: 'Same drug class',
  indication: 'Same indication',
  contraindication: 'Shared contraindication',
  off_label: 'Same off-label use',
  ppi: 'Linked through interacting proteins',
}

const NODE_TYPE_LABEL: Record<string, string> = {
  protein: 'protein',
  effect: 'side effect',
  disease: 'disease',
  atc: 'drug class',
  drug: 'drug',
}

/** GNNExplainer attributions: the neighbours whose edges the model needed to keep its score. */
function ModelLookedAt({ d }: { d: PairDetail }) {
  const m = d.model_explanation!
  const onPath = new Set(d.paths.flatMap((p) => p.via.map((v) => v.id)))
  const drugName = (idx: number) => (idx === d.a.idx ? d.a.name : idx === d.b.idx ? d.b.name : '')
  return (
    <div className="card">
      <div className="card-head">
        <h2>What the model looked at</h2>
        <span className="small muted">GNNExplainer on the served R-GCN</span>
      </div>
      <p className="small muted">
        The neighbours whose links to {d.a.name} or {d.b.name} the model needed to keep its score, out of{' '}
        {m.n_edges.toLocaleString()} links feeding the two drugs. Importance is relative to the top neighbour.{' '}
        <span className="tag">also a KG path</span> marks neighbours that the explanation above uses too.
      </p>
      {m.top_nodes.map((n) => (
        <div className="probbar" key={n.id} style={{ gridTemplateColumns: 'minmax(0, 1fr) 120px 48px' }}>
          <span>
            <strong>{n.name}</strong>{' '}
            <span className="muted small">
              {NODE_TYPE_LABEL[n.type] ?? n.type}
              {n.rels.length > 0 && ` · ${n.rels.join(', ')}`}
              {n.drugs.length > 0 && ` of ${n.shared ? 'both drugs' : n.drugs.map(drugName).join(', ')}`}
            </span>{' '}
            {onPath.has(n.id) && <span className="tag">also a KG path</span>}
          </span>
          <div className="track" aria-hidden>
            <div className="fill" style={{ width: `${n.score * 100}%` }} />
          </div>
          <span className="num" style={{ textAlign: 'right' }}>
            {n.score.toFixed(2)}
          </span>
        </div>
      ))}
      <table style={{ marginTop: 12 }}>
        <tbody>
          <tr>
            <th>Score with every link</th>
            <td className="num">{m.score.toFixed(2)}</td>
          </tr>
          <tr>
            <th>Only the links of these {m.top_nodes.length} neighbours</th>
            <td className="num">{m.score_only_top.toFixed(2)}</td>
          </tr>
          <tr>
            <th>Every link except theirs</th>
            <td className="num">{m.score_without_top.toFixed(2)}</td>
          </tr>
          <tr>
            <th>No links at all (the drugs' own features only)</th>
            <td className="num">{m.score_no_edges.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
      <p className="small muted" style={{ marginTop: 8 }}>
        This is what drives the model's score, not a proof of mechanism. If the score with only these links stays
        close to the full score, the attribution is faithful; a small drop without them means other links carry the
        same signal.
      </p>
    </div>
  )
}

export default function PairPage() {
  const { a = '', b = '' } = useParams()
  const [d, setD] = useState<PairDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setD(null)
    setError(null)
    api.pair(a, b).then(setD).catch((e) => setError(String(e.message)))
  }, [a, b])

  const { result } = useRegimen()
  // the rest of the analysed list, so alternatives are checked against every other drug the patient takes
  const meds = useMemo(() => (result?.drugs ?? []).map((x) => x.id), [result])
  const highlight = useMemo(() => d?.graph.edges.filter((e) => e.hl) ?? [], [d])

  if (error) return <p className="error">Could not load this pair: {error}</p>
  if (!d) return <p className="muted">Loading pair…</p>

  const maxP = Math.max(...d.all_effects.map((e) => e.prob), 0.01)
  return (
    <div className="stack">
      <div>
        <Link to="/results" className="small">
          ← Back to results
        </Link>
        <h1 style={{ marginTop: 8 }}>
          {d.a.name} + {d.b.name}
        </h1>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <SeverityBadge severity={d.severity} />
          <span className="muted small">{d.severity_source}</span>
          <span className="tag">
            {d.status === 'known' ? 'Known interaction' : d.status === 'predicted' ? 'Predicted interaction' : 'No interaction found'}
          </span>
          <span className="small">
            Model probability <strong className="num">{d.score.toFixed(2)}</strong>{' '}
            <span className="muted">(threshold {d.threshold.toFixed(2)})</span>
          </span>
        </div>
      </div>

      <div className="card">
        <h2>What is going on</h2>
        <p style={{ marginBottom: 10 }}>
          <PathChain steps={d.path_chain} />
        </p>
        <p className="explain">{d.explanation}</p>
        {d.status === 'none' && (
          <p className="small muted" style={{ marginTop: 10 }}>
            No interaction is recorded or predicted, but missing data is common: this is not evidence that the
            combination is safe.
          </p>
        )}
      </div>

      <div className="grid-2-even" style={{ gridTemplateColumns: "minmax(0, 1fr)" }}>
        <div className="card">
          <div className="card-head">
            <h2>Explanation path</h2>
            <Link className="small" to={`/graph?drug=${d.a.id}&with=${d.b.id}`}>
              Open in graph explorer →
            </Link>
          </div>
          <KGGraph
            nodes={d.graph.nodes}
            edges={d.graph.edges}
            highlight={highlight}
            layout="pair"
            ariaLabel={`Knowledge-graph paths linking ${d.a.name} and ${d.b.name}`}
          />
        </div>
        <div className="card">
          <h2>Ranked mechanisms</h2>
          {d.paths.length === 0 && <p className="muted">No shared enzyme, target, transporter, side effect or class.</p>}
          <ol style={{ paddingLeft: 18, margin: 0 }}>
            {d.paths.map((p, i) => (
              <li key={i} style={{ marginBottom: 10 }}>
                <strong>{KIND_LABEL[p.kind] ?? p.kind}:</strong> {p.via.map((v) => v.name).join(' – ')}
                {p.matches_effect && <span className="tag" style={{ marginLeft: 6 }}>matches predicted effect</span>}
                <div className="small muted">{p.text}</div>
              </li>
            ))}
          </ol>
          <p className="small muted" style={{ marginTop: 8 }}>
            Ranked by relation type and specificity: a protein shared by few drugs counts for more than one shared by
            hundreds.
          </p>
        </div>
      </div>

      {d.model_explanation && <ModelLookedAt d={d} />}

      <div className="grid-2">
        <div className="card">
          <h2>Predicted side effects for the pair</h2>
          <p className="small muted">
            Probability from the side-effect head (trained on TWOSIDES).{' '}
            <span className="tag">TWOSIDES</span> marks effects already reported for this pair.
          </p>
          {d.all_effects.map((e) => (
            <div className="probbar" key={e.name}>
              <span>
                {e.name} {e.twosides && <span className="tag">TWOSIDES</span>}
              </span>
              <div className="track" aria-hidden>
                <div className="fill" style={{ width: `${(e.prob / maxP) * 100}%` }} />
              </div>
              <span className="num" style={{ textAlign: 'right' }}>
                {(e.prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>

        <div className="card">
          <h2>Evidence</h2>
          <table>
            <tbody>
              <tr>
                <th>DrugBank (TDC benchmark)</th>
                <td>{d.sources.drugbank_tdc ? d.mechanism : 'not listed'}</td>
              </tr>
              <tr>
                <th>DrugBank via PrimeKG</th>
                <td>{d.sources.primekg ? 'interaction listed' : 'not listed'}</td>
              </tr>
              <tr>
                <th>DDInter severity</th>
                <td>{d.sources.ddinter_level ?? (d.sources.ddinter ? 'listed, level unknown' : 'not listed')}</td>
              </tr>
              <tr>
                <th>TWOSIDES</th>
                <td>
                  {d.twosides_count
                    ? `${d.twosides_count} significant side-effect associations, e.g. ${d.twosides_examples.slice(0, 6).join(', ')}`
                    : 'pair not in TWOSIDES'}
                  <div className="small muted">Statistical associations from spontaneous reports, not proven causation.</div>
                </td>
              </tr>
              <tr>
                <th>FAERS report count</th>
                <td>
                  <FaersEvidence a={d.a.id} b={d.b.id} />
                </td>
              </tr>
              <tr>
                <th>Safer alternatives</th>
                <td>
                  <SaferAlternatives a={d.a.id} b={d.b.id} meds={meds} flagged={d.severity !== 'none'} />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2-even">
        {(['a', 'b'] as const).map((k) => {
          const info = d.drug_info[k]
          const card = d[k]
          return (
            <div className="card" key={k}>
              <h3>{card.name}</h3>
              <p className="small muted">
                {card.id}
                {info.atc ? ` · ${info.atc.split('; ')[0]}` : ''} · {info.n_targets} targets · {info.n_enzymes} enzymes ·{' '}
                {info.n_side_effects} known side effects
              </p>
              {info.mechanism_of_action && (
                <details>
                  <summary>Mechanism of action</summary>
                  <p className="small" style={{ marginTop: 8 }}>
                    {info.mechanism_of_action}
                  </p>
                </details>
              )}
            </div>
          )
        })}
      </div>
      <Disclaimer text={d.disclaimer} />
    </div>
  )
}
