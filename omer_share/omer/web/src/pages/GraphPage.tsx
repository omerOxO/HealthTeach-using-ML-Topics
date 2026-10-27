import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, type GraphNode, type Neighbourhood } from '../api'
import { DrugSearch } from '../components/DrugSearch'
import { KGGraph, TYPE_LABEL } from '../components/KGGraph'

export default function GraphPage() {
  const [params, setParams] = useSearchParams()
  const drug = params.get('drug') ?? 'DB00682'
  const withId = params.get('with') ?? undefined
  const [g, setG] = useState<Neighbourhood | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)

  useEffect(() => {
    setError(null)
    setSelected(null)
    api.graph(drug, withId).then(setG).catch((e) => setError(String(e.message)))
  }, [drug, withId])

  const onNodeClick = useCallback((n: GraphNode) => setSelected(n), [])

  return (
    <div className="stack">
      <div>
        <p className="muted" style={{ maxWidth: 760 }}>
          A drug&apos;s neighbourhood in the knowledge graph: targets, metabolising enzymes, transporters, single-drug
          side effects, indications and ATC classes. The most specific neighbours are shown first (up to 12 per type).
        </p>
      </div>
      <div className="grid-2">
        <div className="card">
          <div className="card-head" style={{ alignItems: 'center' }}>
            <h2>{g ? g.drug.name : '…'}</h2>
            {withId && g?.highlight && (
              <span className="small">
                Highlighting the explanation path to the second drug ·{' '}
                <button className="link small" onClick={() => setParams({ drug })}>
                  clear
                </button>
              </span>
            )}
          </div>
          {error && <p className="error">{error}</p>}
          {g && (
            <KGGraph
              nodes={g.nodes}
              edges={g.edges}
              highlight={g.highlight ?? []}
              tall
              layout="concentric"
              onNodeClick={onNodeClick}
              ariaLabel={`Knowledge-graph neighbourhood of ${g.drug.name}`}
            />
          )}
        </div>
        <div className="stack">
          <div className="card">
            <h2>Explore a drug</h2>
            <DrugSearch onPick={(d) => setParams({ drug: d.id })} placeholder="Jump to a drug" />
            {g && (
              <div style={{ marginTop: 14 }}>
                <p className="small muted" style={{ marginBottom: 6 }}>
                  Compare with a second drug to highlight the connecting path
                </p>
                <DrugSearch onPick={(d) => setParams({ drug, with: d.id })} exclude={[drug]} placeholder="Second drug" />
              </div>
            )}
          </div>
          {selected && (
            <div className="card">
              <h3>{selected.label}</h3>
              <p className="small muted">{TYPE_LABEL[selected.type] ?? selected.type}</p>
            </div>
          )}
          {g && (
            <div className="card">
              <h3>Neighbour counts</h3>
              <table>
                <tbody>
                  {Object.entries(g.counts).map(([k, v]) => (
                    <tr key={k}>
                      <td>{TYPE_LABEL[k] ?? k}</td>
                      <td className="num" style={{ textAlign: 'right' }}>
                        {v.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {g.drug.mechanism_of_action && (
                <details style={{ marginTop: 10 }}>
                  <summary>Mechanism of action</summary>
                  <p className="small" style={{ marginTop: 8 }}>
                    {g.drug.mechanism_of_action}
                  </p>
                </details>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
