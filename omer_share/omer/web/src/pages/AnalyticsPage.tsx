import { useEffect, useMemo, useState } from 'react'
import { api, type Eda, type Metrics, type SplitMetrics } from '../api'
import { ChartCard, Columns, GroupedColumns, HBars } from '../components/Charts'

const MODEL_LABEL: Record<string, string> = {
  fp_lr: 'Fingerprint LR',
  fp_xgb: 'Fingerprint XGBoost',
  gnn_rgcn: 'R-GCN',
  gnn_rgat: 'R-GAT',
  gnn_rgcn_gin: 'R-GCN + GIN',
}
const MODEL_ORDER = ['fp_lr', 'fp_xgb', 'gnn_rgcn', 'gnn_rgat', 'gnn_rgcn_gin']
const METRICS: { key: keyof SplitMetrics; label: string }[] = [
  { key: 'auroc', label: 'AUROC' },
  { key: 'auprc', label: 'AUPRC' },
  { key: 'f1', label: 'F1' },
  { key: 'hits@20', label: 'Hits@20' },
  { key: 'hits@50', label: 'Hits@50' },
  { key: 'hits@100', label: 'Hits@100' },
]
const SPLITS = [
  { key: 'random', label: 'Random pair split', color: 'var(--series-1)' },
  { key: 'cold', label: 'Cold-start (unseen drugs)', color: 'var(--series-2)' },
]
const NODE_LABEL: Record<string, string> = {
  drug: 'Drugs',
  protein: 'Proteins',
  effect: 'Side effects',
  disease: 'Diseases',
  atc: 'ATC classes',
}

function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: { v: T; l: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="seg" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o.v} role="radio" aria-checked={value === o.v} onClick={() => onChange(o.v)}>
            {o.l}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [eda, setEda] = useState<Eda | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [labels, setLabels] = useState<'merged' | 'tdc'>('merged')
  const [metric, setMetric] = useState<keyof SplitMetrics>('auprc')
  const [negs, setNegs] = useState<'test' | 'test_hard'>('test')

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(String(e.message)))
    api.eda().then(setEda).catch((e) => setError(String(e.message)))
  }, [])

  const models = useMemo(() => MODEL_ORDER.filter((m) => metrics?.[m]?.[labels]), [metrics, labels])

  return (
    <div className="stack">
      <div>
        <p className="muted" style={{ maxWidth: 760 }}>
          Model comparison on two split types, plus exploratory analysis of the knowledge graph. Unknown pairs are
          treated as unlabelled, not safe, so precision-type metrics are lower bounds.
        </p>
      </div>
      {error && <p className="error">{error}</p>}

      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <Seg label="Labels" value={labels} onChange={setLabels}
          options={[{ v: 'merged', l: 'Merged (TDC + PrimeKG + DDInter)' }, { v: 'tdc', l: 'TDC DrugBank only' }]} />
        <Seg label="Metric" value={metric} onChange={setMetric} options={METRICS.map((m) => ({ v: m.key, l: m.label }))} />
        <Seg label="Unlabelled pairs" value={negs} onChange={setNegs}
          options={[{ v: 'test', l: 'Uniform' }, { v: 'test_hard', l: 'Degree-matched (harder)' }]} />
      </div>

      {metrics && models.length > 0 ? (
        <ChartCard
          title={`${METRICS.find((m) => m.key === metric)?.label} on the test set`}
          sub="The random split overstates performance; the cold-start split shows how well the model handles drugs it never saw in training."
          table={
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Split</th>
                  {METRICS.map((m) => (
                    <th key={m.key} className="num">{m.label}</th>
                  ))}
                  <th className="num">Train s</th>
                </tr>
              </thead>
              <tbody>
                {models.flatMap((m) =>
                  SPLITS.filter((s) => metrics[m][labels][s.key]).map((s) => {
                    const r = metrics[m][labels][s.key]
                    return (
                      <tr key={m + s.key}>
                        <td>{MODEL_LABEL[m]}</td>
                        <td>{s.label}</td>
                        {METRICS.map((k) => (
                          <td key={k.key} className="num">{(r[negs][k.key] as number).toFixed(3)}</td>
                        ))}
                        <td className="num">{r.train_seconds ?? '–'}</td>
                      </tr>
                    )
                  }),
                )}
              </tbody>
            </table>
          }
        >
          <GroupedColumns
            groups={models.map((m) => MODEL_LABEL[m])}
            series={SPLITS.map((s) => ({ name: s.label, color: s.color }))}
            values={models.map((m) => SPLITS.map((s) => (metrics[m][labels][s.key]?.[negs]?.[metric] as number) ?? null))}
            max={metric.startsWith('hits') ? undefined : 1}
            ariaLabel="Model comparison, random versus cold-start split"
          />
        </ChartCard>
      ) : (
        <div className="card muted">No results yet for these labels. Run python -m ddi.run_all.</div>
      )}

      {metrics && models.some((m) => m.startsWith('gnn')) && (
        <div className="card">
          <h2>GNN extra heads</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Split</th>
                  <th className="num">Type accuracy</th>
                  <th className="num">Type macro-F1</th>
                  <th className="num">Side effects: mean per-label AUPRC</th>
                  <th className="num">mean AUROC</th>
                  <th className="num">labels</th>
                </tr>
              </thead>
              <tbody>
                {models
                  .filter((m) => m.startsWith('gnn'))
                  .flatMap((m) =>
                    SPLITS.filter((s) => metrics[m][labels][s.key]).map((s) => {
                      const r = metrics[m][labels][s.key]
                      return (
                        <tr key={m + s.key}>
                          <td>{MODEL_LABEL[m]}</td>
                          <td>{s.label}</td>
                          <td className="num">{r.type?.accuracy.toFixed(3) ?? '–'}</td>
                          <td className="num">{r.type?.macro_f1.toFixed(3) ?? '–'}</td>
                          <td className="num">{r.side_effects?.mean_auprc.toFixed(3) ?? '–'}</td>
                          <td className="num">{r.side_effects?.mean_auroc.toFixed(3) ?? '–'}</td>
                          <td className="num">{r.side_effects?.n_labels ?? '–'}</td>
                        </tr>
                      )
                    }),
                  )}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>
            Type = which of 86 DrugBank interaction types (known pairs only). Side effects = TWOSIDES polypharmacy
            effects, averaged per label because each effect is rare (median positive rate ~5%).
          </p>
        </div>
      )}

      {eda && (
        <>
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Knowledge graph</div>
            <div className="kpis">
              <div className="kpi accent">
                <div className="kpi-label">Known interacting pairs</div>
                <div className="kpi-value">{eda.ddi.positives.merged.toLocaleString()}</div>
                <div className="kpi-foot">{(eda.ddi.density.merged * 100).toFixed(1)}% of all possible pairs</div>
              </div>
              {Object.entries(eda.graph.nodes).map(([k, v]) => (
                <div className="kpi" key={k}>
                  <div className="kpi-label">{NODE_LABEL[k] ?? k}</div>
                  <div className="kpi-value">{v.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid-2-even">
            <ChartCard
              title="Interaction degree distribution"
              sub="Drugs by number of known interaction partners (log-width bins). A long tail of hubs."
              table={<table><tbody>{eda.ddi.degree_hist.map((b) => <tr key={b.lo}><td>{b.lo}–{b.hi}</td><td className="num">{b.count}</td></tr>)}</tbody></table>}
            >
              <Columns
                labels={eda.ddi.degree_hist.map((b) => String(b.lo))}
                values={eda.ddi.degree_hist.map((b) => b.count)}
                tipTitle={(i) => `${eda.ddi.degree_hist[i].lo}–${eda.ddi.degree_hist[i].hi} partners`}
                valueLabel="drugs"
                xTitle="Known interaction partners (bin start)"
                ariaLabel="Histogram of drug interaction degree"
              />
            </ChartCard>
            <ChartCard
              title="Most connected drugs"
              sub="Known interaction partners, merged labels"
              table={<table><tbody>{eda.ddi.top_drugs.map((d) => <tr key={d.id}><td>{d.name}</td><td className="num">{d.degree}</td></tr>)}</tbody></table>}
            >
              <HBars items={eda.ddi.top_drugs.slice(0, 12).map((d) => ({ label: d.name, value: d.degree }))} valueLabel="partners" ariaLabel="Most connected drugs" labelWidth={150} />
            </ChartCard>
          </div>

          <ChartCard
            title="Most common DrugBank interaction types"
            sub="TDC benchmark pairs per type (86 types; heavily imbalanced)"
            table={<table><tbody>{eda.ddi.type_counts.map((t) => <tr key={t.type_id}><td>{t.template}</td><td className="num">{t.count}</td></tr>)}</tbody></table>}
          >
            <HBars
              items={eda.ddi.type_counts.slice(0, 12).map((t) => ({
                label: t.template.replace(/#Drug1|#Drug2/g, 'X').replace('The risk or severity of ', 'Risk of '),
                value: t.count,
                note: t.template,
              }))}
              valueLabel="pairs"
              ariaLabel="Most common interaction types"
              labelWidth={420}
            />
          </ChartCard>

          <div className="grid-2-even">
            <ChartCard
              title="Most frequent polypharmacy side effects"
              sub={`TWOSIDES pairs per effect (${eda.twosides.side_effects} effects kept, at least 500 pairs each)`}
              table={<table><tbody>{eda.twosides.top_side_effects.map((s) => <tr key={s.name}><td>{s.name}</td><td className="num">{s.pairs}</td></tr>)}</tbody></table>}
            >
              <HBars items={eda.twosides.top_side_effects.slice(0, 12).map((s) => ({ label: s.name, value: s.pairs }))} valueLabel="drug pairs" ariaLabel="Most frequent TWOSIDES side effects" labelWidth={190} />
            </ChartCard>
            <div className="stack">
            <div className="card">
              <h2>Cold-start exposure</h2>
              <table>
                <tbody>
                  <tr><td>Drugs with no target, enzyme or transporter link</td><td className="num">{eda.cold_start.drugs_without_protein_links}</td></tr>
                  <tr><td>Drugs with no SIDER side effect</td><td className="num">{eda.cold_start.drugs_without_side_effects}</td></tr>
                  <tr><td>Drugs with no KG edge at all</td><td className="num">{eda.cold_start.drugs_with_no_kg_edges}</td></tr>
                  <tr><td>Drugs not found in PrimeKG</td><td className="num">{eda.cold_start.drugs_not_in_primekg}</td></tr>
                </tbody>
              </table>
              <p className="small muted" style={{ marginTop: 8 }}>
                Drugs with few KG edges rely on their molecular fingerprint, which is why the GNN starts drug nodes from
                Morgan fingerprints rather than learned IDs.
              </p>
            </div>
            <div className="card">
              <h2>Where the interaction labels come from</h2>
              <table>
                <tbody>
                  {Object.entries(eda.ddi.source_overlap).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td className="num">{v.toLocaleString()}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="small muted" style={{ marginTop: 8 }}>
                DDInter severity levels: {Object.entries(eda.ddi.ddinter_levels).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ')}
              </p>
            </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
