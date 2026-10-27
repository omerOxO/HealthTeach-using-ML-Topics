import type { ChainStep } from '../api'

const TYPE_VAR: Record<string, string> = {
  drug: 'var(--node-drug)',
  protein: 'var(--node-protein)',
  effect: 'var(--node-effect)',
  disease: 'var(--node-disease)',
  atc: 'var(--node-atc)',
}
const REL: Record<string, string> = {
  enzyme: 'metabolised by',
  target: 'acts on',
  transporter: 'transported by',
  carrier: 'carried by',
  side_effect: 'linked to',
  in_class: 'in class',
  indication: 'treats',
  contraindication: 'contraindicated in',
  off_label: 'off-label for',
  ppi: 'protein interaction',
}

export function PathChain({ steps }: { steps: ChainStep[] }) {
  if (!steps.length) return <span className="muted small">No connecting path in the knowledge graph.</span>
  return (
    <span style={{ display: 'inline-flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
      {steps.map((s, i) =>
        'arrow' in s ? (
          <span key={i} className="muted" title={REL[s.rel] ?? s.rel} aria-label={REL[s.rel] ?? s.rel}>
            {s.arrow}
          </span>
        ) : (
          <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontWeight: s.type === 'drug' ? 600 : 400 }}>
            <i style={{ width: 8, height: 8, borderRadius: '50%', background: TYPE_VAR[s.type] ?? 'var(--muted)', display: 'inline-block' }} />
            {s.label}
          </span>
        ),
      )}
    </span>
  )
}
