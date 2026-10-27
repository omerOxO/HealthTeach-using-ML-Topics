import cytoscape from 'cytoscape'
import { useEffect, useRef, useState } from 'react'
import type { GraphEdge, GraphNode } from '../api'

const TYPE_VAR: Record<string, string> = {
  drug: '--node-drug',
  protein: '--node-protein',
  effect: '--node-effect',
  disease: '--node-disease',
  atc: '--node-atc',
}
export const TYPE_LABEL: Record<string, string> = {
  drug: 'Drug',
  protein: 'Protein (target / enzyme / transporter)',
  effect: 'Side effect',
  disease: 'Disease',
  atc: 'Drug class (ATC)',
}
const REL_LABEL: Record<string, string> = {
  enzyme: 'metabolised by',
  target: 'targets',
  transporter: 'transported by',
  carrier: 'carried by',
  side_effect: 'side effect',
  in_class: 'class',
  indication: 'treats',
  contraindication: 'contraindicated',
  off_label: 'off-label',
  ppi: 'interacts (protein)',
}

/** Two focus drugs left and right, nodes they share in the middle column, each drug's own
 * nodes (protein bridges, side-effect tails) in a column beside it. */
function pairPositions(nodes: GraphNode[], edges: GraphEdge[]) {
  const focus = nodes.filter((n) => n.focus).map((n) => n.id)
  const [a, b] = focus
  const nb = new Map<number, Set<number>>()
  edges.forEach((e) => {
    if (!nb.has(e.source)) nb.set(e.source, new Set())
    if (!nb.has(e.target)) nb.set(e.target, new Set())
    nb.get(e.source)!.add(e.target)
    nb.get(e.target)!.add(e.source)
  })
  const cols: Record<string, number[]> = { mid: [], left: [], right: [] }
  nodes.forEach((n) => {
    if (n.focus) return
    const s = nb.get(n.id) ?? new Set()
    if (s.has(a) && s.has(b)) cols.mid.push(n.id)
    else if (s.has(a)) cols.left.push(n.id)
    else if (s.has(b)) cols.right.push(n.id)
    else {
      // two hops away (second protein of a bridge): side of the drug it reaches through
      const viaA = [...s].some((m) => nb.get(m)?.has(a))
      ;(viaA ? cols.left : cols.right).push(n.id)
    }
  })
  const pos: Record<string, { x: number; y: number }> = {}
  const place = (ids: number[], x: number, y0 = 0) =>
    ids.forEach((id, i) => (pos[String(id)] = { x, y: y0 + (i - (ids.length - 1) / 2) * 56 }))
  pos[String(a)] = { x: 0, y: 0 }
  pos[String(b)] = { x: 360, y: 0 }
  place(cols.mid, 180)
  place(cols.left, 70, 90 + (cols.left.length - 1) * 28)
  place(cols.right, 290, 90 + (cols.right.length - 1) * 28)
  return pos
}

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'
}

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  highlight?: GraphEdge[]
  tall?: boolean
  layout?: 'cose' | 'concentric' | 'breadthfirst' | 'pair'
  onNodeClick?: (n: GraphNode) => void
  ariaLabel: string
}

export function KGGraph({ nodes, edges, highlight = [], tall, layout = 'cose', onNodeClick, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  // colours are read from CSS variables, so redraw when the theme changes
  const [themeTick, setThemeTick] = useState(0)
  useEffect(() => {
    const f = () => setThemeTick((t) => t + 1)
    window.addEventListener('themechange', f)
    return () => window.removeEventListener('themechange', f)
  }, [])

  useEffect(() => {
    if (!ref.current) return
    const hl = new Set(highlight.map((e) => `${e.source}-${e.target}-${e.rel}`))
    const hlNodes = new Set(highlight.flatMap((e) => [e.source, e.target]))
    const ink = cssVar('--ink')
    const surface = cssVar('--surface')
    const axis = cssVar('--axis')
    const accent = cssVar('--red')
    const cy = cytoscape({
      container: ref.current,
      elements: [
        ...nodes.map((n) => ({
          data: { id: String(n.id), label: n.label, type: n.type, focus: n.focus ? 1 : 0, hl: hlNodes.has(n.id) ? 1 : 0 },
        })),
        ...edges.map((e, i) => ({
          data: {
            id: `e${i}`,
            source: String(e.source),
            target: String(e.target),
            label: REL_LABEL[e.rel] ?? e.rel,
            hl: hl.has(`${e.source}-${e.target}-${e.rel}`) ? 1 : 0,
          },
        })),
        ...highlight
          .filter((e) => !edges.some((x) => x.source === e.source && x.target === e.target && x.rel === e.rel))
          .map((e, i) => ({
            data: { id: `h${i}`, source: String(e.source), target: String(e.target), label: REL_LABEL[e.rel] ?? e.rel, hl: 1 },
          })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (el: cytoscape.NodeSingular) => cssVar(TYPE_VAR[el.data('type')] ?? '--muted'),
            label: 'data(label)',
            color: ink,
            'font-size': 11,
            'text-valign': 'bottom',
            'text-margin-y': 4,
            'text-wrap': 'ellipsis',
            'text-max-width': '120px',
            width: 18,
            height: 18,
            'border-width': 2,
            'border-color': surface,
          },
        },
        { selector: 'node[focus = 1]', style: { width: 30, height: 30, 'font-size': 13, 'font-weight': 'bold' } },
        { selector: 'node[hl = 1]', style: { 'border-color': accent, 'border-width': 3 } },
        {
          selector: 'edge',
          style: {
            width: 1.5,
            'line-color': axis,
            'curve-style': 'bezier',
            label: edges.length > 20 ? '' : 'data(label)',
            'font-size': 9,
            color: cssVar('--muted'),
            'text-rotation': 'autorotate',
            'text-background-color': surface,
            'text-background-opacity': 1,
            'text-background-padding': '1px',
          },
        },
        { selector: 'edge[hl = 1]', style: { width: 3, 'line-color': accent } },
      ],
      layout:
        layout === 'pair'
          ? { name: 'preset', positions: pairPositions(nodes, edges), fit: true, padding: 40 }
          : layout === 'concentric'
          ? { name: 'concentric', concentric: (n: cytoscape.NodeSingular) => (n.data('focus') ? 10 : n.data('type') === 'drug' ? 0 : 1), levelWidth: () => 1, minNodeSpacing: 28 }
          : layout === 'breadthfirst'
            ? { name: 'breadthfirst', directed: false, spacingFactor: 1.1 }
            : { name: 'cose', animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 90, padding: 20 },
      wheelSensitivity: 0.2,
      minZoom: 0.3,
      maxZoom: 3,
    })
    if (onNodeClick) {
      cy.on('tap', 'node', (evt) => {
        const id = Number(evt.target.id())
        const n = nodes.find((x) => x.id === id)
        if (n) onNodeClick(n)
      })
    }
    // Fit again once the card has its final size and the web font has loaded (label boxes change),
    // and on later resizes.
    let alive = true
    const refit = () => {
      if (!alive) return
      cy.resize()
      cy.fit(undefined, 30)
    }
    const ro = new ResizeObserver(refit)
    ro.observe(ref.current)
    const raf = requestAnimationFrame(refit)
    const timer = setTimeout(refit, 250)
    document.fonts?.ready.then(refit)
    return () => {
      alive = false
      ro.disconnect()
      cancelAnimationFrame(raf)
      clearTimeout(timer)
      cy.destroy()
    }
  }, [nodes, edges, highlight, layout, onNodeClick, themeTick])

  const types = Array.from(new Set(nodes.map((n) => n.type)))
  return (
    <div>
      <div ref={ref} className={`cy${tall ? ' tall' : ''}`} role="img" aria-label={ariaLabel} />
      <div className="legend">
        {types.map((t) => (
          <span key={t}>
            <i style={{ background: `var(${TYPE_VAR[t] ?? '--muted'})` }} />
            {TYPE_LABEL[t] ?? t}
          </span>
        ))}
        {highlight.length > 0 && (
          <span>
            <span className="key" style={{ background: 'var(--red)' }} />
            explanation path
          </span>
        )}
      </div>
    </div>
  )
}
