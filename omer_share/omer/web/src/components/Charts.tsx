import { useEffect, useId, useRef, useState, type ReactNode } from 'react'

/* Shared tooltip: values lead, labels follow. Text is rendered as React text nodes (never innerHTML). */
interface Tip {
  x: number
  y: number
  title: string
  rows: { key?: string; label: string; value: string }[]
}
function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return (
    <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
      <div className="muted small">{tip.title}</div>
      {tip.rows.map((r, i) => (
        <div key={i}>
          {r.key && <span className="key" style={{ background: r.key }} />}
          <strong className="num">{r.value}</strong> <span className="muted">{r.label}</span>
        </div>
      ))}
    </div>
  )
}

function useTip() {
  const [tip, setTip] = useState<Tip | null>(null)
  const show = (e: { clientX: number; clientY: number }, t: Omit<Tip, 'x' | 'y'>) => setTip({ ...t, x: e.clientX, y: e.clientY })
  const showAt = (el: Element, t: Omit<Tip, 'x' | 'y'>) => {
    const r = el.getBoundingClientRect()
    setTip({ ...t, x: r.left + r.width / 2, y: r.top })
  }
  return { tip, show, showAt, hide: () => setTip(null) }
}


/** Width of the chart container, so the SVG is drawn 1:1 and text stays at its CSS size. */
function useWidth(fallback = 640) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(fallback)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Rounded-top bar path: 4px radius at the data end, square at the baseline. */
function colPath(x: number, y: number, w: number, h: number, r = 4) {
  if (h <= 0) return ''
  const rr = Math.min(r, w / 2, h)
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`
}
function rowPath(x: number, y: number, w: number, h: number, r = 4) {
  if (w <= 0) return ''
  const rr = Math.min(r, h / 2, w)
  return `M${x},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h - rr}Q${x + w},${y + h} ${x + w - rr},${y + h}H${x}Z`
}

function niceMax(v: number) {
  if (v <= 0) return 1
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  const n = v / p
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p
}
export function compact(v: number) {
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}K`
  return Number.isInteger(v) ? String(v) : v.toFixed(2)
}

/* ---------------------------------------------------------------- grouped columns */
export interface Series {
  name: string
  color: string
}
interface GroupedProps {
  groups: string[]
  series: Series[]
  values: (number | null)[][] // [group][series]
  max?: number
  format?: (v: number) => string
  height?: number
  ariaLabel: string
}
export function GroupedColumns({ groups, series, values, max, format = (v) => v.toFixed(3), height = 260, ariaLabel }: GroupedProps) {
  const { tip, show, showAt, hide } = useTip()
  const [box, W] = useWidth()
  const hatchId = useId().replace(/:/g, '')
  // The second series is achromatic (white/black), so it also carries a 45° hatch as a secondary encoding.
  const fillFor = (si: number) => (si === 0 ? series[0].color : `url(#${hatchId}-${si})`)
  const m = { t: 22, r: 8, b: 44, l: 40 }
  const ih = height - m.t - m.b
  const iw = W - m.l - m.r
  const top = max ?? niceMax(Math.max(...values.flat().map((v) => v ?? 0)))
  const band = iw / groups.length
  const barW = Math.min(24, (band * 0.6) / series.length)
  const gap = 2
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top)
  const y = (v: number) => m.t + ih - (v / top) * ih
  return (
    <div ref={box}>
      <div className="legend" style={{ marginTop: 0, marginBottom: 6 }}>
        {series.map((s) => (
          <span key={s.name}>
            <span className="swatch" style={{ background: s === series[0] ? s.color
              : `repeating-linear-gradient(45deg, ${s.color} 0 2px, var(--surface) 2px 3.5px)` }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel} onMouseLeave={hide}>
        <defs>
          {series.slice(1).map((s, k) => (
            <pattern key={s.name} id={`${hatchId}-${k + 1}`} width="6" height="6" patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)">
              <rect width="6" height="6" style={{ fill: s.color }} />
              <line x1="0" y1="0" x2="0" y2="6" style={{ stroke: 'var(--surface)', strokeWidth: 2 }} />
            </pattern>
          ))}
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === 0 ? 'axis' : 'grid'} x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="num">
              {top <= 1 ? t.toFixed(2) : compact(t)}
            </text>
          </g>
        ))}
        {groups.map((g, gi) => {
          const total = series.length * barW + (series.length - 1) * gap
          const x0 = m.l + gi * band + (band - total) / 2
          const groupTip = {
            title: g,
            rows: series
              .map((s, si) => ({ key: s.color, label: s.name, value: values[gi][si] == null ? '–' : format(values[gi][si]!) })),
          }
          return (
            <g key={g}>
              {/* the whole band is the hit target; one tooltip lists every series */}
              <rect x={m.l + gi * band} y={m.t} width={band} height={ih} fill="transparent"
                onMouseMove={(e) => show(e, groupTip)} />
              {series.map((s, si) => {
                const v = values[gi][si]
                if (v == null) return null
                const x = x0 + si * (barW + gap)
                const t = groupTip
                return (
                  <g key={s.name}>
                    <path
                      className="mark"
                      d={colPath(x, y(v), barW, y(0) - y(v))}
                      fill={fillFor(si)}
                      tabIndex={0}
                      aria-label={`${g}, ${s.name}: ${format(v)}`}
                      onMouseMove={(e) => show(e, t)}
                      onFocus={(e) => showAt(e.currentTarget, t)}
                      onBlur={hide}
                    />
                    <text x={x + barW / 2} y={y(v) - 5} textAnchor="middle" className="num" style={{ fontSize: 10 }}>
                      {format(v)}
                    </text>
                  </g>
                )
              })}
              <text x={m.l + gi * band + band / 2} y={height - m.b + 16} textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {g}
              </text>
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ---------------------------------------------------------------- single-series columns (histogram) */
interface ColumnsProps {
  labels: string[]
  values: number[]
  color?: string
  height?: number
  xTitle?: string
  ariaLabel: string
  tipTitle?: (i: number) => string
  valueLabel: string
}
export function Columns({ labels, values, color = 'var(--bar)', height = 220, xTitle, ariaLabel, tipTitle, valueLabel }: ColumnsProps) {
  const { tip, show, showAt, hide } = useTip()
  const [box, W] = useWidth()
  const m = { t: 12, r: 8, b: xTitle ? 48 : 32, l: 44 }
  const ih = height - m.t - m.b
  const iw = W - m.l - m.r
  const top = niceMax(Math.max(...values, 1))
  const band = iw / labels.length
  const bw = Math.min(24, band - 2)
  const y = (v: number) => m.t + ih - (v / top) * ih
  const ticks = [0, 0.5, 1].map((f) => f * top)
  const every = Math.ceil(labels.length / Math.max(3, Math.floor(iw / 60)))
  return (
    <div ref={box}>
      <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel} onMouseLeave={hide}>
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === 0 ? 'axis' : 'grid'} x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="num">
              {compact(t)}
            </text>
          </g>
        ))}
        {values.map((v, i) => {
          const x = m.l + i * band + (band - bw) / 2
          const t = { title: tipTitle ? tipTitle(i) : labels[i], rows: [{ label: valueLabel, value: v.toLocaleString() }] }
          return (
            <g key={i}>
              <rect x={m.l + i * band} y={m.t} width={band} height={ih} fill="transparent"
                onMouseMove={(e) => show(e, t)} />
              <path className="mark" d={colPath(x, y(v), bw, y(0) - y(v))} fill={color} tabIndex={0}
                aria-label={`${t.title}: ${v}`} onMouseMove={(e) => show(e, t)}
                onFocus={(e) => showAt(e.currentTarget, t)} onBlur={hide} />
              {i % every === 0 && (
                <text x={m.l + i * band + band / 2} y={m.t + ih + 14} textAnchor="middle" className="num">
                  {labels[i]}
                </text>
              )}
            </g>
          )
        })}
        {xTitle && (
          <text x={m.l + iw / 2} y={height - 8} textAnchor="middle">
            {xTitle}
          </text>
        )}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/* ---------------------------------------------------------------- horizontal bars */
interface BarsProps {
  items: { label: string; value: number; note?: string }[]
  color?: string
  valueLabel: string
  ariaLabel: string
  labelWidth?: number
}
export function HBars({ items, color = 'var(--bar)', valueLabel, ariaLabel, labelWidth = 210 }: BarsProps) {
  const { tip, show, showAt, hide } = useTip()
  const [box, W] = useWidth()
  const row = 22
  const lw = Math.min(labelWidth, Math.round(W * 0.45))
  const m = { t: 4, r: 44, l: lw }
  const maxChars = Math.floor(lw / 6.2)
  const H = m.t + items.length * row + 4
  const iw = W - m.l - m.r
  const top = Math.max(...items.map((i) => i.value), 1)
  return (
    <div ref={box}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} onMouseLeave={hide}>
        <line className="axis" x1={m.l} x2={m.l} y1={m.t} y2={H - 4} />
        {items.map((it, i) => {
          const y = m.t + i * row + 4
          const w = (it.value / top) * iw
          const t = { title: it.note ?? it.label, rows: [{ label: valueLabel, value: it.value.toLocaleString() }] }
          return (
            <g key={it.label + i} onMouseMove={(e) => show(e, t)}>
              <rect x={0} y={y - 4} width={W} height={row} fill="transparent" />
              <text x={m.l - 8} y={y + 11} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                {it.label.length > maxChars ? it.label.slice(0, maxChars - 1) + '…' : it.label}
              </text>
              <path className="mark" d={rowPath(m.l, y, w, 14)} fill={color} tabIndex={0}
                aria-label={`${it.label}: ${it.value}`} onFocus={(e) => showAt(e.currentTarget, t)} onBlur={hide} />
              <text x={m.l + w + 6} y={y + 11} className="num">
                {compact(it.value)}
              </text>
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

export function ChartCard({ title, sub, children, table }: { title: string; sub?: string; children: ReactNode; table?: ReactNode }) {
  const [showTable, setShowTable] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  return (
    <div className="card" ref={ref}>
      <div className="card-head">
        <div>
          <h2 style={{ marginBottom: 2 }}>{title}</h2>
          {sub && <p className="muted small" style={{ margin: 0 }}>{sub}</p>}
        </div>
        {table && (
          <button className="link small" onClick={() => setShowTable((s) => !s)} aria-expanded={showTable}>
            {showTable ? 'Show chart' : 'Show table'}
          </button>
        )}
      </div>
      {showTable && table ? <div className="table-wrap">{table}</div> : children}
    </div>
  )
}
