import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api'
import { Disclaimer } from '../components/Disclaimer'

/** Minimal Markdown: headings, bullet/numbered lists, tables, paragraphs, **bold**, `code`. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
    part.startsWith('**') ? <strong key={i}>{part.slice(2, -2)}</strong> : part.startsWith('`') ? <code key={i}>{part.slice(1, -1)}</code> : part,
  )
}

function Markdown({ src }: { src: string }) {
  const out: ReactNode[] = []
  const lines = src.split(/\r?\n/)
  let i = 0
  while (i < lines.length) {
    const l = lines[i]
    if (!l.trim()) {
      i++
      continue
    }
    const h = /^(#{1,3})\s+(.*)/.exec(l)
    if (h) {
      const level = h[1].length
      const content = inline(h[2])
      out.push(level === 1 ? <h1 key={i}>{content}</h1> : level === 2 ? <h2 key={i}>{content}</h2> : <h3 key={i}>{content}</h3>)
      i++
      continue
    }
    if (l.startsWith('|')) {
      const rows: string[][] = []
      while (i < lines.length && lines[i].startsWith('|')) {
        if (!/^\|\s*-/.test(lines[i])) rows.push(lines[i].split('|').slice(1, -1).map((c) => c.trim()))
        i++
      }
      out.push(
        <div className="table-wrap" key={`t${i}`}>
          <table>
            <thead>
              <tr>{rows[0].map((c, j) => <th key={j}>{inline(c)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.slice(1).map((r, k) => (
                <tr key={k}>{r.map((c, j) => <td key={j} className={/^[\d.]+$/.test(c) ? 'num' : undefined}>{inline(c)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }
    if (/^(\s*[-*]|\s*\d+\.)\s/.test(l)) {
      const ordered = /^\s*\d+\./.test(l)
      const items: string[] = []
      while (i < lines.length && /^(\s*[-*]|\s*\d+\.)\s/.test(lines[i])) {
        items.push(lines[i].replace(/^(\s*[-*]|\s*\d+\.)\s/, ''))
        i++
      }
      const lis = items.map((t, k) => <li key={k}>{inline(t)}</li>)
      out.push(ordered ? <ol key={`l${i}`}>{lis}</ol> : <ul key={`l${i}`}>{lis}</ul>)
      continue
    }
    const para: string[] = []
    while (i < lines.length && lines[i].trim() && !/^(#|\||\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) {
      para.push(lines[i])
      i++
    }
    out.push(<p key={`p${i}`}>{inline(para.join(' '))}</p>)
  }
  return <div className="prose">{out}</div>
}

export default function AboutPage() {
  const [card, setCard] = useState<string | null>(null)
  const [disclaimer, setDisclaimer] = useState<string | undefined>()
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api
      .modelCard()
      .then((r) => {
        setCard(r.card)
        setDisclaimer(r.disclaimer)
      })
      .catch((e) => setError(String(e.message)))
  }, [])
  return (
    <div className="stack">
      <Disclaimer text={disclaimer} />
      <div className="card">
        {error && <p className="error">{error}</p>}
        {card === null ? <p className="muted">Loading model card…</p> : <Markdown src={card} />}
      </div>
    </div>
  )
}
