import { useEffect, useId, useRef, useState } from 'react'
import { api, type DrugCard } from '../api'

interface Props {
  onPick: (d: DrugCard) => void
  exclude?: string[]
  placeholder?: string
  autoFocus?: boolean
}

export function DrugSearch({ onPick, exclude = [], placeholder = 'Search a medication, e.g. warfarin', autoFocus }: Props) {
  const [q, setQ] = useState('')
  const [items, setItems] = useState<DrugCard[]>([])
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [itemsFor, setItemsFor] = useState('')
  const listId = useId()
  const reqId = useRef(0)

  // a joined key, so a fresh `exclude` array each render (or the [] default) does not re-run the effect forever
  const excludeKey = exclude.join(',')
  useEffect(() => {
    const excluded = excludeKey ? excludeKey.split(',') : []
    const text = q.trim()
    if (!text) {
      setItems([])
      setItemsFor('')
      return
    }
    const id = ++reqId.current
    const t = setTimeout(() => {
      api
        .search(text)
        .then((r) => {
          if (id !== reqId.current) return
          const list = r.filter((d) => !excluded.includes(d.id))
          setItems(list)
          setItemsFor(text)
          setActive(0)
          setOpen(true)
          setError(null)
        })
        .catch((e) => id === reqId.current && setError(String(e.message ?? e)))
    }, 120)
    return () => clearTimeout(t)
  }, [q, excludeKey])

  const onPickRef = useRef(onPick)
  onPickRef.current = onPick

  function pick(d: DrugCard) {
    onPickRef.current(d)
    setQ('')
    setItems([])
    setItemsFor('')
    setOpen(false)
  }

  return (
    <div className="search">
      <input
        type="text"
        role="combobox"
        aria-expanded={open && items.length > 0}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-label="Medication search"
        placeholder={placeholder}
        value={q}
        autoFocus={autoFocus}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => items.length && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault()
            setActive((a) => Math.min(a + 1, items.length - 1))
          } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setActive((a) => Math.max(a - 1, 0))
          } else if (e.key === 'Enter') {
            e.preventDefault()
            const text = q.trim()
            if (itemsFor === text && items[active]) pick(items[active])
            else if (text) {
              // results for this text are not in yet: resolve it directly and free the box for the next drug
              reqId.current++
              setQ('')
              setItems([])
              setOpen(false)
              api
                .search(text)
                .then((r) => {
                  const hit = r.find((d) => !exclude.includes(d.id))
                  if (hit) onPickRef.current(hit)
                  else setError(`No medication matches "${text}"`)
                })
                .catch((err) => setError(String(err.message ?? err)))
            }
          } else if (e.key === 'Escape') setOpen(false)
        }}
      />
      {error && <p className="error small">Search failed: {error}</p>}
      {open && items.length > 0 && (
        <ul className="suggest" id={listId} role="listbox">
          {items.map((d, i) => (
            <li
              key={d.id}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => {
                e.preventDefault()
                pick(d)
              }}
              onMouseEnter={() => setActive(i)}
            >
              <span>
                {d.name}
                {d.matched && <span className="muted small"> ({d.matched})</span>}
              </span>
              <span className="muted small">{d.class ?? d.id}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
