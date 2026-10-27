import { useEffect, useState, type FormEvent } from 'react'
import { api, type AuditLog } from '../api'
import { useAuth } from '../auth'
import { SevIcon } from '../components/Severity'

export function LoginForm({ note }: { note?: string }) {
  const { setUser } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      setUser(await api.login(username, password))
      setPassword('')
    } catch (err) {
      setError(String((err as Error).message))
    } finally {
      setBusy(false)
    }
  }
  return (
    <form className="card login" onSubmit={submit}>
      <h2>Sign in</h2>
      {note && <p className="small muted">{note}</p>}
      <div className="stack">
        <div className="field">
          <label htmlFor="user">Username</label>
          <input id="user" type="text" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="pw">Password</label>
          <input id="pw" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </div>
        <button className="primary" type="submit" disabled={!username || !password || busy}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
        {error && <p className="error small">{error}</p>}
      </div>
    </form>
  )
}

const LEVELS = ['', 'HIGH', 'MODERATE', 'LOW']
const LEVEL_SEV = { HIGH: 'major', MODERATE: 'moderate', LOW: 'none' } as const

export default function AuditPage() {
  const { user } = useAuth()
  const [log, setLog] = useState<AuditLog | null>(null)
  const [level, setLevel] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (user.role !== 'admin') return
    setError(null)
    api.audit(level || undefined).then(setLog).catch((e) => setError(String(e.message)))
  }, [user, level])

  if (user.role !== 'admin')
    return (
      <div className="stack">
        <LoginForm
          note={
            user.role === 'clinician'
              ? `Signed in as ${user.username} (clinician). The audit log needs an administrator account.`
              : 'The audit log is restricted to administrators.'
          }
        />
      </div>
    )

  return (
    <div className="stack">
      <p className="muted" style={{ maxWidth: 820 }}>
        Every analysis is logged with its time, drug list and result summary. No patient details, usernames or IP
        addresses are stored: only the role of whoever ran it and which kinds of patient context were supplied.
      </p>
      <div className="kpis">
        <div className="kpi accent">
          <div className="kpi-label">Analyses logged</div>
          <div className="kpi-value num">{log ? Object.values(log.levels).reduce((s, n) => s + n, 0) : '–'}</div>
          <div className="kpi-foot">stored in data/audit.sqlite</div>
        </div>
        {(['HIGH', 'MODERATE', 'LOW'] as const).map((l) => (
          <div className="kpi" key={l}>
            <div className="kpi-label">{l} regimen risk</div>
            <div className="kpi-value num">{log?.levels[l] ?? 0}</div>
          </div>
        ))}
      </div>
      <div className="card">
        <div className="card-head">
          <h2>Audit log</h2>
          <div className="seg" role="radiogroup" aria-label="Filter by regimen risk">
            {LEVELS.map((l) => (
              <button key={l} role="radio" aria-checked={level === l} onClick={() => setLevel(l)}>
                {l || 'All'}
              </button>
            ))}
          </div>
        </div>
        {error && <p className="error">{error}</p>}
        {log && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time (UTC)</th>
                  <th>Role</th>
                  <th>Medications</th>
                  <th>Risk</th>
                  <th className="num">Pairs / flagged</th>
                  <th>Top flagged pairs</th>
                  <th>Context used</th>
                </tr>
              </thead>
              <tbody>
                {log.entries.length === 0 && (
                  <tr>
                    <td colSpan={7} className="muted">No analyses logged yet.</td>
                  </tr>
                )}
                {log.entries.map((e) => (
                  <tr key={e.id}>
                    <td className="num" style={{ whiteSpace: 'nowrap' }}>{e.ts.replace('T', ' ').slice(0, 19)}</td>
                    <td>{e.role}</td>
                    <td>{e.drugs.map((d) => d.name).join(', ')}</td>
                    <td>
                      <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontWeight: 600 }}>
                        <SevIcon severity={LEVEL_SEV[e.level]} /> {e.level}
                      </span>
                      <div className="small muted num">score {e.score.toFixed(2)}</div>
                    </td>
                    <td className="num">
                      {e.pairs_checked} / {e.flagged}
                    </td>
                    <td className="small">
                      {e.top_pairs.slice(0, 3).map((p) => `${p.a} + ${p.b} (${p.severity})`).join('; ') || '–'}
                    </td>
                    <td className="small">{e.context_used.join(', ') || '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {log.total > log.entries.length && (
              <p className="small muted">Showing the latest {log.entries.length} of {log.total}.</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
