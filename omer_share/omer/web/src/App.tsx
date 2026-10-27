import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { AuthProvider, useAuth } from './auth'
import { DISCLAIMER } from './components/Disclaimer'
import AboutPage from './pages/AboutPage'
import AnalyticsPage from './pages/AnalyticsPage'
import AuditPage from './pages/AuditPage'
import FaersPage from './pages/FaersPage'
import GraphPage from './pages/GraphPage'
import InputPage from './pages/InputPage'
import PairPage from './pages/PairPage'
import ResultsPage from './pages/ResultsPage'

type Theme = 'dark' | 'light'

const icon = (d: ReactNode) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    {d}
  </svg>
)

const NAV = [
  { to: '/', label: 'Check', end: true, title: 'Check a medication list', sub: 'Pairwise interaction screening',
    icon: icon(<><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h8M8 16h5" /></>) },
  { to: '/results', label: 'Results', title: 'Results', sub: 'Pairs ranked by severity',
    icon: icon(<><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></>) },
  { to: '/graph', label: 'Graph explorer', title: 'Graph explorer', sub: 'Knowledge-graph neighbourhoods',
    icon: icon(<><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="7" r="2.5" /><circle cx="12" cy="18" r="2.5" /><path d="M8.2 7.2 15.6 7M7.3 8.2l3.6 7.6M17 9.3l-3.8 6.6" /></>) },
  { to: '/analytics', label: 'Analytics', title: 'Analytics', sub: 'Model comparison and EDA',
    icon: icon(<><path d="M3 3v18h18" /><path d="m7 15 4-5 3 3 5-7" /></>) },
  { to: '/faers', label: 'FAERS reports', title: 'FAERS reports', sub: 'Adverse-event reports from openFDA',
    icon: icon(<><path d="M12 3 2 20h20L12 3Z" /><path d="M12 10v4M12 17v.5" /></>) },
  { to: '/audit', label: 'Audit log', title: 'Audit log', sub: 'Administrators only',
    icon: icon(<><rect x="5" y="10" width="14" height="10" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></>) },
  { to: '/about', label: 'Model card', title: 'Model card', sub: 'Intended use, results, limitations',
    icon: icon(<><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.5" /></>) },
]

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem('theme') === 'light' ? 'light' : 'dark'
    } catch {
      return 'dark'
    }
  })
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('theme', theme)
    } catch {
      /* ignore */
    }
    window.dispatchEvent(new Event('themechange'))
  }, [theme])
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button className="icon-btn" onClick={() => setTheme(next)} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      {theme === 'dark'
        ? icon(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>)
        : icon(<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />)}
    </button>
  )
}

function UserMenu() {
  const { user, setUser } = useAuth()
  if (user.role === 'guest')
    return (
      <Link to="/audit" className="small user-chip">
        Sign in
      </Link>
    )
  return (
    <span className="small user-chip">
      {user.username} <span className="tag">{user.role}</span>
      <button
        className="link small"
        onClick={async () => {
          await api.logout().catch(() => undefined)
          setUser({ username: null, role: 'guest' })
        }}
      >
        Sign out
      </button>
    </span>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  )
}

function Shell() {
  const { pathname } = useLocation()
  const page = pathname.startsWith('/pair')
    ? { title: 'Pair details', sub: 'Mechanism, evidence and predicted effects' }
    : NAV.find((n) => (n.end ? pathname === n.to : pathname.startsWith(n.to))) ?? NAV[0]

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Main">
        <Link to="/" className="brand">
          <div className="brand-mark"><span /></div>
          <div className="brand-text">
            <div className="brand-title">Interaction Check</div>
            <div className="brand-sub">DDI predictor</div>
          </div>
        </Link>
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end} className="nav-item">
            {n.icon}
            <span>{n.label}</span>
          </NavLink>
        ))}
        <div className="sidebar-foot">
          <p>Decision support only. Not a medical device. A pair with nothing flagged is not proven safe.</p>
          <p>Data: DrugBank (via TDC, PrimeKG), DDInter, TWOSIDES, SIDER.</p>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <h1 className="topbar-title">
            {page.title}
            <small>{page.sub}</small>
          </h1>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <UserMenu />
            <ThemeToggle />
          </div>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<InputPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/pair/:a/:b" element={<PairPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/faers" element={<FaersPage />} />
            <Route path="/audit" element={<AuditPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <footer className="footer-disclaimer">{DISCLAIMER}</footer>
      </div>
    </div>
  )
}
