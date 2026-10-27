import type { Severity } from '../api'

const LABEL: Record<Severity, string> = {
  major: 'Major',
  moderate: 'Moderate',
  minor: 'Minor',
  none: 'None found',
}

export const SEV_VAR: Record<Severity, string> = {
  major: 'var(--sev-major)',
  moderate: 'var(--sev-moderate)',
  minor: 'var(--sev-minor)',
  none: 'var(--sev-none)',
}

/** Icon + label + colour: severity is never carried by colour alone. */
export function SevIcon({ severity, size = 14 }: { severity: Severity; size?: number }) {
  const c = SEV_VAR[severity]
  if (severity === 'major')
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
        <path d="M8 1.5 15 14H1L8 1.5Z" fill={c} />
        <rect x="7.2" y="5.5" width="1.6" height="4.6" rx="0.8" fill="#fff" />
        <circle cx="8" cy="12" r="0.95" fill="#fff" />
      </svg>
    )
  if (severity === 'moderate')
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
        <path d="M8 1 15 8 8 15 1 8Z" fill={c} />
        <rect x="7.2" y="4.6" width="1.6" height="4.6" rx="0.8" fill="#fff" />
        <circle cx="8" cy="11.2" r="0.95" fill="#fff" />
      </svg>
    )
  if (severity === 'minor')
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
        <circle cx="8" cy="8" r="7" fill={c} />
        <rect x="4.5" y="7.2" width="7" height="1.6" rx="0.8" fill="#0b0b0b" />
      </svg>
    )
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="8" r="7" fill={c} />
      <path d="M4.8 8.2 7 10.4l4.2-4.6" stroke="#fff" strokeWidth="1.7" fill="none" strokeLinecap="round" />
    </svg>
  )
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <span className={`sev sev-${severity}`}>
      <SevIcon severity={severity} />
      {LABEL[severity]}
    </span>
  )
}

export const severityLabel = (s: Severity) => LABEL[s]
