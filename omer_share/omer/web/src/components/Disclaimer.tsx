export const DISCLAIMER =
  'Decision support only. This tool is a research prototype, not a medical device. Predictions come from ' +
  'statistical models trained on public databases and can be wrong or incomplete. A pair with no flagged ' +
  'interaction has not been shown to be safe; it may simply be untested. Always confirm with a pharmacist, ' +
  'prescriber, or an authoritative reference before changing any medication.'

export function Disclaimer({ text = DISCLAIMER }: { text?: string }) {
  return (
    <div className="disclaimer" role="note">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
        <path d="M12 3 2 20h20L12 3Z" />
        <path d="M12 10v4M12 17v.5" strokeLinecap="round" />
      </svg>
      <div>
        <strong>Medical disclaimer. </strong>
        {text}
      </div>
    </div>
  )
}
