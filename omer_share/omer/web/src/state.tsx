import { createContext, useContext, useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react'
import type { AnalyzeResult, DrugCard, PatientContext } from './api'

interface Regimen {
  drugs: DrugCard[]
  setDrugs: Dispatch<SetStateAction<DrugCard[]>>
  context: PatientContext
  setContext: (c: PatientContext) => void
  result: AnalyzeResult | null
  setResult: (r: AnalyzeResult | null) => void
}

const EMPTY_CONTEXT: PatientContext = { age: null, kidney: false, liver: false, conditions: [], allergies: [] }
const KEY = 'ddi-regimen-v1'

function load() {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

const Ctx = createContext<Regimen | null>(null)

export function RegimenProvider({ children }: { children: ReactNode }) {
  const saved = load()
  const [drugs, setDrugs] = useState<DrugCard[]>(saved?.drugs ?? [])
  const [context, setContext] = useState<PatientContext>(saved?.context ?? EMPTY_CONTEXT)
  const [result, setResult] = useState<AnalyzeResult | null>(saved?.result ?? null)
  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify({ drugs, context, result }))
    } catch {
      /* storage unavailable: state just won't survive a reload */
    }
  }, [drugs, context, result])
  return <Ctx.Provider value={{ drugs, setDrugs, context, setContext, result, setResult }}>{children}</Ctx.Provider>
}

export function useRegimen() {
  const c = useContext(Ctx)
  if (!c) throw new Error('useRegimen outside provider')
  return c
}
