import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, type User } from './api'

interface Auth {
  user: User
  refresh: () => Promise<void>
  setUser: (u: User) => void
}

const GUEST: User = { username: null, role: 'guest' }
const Ctx = createContext<Auth | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User>(GUEST)
  const refresh = useCallback(async () => {
    try {
      setUser(await api.me())
    } catch {
      setUser(GUEST)
    }
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh])
  return <Ctx.Provider value={{ user, refresh, setUser }}>{children}</Ctx.Provider>
}

export function useAuth() {
  const c = useContext(Ctx)
  if (!c) throw new Error('useAuth outside provider')
  return c
}
