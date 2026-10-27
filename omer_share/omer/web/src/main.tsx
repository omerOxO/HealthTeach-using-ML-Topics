import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'
import { RegimenProvider } from './state'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <RegimenProvider>
        <App />
      </RegimenProvider>
    </BrowserRouter>
  </StrictMode>,
)
