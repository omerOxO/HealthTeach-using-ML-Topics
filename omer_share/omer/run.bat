@echo off
rem One-click start: builds the web app if needed, then serves app + API at http://localhost:8000
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. See USER_GUIDE.md, section "First-time setup".
  pause
  exit /b 1
)
if not exist "artifacts\gnn_rgcn_merged_random.pt" (
  echo Trained model not found. Run:  .venv\Scripts\python -m ddi.run_all --quick
  pause
  exit /b 1
)
if not exist "web\dist\index.html" (
  echo Building the web app...
  pushd web
  if not exist node_modules call npm install
  call npm run build
  popd
)

echo.
echo Starting Interaction Check at http://localhost:8000  (Ctrl+C to stop)
start "" http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn api.main:app --port 8000
