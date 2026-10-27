"""FastAPI backend.

    uvicorn api.main:app --port 8000

Serves /api/* and, if web/dist exists (npm run build), the React app at /.
"""
from pathlib import Path
from typing import List, Optional

import logging
import threading

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ddi.alternatives import Alternatives
from ddi.audit import AuditStore
from ddi.config import DISCLAIMER, ROOT
from ddi.faers import BIAS_NOTE, FaersClient
from ddi.service import DDIService

log = logging.getLogger("api")
COOKIE = "ddi_session"

app = FastAPI(title="DDI predictor", version="0.1.0",
              description="Drug interaction and adverse-effect predictor. " + DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_svc: Optional[DDIService] = None


def svc() -> DDIService:
    global _svc
    if _svc is None:
        _svc = DDIService()
    return _svc


_extras = {}
_extras_lock = threading.Lock()


def _extra(name, make):
    with _extras_lock:
        if name not in _extras:
            _extras[name] = make()
        return _extras[name]


def alternatives() -> Alternatives:
    return _extra("alt", lambda: Alternatives(svc()))


def faers() -> FaersClient:
    return _extra("faers", lambda: FaersClient(svc().rxnorm))


def audit() -> AuditStore:
    return _extra("audit", AuditStore)


@app.on_event("startup")
def _warm():
    svc()
    audit()
    # the NER models take ~20 s to load; do it in the background so the first paste is quick
    threading.Thread(target=lambda: svc().extract("warm up"), daemon=True).start()


class PatientContext(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=120)
    kidney: bool = False
    liver: bool = False
    conditions: List[str] = []
    allergies: List[str] = []


class AnalyzeRequest(BaseModel):
    drugs: List[str] = Field(..., min_length=2, max_length=20)
    context: PatientContext = PatientContext()


@app.get("/api/health")
def health():
    s = svc()
    return {"ok": True, "drugs": s.n, "model": s.model_info}


@app.get("/api/drugs/search")
def search(q: str = Query(..., min_length=1), limit: int = 10):
    return svc().search(q, limit)


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=20_000)


@app.post("/api/extract")
def extract(req: ExtractRequest):
    return svc().extract(req.text)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    ctx = req.context.model_dump()
    res = svc().analyze(req.drugs, ctx)
    if len(res["drugs"]) < 2:
        raise HTTPException(422, {"message": "Need at least two recognised drugs.", "unknown": res["unknown"]})
    user = audit().user(request.cookies.get(COOKIE))
    try:
        audit().log_analysis(res, ctx, user["role"] if user else "guest")
    except Exception:  # never fail an analysis because the audit write failed
        log.exception("audit log write failed")
    return res


@app.get("/api/pair")
def pair(a: str, b: str):
    d = svc().pair_detail(a, b)
    if d is None:
        raise HTTPException(404, "Unknown drug id")
    return d


@app.get("/api/graph/drug/{drug_id}")
def graph(drug_id: str, highlight: Optional[str] = None):
    g = svc().neighbourhood(drug_id, highlight)
    if g is None:
        raise HTTPException(404, "Unknown drug id")
    return g


@app.get("/api/metrics")
def metrics():
    return svc().read_json("metrics.json") or {}


@app.get("/api/eda")
def eda():
    return svc().read_json("eda.json") or {}


@app.get("/api/model-card")
def model_card():
    s = svc()
    return {"disclaimer": DISCLAIMER, "model": s.model_info, "threshold": s.threshold,
            "card": (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8") if (ROOT / "MODEL_CARD.md").exists() else ""}


# ---------------------------------------------------------------- safer alternatives
@app.get("/api/alternatives")
def alternatives_for(a: str, b: str, meds: str = ""):
    """Safer alternatives for the pair a+b given the rest of the list (meds: comma-separated DrugBank ids)."""
    s = svc()
    ia, ib = s.id2idx.get(a), s.id2idx.get(b)
    if ia is None or ib is None or ia == ib:
        raise HTTPException(404, "Unknown drug id")
    med_idx = [s.id2idx[m] for m in meds.split(",") if m in s.id2idx][:20]
    return alternatives().suggest(ia, ib, med_idx)


# ---------------------------------------------------------------- FAERS
def _names(ids):
    s = svc()
    idx = [s.id2idx.get(i) for i in ids]
    if any(i is None for i in idx):
        raise HTTPException(404, "Unknown drug id")
    return [s.names[i] for i in idx]


@app.get("/api/faers/pair")
def faers_pair(a: str, b: str):
    na, nb = _names([a, b])
    res = faers().pair(na, nb)
    res["drugs"] = {"a": svc().drug_card(svc().id2idx[a]), "b": svc().drug_card(svc().id2idx[b])}
    return res


@app.get("/api/faers/overview")
def faers_overview(ids: str):
    """Co-report counts for every pair of up to 10 drugs (FAERS dashboard)."""
    id_list = list(dict.fromkeys(x for x in ids.split(",") if x))[:10]
    names = _names(id_list)
    items = [(f"{id_list[i]}|{id_list[j]}", names[i], names[j])
             for i in range(len(id_list)) for j in range(i + 1, len(id_list))]
    counts = faers().co_reports(items)
    if counts is None:
        return {"ok": False, "error": "openFDA unavailable. Try again later."}
    s = svc()
    pairs = [{"a": s.drug_card(s.id2idx[k.split("|")[0]]), "b": s.drug_card(s.id2idx[k.split("|")[1]]), "reports": c}
             for k, c in counts.items()]
    pairs.sort(key=lambda p: -p["reports"])
    return {"ok": True, "pairs": pairs, "bias_note": BIAS_NOTE}


# ---------------------------------------------------------------- login + audit log
class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=256)


@app.post("/api/auth/login")
def login(req: LoginRequest, response: Response):
    r = audit().login(req.username.strip(), req.password)
    if r is None:
        raise HTTPException(401, "Wrong username or password.")
    token, user = r
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=8 * 3600)
    return user


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    audit().logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request):
    return audit().user(request.cookies.get(COOKIE)) or {"username": None, "role": "guest"}


@app.get("/api/audit")
def audit_log(request: Request, limit: int = Query(100, le=500), offset: int = 0, level: Optional[str] = None):
    user = audit().user(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "Sign in as an administrator to view the audit log.")
    if user["role"] != "admin":
        raise HTTPException(403, "Only administrators can view the audit log.")
    return audit().entries(limit, offset, level)


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
