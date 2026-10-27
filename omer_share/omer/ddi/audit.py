"""Audit log and simple local user roles (prototype), both in SQLite at data/audit.sqlite.

Audit rows hold the timestamp, the drug list, the result summary and the role of whoever ran the analysis.
They never hold patient details (age, conditions, allergies), usernames, IP addresses or free text: only
which kinds of patient context were supplied.

Roles: "admin" can read the audit log; "clinician" can sign in; anyone can run analyses (logged as "guest").
Seed accounts are created on first start from DDI_ADMIN_PASSWORD / DDI_CLINICIAN_PASSWORD, defaulting to
admin/admin and clinician/clinician. Change them before showing the app to anyone. Passwords are stored as
salted PBKDF2-SHA256 hashes; sessions are random tokens in an HttpOnly cookie, kept in memory, 8 hours.
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone

from .config import ROOT

DB = ROOT / "data" / "audit.sqlite"
SESSION_SECONDS = 8 * 3600
ROLES = ("admin", "clinician")


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


class AuditStore:
    def __init__(self, path=DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.sessions = {}  # token -> (username, role, expiry)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, role TEXT NOT NULL,
                drugs TEXT NOT NULL, n_drugs INTEGER, unknown INTEGER, pairs_checked INTEGER, flagged INTEGER,
                level TEXT, score REAL, top_pairs TEXT, context_used TEXT, model TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY, role TEXT NOT NULL, salt BLOB NOT NULL, pw TEXT NOT NULL)""")
            for name, role, env in (("admin", "admin", "DDI_ADMIN_PASSWORD"), ("clinician", "clinician", "DDI_CLINICIAN_PASSWORD")):
                if not c.execute("SELECT 1 FROM users WHERE username=?", (name,)).fetchone():
                    salt = os.urandom(16)
                    c.execute("INSERT INTO users VALUES (?,?,?,?)", (name, role, salt, _hash(os.environ.get(env, name), salt)))

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    # ------------------------------------------------------------ auth
    def login(self, username, password):
        with self._conn() as c:
            row = c.execute("SELECT role, salt, pw FROM users WHERE username=?", (username,)).fetchone()
        if not row or not hmac.compare_digest(_hash(password, row[1]), row[2]):
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self.sessions[token] = (username, row[0], time.time() + SESSION_SECONDS)
        return token, {"username": username, "role": row[0]}

    def user(self, token):
        s = self.sessions.get(token or "")
        if not s or s[2] < time.time():
            self.sessions.pop(token or "", None)
            return None
        return {"username": s[0], "role": s[1]}

    def logout(self, token):
        self.sessions.pop(token or "", None)

    # ------------------------------------------------------------ audit
    def log_analysis(self, result, context, role):
        ctx_used = [k for k in ("age", "kidney", "liver", "conditions", "allergies") if context.get(k)]
        top = [{"a": p["a"]["name"], "b": p["b"]["name"], "severity": p["severity"], "status": p["status"],
                "score": p["score"]} for p in result["pairs"] if p["severity"] != "none"][:5]
        reg = result["regimen"]
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), role,
               json.dumps([{"id": d["id"], "name": d["name"]} for d in result["drugs"]]),
               len(result["drugs"]), len(result["unknown"]), reg["pairs_checked"], reg["flagged"],
               reg["level"], reg["score"], json.dumps(top), json.dumps(ctx_used), result["model"]["name"])
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO analyses (ts, role, drugs, n_drugs, unknown, pairs_checked, flagged, level, score, "
                      "top_pairs, context_used, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)

    def entries(self, limit=200, offset=0, level=None):
        q, args = "SELECT * FROM analyses", []
        if level:
            q, args = q + " WHERE level=?", [level]
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(q + " ORDER BY id DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
            total = c.execute("SELECT COUNT(*) FROM analyses" + (" WHERE level=?" if level else ""), args).fetchone()[0]
            levels = dict(c.execute("SELECT level, COUNT(*) FROM analyses GROUP BY level").fetchall())
        out = []
        for r in rows:
            d = dict(r)
            for k in ("drugs", "top_pairs", "context_used"):
                d[k] = json.loads(d[k] or "[]")
            out.append(d)
        return {"total": total, "levels": levels, "entries": out}
