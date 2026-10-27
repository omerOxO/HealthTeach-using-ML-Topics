"""FAERS evidence through openFDA (https://api.fda.gov/drug/event.json), cached locally.

Names: openFDA indexes FAERS reports by the harmonised `patient.drug.openfda.generic_name` field, which uses
RxNorm-style US generic names ("ASPIRIN", "WARFARIN SODIUM"). DrugBank names are mapped to the RxNorm ingredient
name first (Acetylsalicylic acid -> aspirin), then searched as a phrase, so salts and multi-ingredient products
containing the drug also count. Reports whose drug names openFDA could not harmonise are not counted.

Bias: FAERS is a spontaneous-reporting system. Counts depend on how widely a drug is used, media attention,
litigation, how long it has been marketed and who reports. A co-report means both drugs were listed on one report,
not that the combination caused the reaction. Counts cannot give incidence or prove causation.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import ROOT

CACHE = ROOT / "data" / "cache" / "faers.json"
URL = "https://api.fda.gov/drug/event.json"
TTL = 30 * 24 * 3600
TIMEOUT = 15
FIELD = "patient.drug.openfda.generic_name"

BIAS_NOTE = ("FAERS is a voluntary, spontaneous-reporting system. A report listing both drugs does not show that the "
             "combination caused the reaction. Counts depend on how widely each drug is used, publicity, litigation "
             "and reporting habits; they are not incidence rates and many events are never reported. openFDA only "
             "matches reports whose drug names it could harmonise.")


class FaersClient:
    def __init__(self, rxnorm, cache_file=CACHE):
        self.rxnorm = rxnorm
        self.cache_file = cache_file
        self._lock = threading.Lock()
        try:
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}
        self.pool = ThreadPoolExecutor(max_workers=6)

    # -------------------------------------------------------------- http + cache
    def _save(self):
        """Best effort: a cache that cannot be written (file locked by a scanner or a second server) only
        costs a repeat network call later."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_file.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.cache), encoding="utf-8")
            for _ in range(3):
                try:
                    tmp.replace(self.cache_file)
                    return
                except PermissionError:
                    time.sleep(0.2)
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    def _query(self, search=None, count=None, limit=1):
        key = f"{search}|{count}|{limit}"
        hit = self.cache.get(key)
        if hit and time.time() - hit["ts"] < TTL:
            return hit["data"]
        params = {"limit": limit}
        if os.environ.get("OPENFDA_API_KEY"):  # optional; raises the daily quota
            params["api_key"] = os.environ["OPENFDA_API_KEY"]
        if search:
            params["search"] = search
        if count:
            params["count"] = count
        for attempt in range(3):
            r = requests.get(URL, params=params, timeout=TIMEOUT)
            if r.status_code == 404:  # openFDA's "no matches"
                data = {"total": 0, "results": []}
                break
            if r.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            data = {"total": j.get("meta", {}).get("results", {}).get("total"), "results": j.get("results", [])}
            break
        with self._lock:
            self.cache[key] = {"ts": time.time(), "data": data}
            self._save()
        return data

    # -------------------------------------------------------------- names
    def faers_name(self, drug_name):
        """DrugBank name -> the generic name openFDA uses (via RxNorm), and how it was found."""
        rx = self.rxnorm.lookup(drug_name)
        ins = [c["name"] for c in (rx or {}).get("ingredients", []) if c["tty"] == "IN"]
        if rx and rx.get("match") == "exact" and len(ins) == 1:
            return ins[0].lower(), "RxNorm ingredient"
        return drug_name.lower(), "DrugBank name"

    @staticmethod
    def _term(name):
        return f'{FIELD}:"{name.replace(chr(34), "")}"'

    # -------------------------------------------------------------- reports
    def pair(self, name_a, name_b, top=12):
        (fa, how_a), (fb, how_b) = self.faers_name(name_a), self.faers_name(name_b)
        ta, tb = self._term(fa), self._term(fb)
        both = f"{ta} AND {tb}"
        jobs = {
            "total": (None, None, 1),
            "a": (ta, None, 1), "b": (tb, None, 1), "both": (both, None, 1),
            "serious": (f"{both} AND serious:1", None, 1),
            "death": (f"{both} AND seriousnessdeath:1", None, 1),
            "reactions": (both, "patient.reaction.reactionmeddrapt.exact", top),
            "reactions_a": (ta, "patient.reaction.reactionmeddrapt.exact", 100),
            "reactions_b": (tb, "patient.reaction.reactionmeddrapt.exact", 100),
            "dates": (both, "receivedate", 100),  # date counts come back in full at any limit
        }
        futs = {k: self.pool.submit(self._query, *v) for k, v in jobs.items()}
        try:
            res = {k: f.result() for k, f in futs.items()}
        except requests.RequestException as e:
            return {"ok": False, "error": f"openFDA unavailable ({type(e).__name__}). Try again later.",
                    "bias_note": BIAS_NOTE}
        n, na, nb, nab = (res[k]["total"] or 0 for k in ("total", "a", "b", "both"))
        expected = na * nb / n if n else 0

        def props(rows, total):
            return {r["term"]: r["count"] / total for r in rows} if total else {}
        pa, pb = props(res["reactions_a"]["results"], na), props(res["reactions_b"]["results"], nb)
        floor_a = min(pa.values(), default=0)  # not in a drug's top 100 -> at most the 100th share
        floor_b = min(pb.values(), default=0)
        reactions = []
        for r in res["reactions"]["results"]:
            share = r["count"] / nab if nab else 0
            alone = max(pa.get(r["term"], floor_a), pb.get(r["term"], floor_b))
            reactions.append({"term": r["term"].capitalize(), "count": r["count"], "share": round(share, 4),
                              "share_alone": round(alone, 4), "ratio": round(share / alone, 2) if alone else None})
        years = {}
        for r in res["dates"]["results"]:
            y = r["time"][:4]
            years[y] = years.get(y, 0) + r["count"]
        return {
            "ok": True,
            "names": {"a": {"query": fa, "how": how_a}, "b": {"query": fb, "how": how_b}},
            "total_reports": n, "reports_a": na, "reports_b": nb, "reports_both": nab,
            "expected_both": round(expected, 1), "co_report_ratio": round(nab / expected, 2) if expected else None,
            "serious": res["serious"]["total"] or 0, "deaths": res["death"]["total"] or 0,
            "reactions": reactions,
            "by_year": [{"year": y, "count": c} for y, c in sorted(years.items())],
            "bias_note": BIAS_NOTE, "source": "openFDA drug adverse event API (FAERS)",
            "query": both,
        }

    def co_reports(self, names):
        """Co-report counts for many pairs (dashboard overview). names: list of (key, name_a, name_b)."""
        def one(item):
            key, a, b = item
            fa, fb = self.faers_name(a)[0], self.faers_name(b)[0]
            return key, self._query(f"{self._term(fa)} AND {self._term(fb)}")["total"] or 0
        try:
            return dict(self.pool.map(one, names))
        except requests.RequestException:
            return None
