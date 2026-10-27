"""RxNorm normalisation through the free RxNav REST API (https://rxnav.nlm.nih.gov), with a local cache.

Used only when a typed name is not in the local DrugBank name index: brand or salt names
(Tylenol, Bayer, warfarin sodium) are mapped to their RxNorm ingredient names, which are then
looked up locally. Every answer, including "not found" and network failures, is cached in
data/cache/rxnorm.json so each name hits the network at most once (failures are retried after a day).
"""
import json
import threading
import time

import requests

from .config import ROOT

CACHE = ROOT / "data" / "cache" / "rxnorm.json"
BASE = "https://rxnav.nlm.nih.gov/REST"
TIMEOUT = 6
RETRY_FAILED_AFTER = 24 * 3600


class RxNorm:
    def __init__(self, cache_file=CACHE):
        self.cache_file = cache_file
        self._lock = threading.Lock()
        try:
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

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

    @staticmethod
    def _get(path, **params):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _rxcuis(self, name):
        """Exact/normalised match first, then RxNav's approximate match (misspellings)."""
        ids = self._get("rxcui.json", name=name, search=2).get("idGroup", {}).get("rxnormId") or []
        if ids:
            return ids[:3], "exact"
        cands = self._get("approximateTerm.json", term=name, maxEntries=5).get("approximateGroup", {}).get("candidate") or []
        cands.sort(key=lambda c: int(c.get("rank", 99)))
        return list(dict.fromkeys(c["rxcui"] for c in cands))[:4], "approximate"

    def _ingredients(self, rxcui):
        groups = self._get(f"rxcui/{rxcui}/related.json", tty="IN PIN MIN").get("relatedGroup", {}).get("conceptGroup") or []
        names = []
        for g in groups:  # IN (ingredient) before PIN (precise ingredient, e.g. the salt) before MIN
            for c in g.get("conceptProperties") or []:
                names.append({"tty": g["tty"], "name": c["name"], "rxcui": c["rxcui"]})
        order = {"IN": 0, "PIN": 1, "MIN": 2}
        return sorted(names, key=lambda c: order.get(c["tty"], 3))

    def lookup(self, name):
        """-> {"query", "rxcui", "match", "ingredients": [{"name", "tty", "rxcui"}], "error"?} or None."""
        key = str(name).strip().lower()
        if len(key) < 3:
            return None
        hit = self.cache.get(key)
        if hit and not (hit.get("error") and time.time() - hit.get("ts", 0) > RETRY_FAILED_AFTER):
            return hit
        try:
            ids, match = self._rxcuis(key)
            ingredients, used = [], None
            for rxcui in ids:  # best candidate that has ingredients (retired brands have none)
                ingredients = self._ingredients(rxcui)
                if ingredients:
                    used = rxcui
                    break
            res = {"query": key, "rxcui": used or (ids[0] if ids else None), "match": match if ids else None,
                   "ingredients": ingredients, "ts": time.time()}
        except (requests.RequestException, ValueError) as e:
            res = {"query": key, "rxcui": None, "match": None, "ingredients": [], "error": type(e).__name__,
                   "ts": time.time()}
        with self._lock:
            self.cache[key] = res
            self._save()
        return res
