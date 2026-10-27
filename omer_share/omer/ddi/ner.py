"""Prescription / discharge-note extraction: medication mentions with dose, frequency, route and duration.

Mentions come from up to three sources, merged by character span:
  * a transformer biomedical NER model (d4data/biomedical-ner-all, a DistilBERT model trained on the MACCROBAT
    clinical-case corpus) for Medication / Dosage / Frequency / Administration / Duration spans;
  * scispaCy's en_ner_bc5cdr_md (CHEMICAL entities), if installed;
  * the local name index (DrugBank names, SYNONYMS), which also catches brand names the models miss.
Each medication span is resolved to a DrugBank drug through the local index, then RxNorm.
Dose, frequency, route and duration are read from the text after the mention (up to the next mention or
line end) with rules, and the model's Dosage/Frequency spans fill any gaps.
Set DDI_NER=off to skip the models (dictionary only).
"""
import logging
import os
import re
import threading

log = logging.getLogger(__name__)

HF_MODEL = "d4data/biomedical-ner-all"
SPACY_MODEL = "en_ner_bc5cdr_md"

UNIT = r"(?:mg|mcg|µg|ug|micrograms?|milligrams?|g|grams?|ml|mL|units?|iu|IU|meq|mEq|mmol|%|puffs?|tabs?|tablets?|caps?|capsules?|drops?|sprays?|patch(?:es)?)"
DOSE_RE = re.compile(rf"\b(\d+(?:[.,]\d+)?(?:\s*(?:-|to|/)\s*\d+(?:[.,]\d+)?)?)\s*({UNIT})(?![a-z])", re.I)
FREQ_RE = re.compile(
    r"\b(?:once|twice|three times|four times|[1-4]\s?x|one time|two times)\s+(?:a\s+|per\s+|each\s+)?(?:day|daily|week|weekly|night|month)\b"
    r"|\bevery\s+(?:\d+(?:\s*(?:-|to)\s*\d+)?\s*(?:hours?|hrs?|h|days?|weeks?)|morning|evening|night|other day|day|week)\b"
    r"|\b(?:q\.?\s?\d+\s?(?:h|hr|hrs|hours?)|q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|q\.?h\.?s\.?|q\.?a\.?m\.?|q\.?p\.?m\.?|q\.?o\.?d\.?|p\.?r\.?n\.?|stat)(?![a-z])"
    r"|\b(?:daily|nightly|weekly|monthly|at bedtime|at night|in the morning|in the evening|as needed|when needed|with meals|"
    r"(?:before|after|with)\s+(?:breakfast|lunch|dinner|supper|meals|food))\b",
    re.I)
ROUTE_RE = re.compile(r"\b(?:p\.?o\.?|by mouth|orally|oral|i\.?v\.?|intravenous(?:ly)?|i\.?m\.?|intramuscular(?:ly)?|"
                      r"s\.?c\.?|subcut(?:aneous(?:ly)?)?|s\.?l\.?|sublingual(?:ly)?|topical(?:ly)?|inhaled|inhalation|"
                      r"rectal(?:ly)?|p\.?r\.?|transdermal(?:ly)?|nasal(?:ly)?)(?![a-z])", re.I)
DUR_RE = re.compile(r"\b(?:for|x)\s*\d+\s*(?:days?|weeks?|months?|d|wk|wks)\b", re.I)

FREQ_NORMAL = {"qd": "once daily", "od": "once daily", "bid": "twice daily", "tid": "three times daily",
               "qid": "four times daily", "qhs": "at bedtime", "qam": "every morning", "qpm": "every evening",
               "qod": "every other day", "prn": "as needed", "stat": "immediately"}
ROUTE_NORMAL = {"po": "oral", "by mouth": "oral", "orally": "oral", "iv": "intravenous", "im": "intramuscular",
                "sc": "subcutaneous", "subcut": "subcutaneous", "sl": "sublingual", "pr": "rectal"}


def _norm_freq(s):
    k = re.sub(r"[.\s]", "", s.lower())
    m = re.fullmatch(r"q(\d+)(h|hr|hrs|hours?)", k)
    if m:
        return f"every {m.group(1)} hours"
    return FREQ_NORMAL.get(k, s.lower())


def _norm_route(s):
    k = re.sub(r"\.", "", s.lower())
    return ROUTE_NORMAL.get(k, k.rstrip("ly") if k.endswith("ally") else k)


class PrescriptionNER:
    def __init__(self, svc):
        self.svc = svc
        self._hf = self._spacy = None
        self._loaded = False
        self._lock = threading.Lock()
        self.models = []

    # ------------------------------------------------------------ models (lazy)
    def _load(self):
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if os.environ.get("DDI_NER", "").lower() == "off":
                return
            try:
                from transformers import pipeline
                self._hf = pipeline("token-classification", model=HF_MODEL, aggregation_strategy="simple", device=-1)
                self.models.append(HF_MODEL)
            except Exception as e:  # not installed or not downloaded: dictionary + rules still work
                log.warning("HF NER unavailable: %s", e)
            try:
                import warnings
                import spacy
                with warnings.catch_warnings():  # model built for spaCy 3.7; loads on 3.8 with this fix
                    warnings.simplefilter("ignore")
                    self._spacy = spacy.load(SPACY_MODEL, config={
                        "components.tok2vec.model.embed.include_static_vectors": True,
                        "components.ner.model.tok2vec.embed.include_static_vectors": True})
                self.models.append(f"scispaCy {SPACY_MODEL}")
            except Exception as e:
                log.warning("scispaCy unavailable: %s", e)

    def _model_spans(self, text):
        """-> list of (start, end, label, score, source)."""
        spans = []
        if self._hf is not None:
            merged = []
            for e in self._hf(text):
                g = e["entity_group"]
                # glue word pieces and adjacent fragments of one entity ("war" + "##farin")
                if merged and merged[-1][2] == g and e["start"] - merged[-1][1] <= 1 and \
                        (e["word"].startswith("##") or text[merged[-1][1]:e["start"]].strip() == ""):
                    s, _, _, sc, src = merged[-1]
                    merged[-1] = (s, e["end"], g, min(sc, float(e["score"])), src)
                else:
                    merged.append((e["start"], e["end"], g, float(e["score"]), "NER"))
            spans += merged
        if self._spacy is not None:
            for ent in self._spacy(text).ents:
                if ent.label_ == "CHEMICAL":
                    spans.append((ent.start_char, ent.end_char, "Medication", 1.0, "scispaCy"))
        return spans

    # ------------------------------------------------------------ dictionary
    def _dict_spans(self, text, max_words=4):
        toks = [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)]
        out, i = [], 0
        while i < len(toks):
            for n in range(min(max_words, len(toks) - i), 0, -1):
                key = " ".join(t[2] for t in toks[i:i + n])
                j = self.svc.lookup.get(key)
                if j is not None and not toks[i][2].isdigit() and len(key) > 2:
                    out.append((toks[i][0], toks[i + n - 1][1], j))
                    i += n
                    break
            else:
                i += 1
        return out

    def _resolve_span(self, s):
        s = s.strip(" .,;:()[]-")
        r = self.svc.resolve_all(s)
        if not r["idxs"] and " " in s:  # "aspirin EC", "metoprolol succinate ER": try leading words
            words = s.split()
            for n in range(len(words) - 1, 0, -1):
                r = self.svc.resolve_all(" ".join(words[:n]))
                if r["idxs"]:
                    break
        return r

    # ------------------------------------------------------------ main
    def extract(self, text):
        self._load()
        model_spans = self._model_spans(text) if text.strip() else []
        mentions = []  # dicts with start, end, idxs, source

        def overlapping(s, e):
            return next((m for m in mentions if s < m["end"] and e > m["start"]), None)

        unmatched = []
        for s, e, label, score, src in model_spans:
            if label != "Medication":
                continue
            r = self._resolve_span(text[s:e])
            if not r["idxs"]:
                frag = text[s:e].strip()
                if len(frag) > 2 and not any(r.fullmatch(frag) for r in (FREQ_RE, DOSE_RE, ROUTE_RE, DUR_RE)):
                    unmatched.append(frag)
                continue
            m = overlapping(s, e)
            if m:
                m["sources"].add(src)
                continue
            mentions.append({"start": s, "end": e, "idxs": r["idxs"], "sources": {src}, "score": round(score, 3),
                             "via": "RxNorm" if r["source"] == "rxnorm" else None})
        for s, e, j in self._dict_spans(text):
            m = overlapping(s, e)
            if m:
                m["sources"].add("dictionary")
                m["start"], m["end"] = min(m["start"], s), max(m["end"], e)
                if j not in m["idxs"]:
                    m["idxs"] = [j]
            else:
                mentions.append({"start": s, "end": e, "idxs": [j], "sources": {"dictionary"}, "score": None, "via": None})
        mentions.sort(key=lambda m: m["start"])

        # attributes: text after the mention up to the next mention or the end of the line
        attr_spans = [(s, e, label) for s, e, label, _, _ in model_spans if label in ("Dosage", "Frequency", "Duration", "Administration")]
        for k, m in enumerate(mentions):
            nxt = mentions[k + 1]["start"] if k + 1 < len(mentions) else len(text)
            eol = text.find("\n", m["end"])
            stop = min(nxt, eol if eol >= 0 else len(text))
            window = text[m["end"]:stop]
            dose = DOSE_RE.search(window)
            freqs = [f.group() for f in FREQ_RE.finditer(window)]
            route = ROUTE_RE.search(window)
            dur = DUR_RE.search(window)
            m["dose"] = re.sub(r"\s+", " ", dose.group()).strip() if dose else None
            m["frequency"] = ", ".join(dict.fromkeys(_norm_freq(f) for f in freqs)) or None
            m["route"] = _norm_route(route.group()) if route else None
            m["duration"] = dur.group() if dur else None
            if not m["frequency"]:
                f = [text[s:e] for s, e, lab in attr_spans if lab == "Frequency" and m["end"] <= s < stop]
                m["frequency"] = f[0].strip() if f else None
            if not m["dose"]:
                d = [text[s:e] for s, e, lab in attr_spans if lab == "Dosage" and m["end"] <= s < stop]
                m["dose"] = d[0].strip() if d else None

        found, out = [], []
        for m in mentions:
            for j in m["idxs"]:
                if j not in found:
                    found.append(j)
            src = sorted(m["sources"])
            out.append({"text": text[m["start"]:m["end"]], "start": m["start"], "end": m["end"],
                        "drugs": [self.svc.drug_card(j) for j in m["idxs"]], "source": " + ".join(src),
                        "via": m["via"], "score": m["score"], "dose": m["dose"], "frequency": m["frequency"],
                        "route": m["route"], "duration": m["duration"]})
        resolved_text = {o["text"].lower() for o in out}
        return {"drugs": [self.svc.drug_card(j) for j in found], "mentions": out,
                "unmatched": [u for u in dict.fromkeys(unmatched) if u.lower() not in resolved_text],
                "models": self.models or ["dictionary only"]}
