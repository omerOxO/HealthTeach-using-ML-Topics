"""Severity tagging.

Priority:
1. DDInter's curated level (Major / Moderate / Minor) when the pair is in DDInter.
2. Otherwise a rule on the DrugBank interaction type (known or predicted).
   These rules are our own clinical-judgement estimates, not DrugBank labels, and the UI
   says so ("type-based estimate").
"""

# TDC DrugBank DDI type ids (see data/processed/ddi_types.parquet for the templates)
MAJOR_TYPES = {
    6,   # increases anticoagulant activity
    15,  # cardiotoxic
    20, 33,  # QTc prolongation
    25,  # AV block
    27,  # neuroexcitatory (seizure risk)
    35, 44,  # neurotoxic
    40,  # respiratory depression
    41, 79,  # CNS depression combined with hypotension/hypertension
    42, 68,  # hyperkalaemia
    48,  # rhabdomyolysis / myopathy
    50,  # heart failure
    53,  # increases antiplatelet activity
    57,  # nephrotoxic
    63,  # myelosuppression
    64,  # serotonergic (serotonin syndrome)
    66,  # bleeding
    80,  # hepatotoxic
    81,  # thrombogenic
    82,  # arrhythmogenic
    86,  # hypersensitivity
}
MINOR_TYPES = {
    1,   # photosensitising
    8,   # therapeutic efficacy increased
    13,  # absorption decreased (small)
    23,  # stimulatory
    28,  # dermatologic
    38,  # vasodilatory
    39,  # constipating
    43,  # protein binding decreased
    45,  # diagnostic agent effectiveness
    69,  # analgesic activity increased
}

DDINTER_MAP = {"Major": "major", "Moderate": "moderate", "Minor": "minor"}


def type_severity(type_id):
    if type_id is None or type_id < 0:
        return "moderate"  # unknown mechanism: do not under-call it
    if type_id in MAJOR_TYPES:
        return "major"
    if type_id in MINOR_TYPES:
        return "minor"
    return "moderate"


def pair_severity(ddinter_level, type_id, known, prob, threshold):
    """Return (severity, source). 'none' means no known or predicted interaction, never 'safe'."""
    if isinstance(ddinter_level, str) and ddinter_level in DDINTER_MAP:
        return DDINTER_MAP[ddinter_level], "DDInter"
    if known:
        return type_severity(type_id), "type-based estimate (known interaction)"
    if prob >= threshold:
        return type_severity(type_id), "type-based estimate (predicted interaction)"
    return "none", "no known or predicted interaction"
