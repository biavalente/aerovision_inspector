from __future__ import annotations

import re
import unicodedata


# The checkpoint stores these stable technical identifiers.  The dashboard and
# audit trail present the Portuguese names below; keeping the identifiers stable
# avoids invalidating the existing trained weights.
ROI_CLASS_LABELS_PT = {
    "normal_roi": "Região normal",
    "ablation": "Ablação / perda de material",
    "breakdown": "Degradação superficial",
    "fracture": "Fratura",
    "groove": "Sulco",
    "unknown": "Tipo inconclusivo",
    "visual_anomaly_region": "Região visual suspeita",
    "dent": "Amolgadela (dent)",
}

TRAINABLE_DEFECT_LABELS = ("ablation", "breakdown", "fracture", "groove")


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


_ALIASES = {
    "normal roi": "normal_roi",
    "regiao normal": "normal_roi",
    "normal": "normal_roi",
    "ablation": "ablation",
    "ablacao": "ablation",
    "ablacao perda de material": "ablation",
    "perda de material": "ablation",
    "breakdown": "breakdown",
    "degradacao superficial": "breakdown",
    "degradacao": "breakdown",
    "fracture": "fracture",
    "fratura": "fracture",
    "groove": "groove",
    "sulco": "groove",
    "unknown": "unknown",
    "tipo inconclusivo": "unknown",
    "defect": "unknown",
    "regiao visual suspeita": "visual_anomaly_region",
    "dent": "dent",
    "amolgadela": "dent",
}


def canonical_defect_label(value: str | None) -> str | None:
    """Return a stable class identifier for Portuguese or legacy UI labels."""
    if not value:
        return None
    text = str(value).strip()
    if text.lower().startswith("tipo sugerido:"):
        text = text.split(":", 1)[1].strip()
    return _ALIASES.get(_key(text))


def portuguese_label(value: str | None) -> str:
    canonical = canonical_defect_label(value)
    if canonical and canonical in ROI_CLASS_LABELS_PT:
        return ROI_CLASS_LABELS_PT[canonical]
    return str(value or "Tipo inconclusivo")
