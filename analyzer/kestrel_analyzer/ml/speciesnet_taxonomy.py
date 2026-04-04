"""Parse SpeciesNet semicolon taxonomy strings for pipeline routing."""

from __future__ import annotations

# Full label strings from SpeciesNet taxonomy (see speciesnet.constants.Classification).
_SPECIESNET_BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
_SPECIESNET_VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
_SPECIESNET_HUMAN = (
    "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
)


def split_taxonomy(raw: str) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(";")]


def wildlife_display_name(raw: str) -> str:
    """Last non-empty semicolon segment (common name when present)."""
    parts = split_taxonomy(raw)
    for seg in reversed(parts):
        if seg:
            return seg
    return raw.strip() or "unknown"


def is_ignored_prediction(raw: str) -> bool:
    """blank, vehicle, human — skip for downstream processing."""
    if not raw or not str(raw).strip():
        return True
    s = str(raw).strip()
    if s == _SPECIESNET_BLANK:
        return True
    if s == _SPECIESNET_VEHICLE:
        return True
    if s == _SPECIESNET_HUMAN:
        return True
    parts_lower = [p.lower() for p in split_taxonomy(s) if p]
    last = parts_lower[-1] if parts_lower else ""
    if last in ("blank", "vehicle", "human"):
        return True
    return False


def is_bird_taxon(raw: str) -> bool:
    """True if taxonomy includes class Aves (segment 'aves')."""
    parts = [p.lower() for p in split_taxonomy(raw) if p]
    return "aves" in parts


def route_prediction(
    raw_prediction: str,
    *,
    wildlife_enabled: bool,
) -> tuple[str, str | None]:
    """Return (route, pred_class_label).

    route: 'ignore' | 'bird' | 'wildlife'
    pred_class_label: 'bird' for aves, or display name for wildlife; None if ignore.
    """
    if is_ignored_prediction(raw_prediction):
        return "ignore", None
    if is_bird_taxon(raw_prediction):
        return "bird", "bird"
    if not wildlife_enabled:
        return "ignore", None
    return "wildlife", wildlife_display_name(raw_prediction)
