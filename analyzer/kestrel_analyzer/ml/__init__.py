"""ML wrappers; use lazy exports so `import kestrel_analyzer.ml.speciesnet_taxonomy` does not load TensorFlow."""

from __future__ import annotations

from typing import Any

__all__ = ["BirdSpeciesClassifier", "QualityClassifier", "SpeciesNetSAMHQWrapper"]


def __getattr__(name: str) -> Any:
    if name == "BirdSpeciesClassifier":
        from .bird_species import BirdSpeciesClassifier

        return BirdSpeciesClassifier
    if name == "QualityClassifier":
        from .quality import QualityClassifier

        return QualityClassifier
    if name == "SpeciesNetSAMHQWrapper":
        from .speciesnet_sam_hq import SpeciesNetSAMHQWrapper

        return SpeciesNetSAMHQWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
