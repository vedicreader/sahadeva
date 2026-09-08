"""dhvani — Sanskrit text frontend for TTS.

Deva/any-Brahmic -> SLP1 -> Kannada routing (the IndicF5 champion path), with
first-class Vedic svara handling that upstream frontends drop on the floor.
"""
from .translit import to_deva, to_slp1, to_kannada, detect_script, strip_punct
from .svara import SvaraT, extract_svara, strip_svara, restore_svara, svara_spans, has_svara
from .sandhi import SandhiT, apply_sandhi, visarga_sandhi, visarga_echo_final, homorganic_anusvara
from .meter import syllabify, weights, WeightT, n_aksharas
from .frontend import Frontend, Prepped, prep

__all__ = ["to_deva", "to_slp1", "to_kannada", "detect_script", "strip_punct",
           "SvaraT", "extract_svara", "strip_svara", "restore_svara", "svara_spans", "has_svara",
           "SandhiT", "apply_sandhi", "visarga_sandhi", "visarga_echo_final", "homorganic_anusvara",
           "syllabify", "weights", "WeightT", "n_aksharas",
           "Frontend", "Prepped", "prep"]
