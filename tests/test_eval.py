"""Tests for the evaluation metrics.

The properties pinned here are the ones that make the numbers comparable at all: scoring is
script-independent, accent-independent, and length-weighted.
"""
import numpy as np
import pytest
from sahaeval.cer import cer, wer, run as cer_run, edit_distance
from sahaeval.conjunct import clusters_in, score_pair
from sahaeval.svara import prescribed_contour, fix_octave_jumps, to_semitones, score

# ── CER/WER ──────────────────────────────────────────────────────────────────────────
def test_identical_is_zero():
    assert cer("रामः गच्छति", "रामः गच्छति") == 0.0
    assert wer("रामः गच्छति", "रामः गच्छति") == 0.0

def test_script_independent():
    "Scoring happens in SLP1, so Devanagari vs Kannada of the same phones is not an error."
    assert cer("रामः गच्छति", "ರಾಮಃ ಗಚ್ಛತಿ") == 0.0

def test_accent_independent_by_default():
    "No Sanskrit ASR transcribes svara — charging the model for it would be scoring noise."
    assert cer("मे॒ मय॑श्च", "मे मयश्च") == 0.0

def test_substitution_is_counted():
    assert cer("रामः", "रामम्") > 0

def test_aggregate_is_length_weighted():
    """A long utterance must not be outweighed by a short one.

    Averaging per-utterance rates is the usual way this number gets quietly inflated.
    """
    pairs = [("रामः", "रामम्"), ("गच्छति गच्छति गच्छति गच्छति", "गच्छति गच्छति गच्छति गच्छति")]
    _, agg = cer_run(pairs)
    mean_of_rates = np.mean([cer(a, b) for a, b in pairs])
    assert agg["cer"] < mean_of_rates

def test_edit_distance_basics():
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("", "abc") == 3

# ── conjuncts ────────────────────────────────────────────────────────────────────────
def test_cluster_detection():
    c = clusters_in("कृष्णः ज्ञानं क्षेत्रम्")
    assert c["kz"] == 1 and c["jY"] == 1        # kṣ and jñ

def test_cluster_survival():
    "jñ smoothed to plain n is exactly the failure this metric exists to catch."
    s = score_pair("कृष्णः ज्ञानं क्षेत्रम्", "कृष्णः जानं क्षेत्रम्")
    assert s["n_clusters"] == 2 and s["kept"] == 1 and s["recall"] == 0.5
    assert "jY" in s["missed"]

def test_perfect_survival():
    s = score_pair("ज्ञानं क्षेत्रम्", "ज्ञानं क्षेत्रम्")
    assert s["recall"] == 1.0

# ── svara ────────────────────────────────────────────────────────────────────────────
def test_prescribed_contour_ordering():
    "Anudātta must map below svarita — inverting this silently inverts the whole metric."
    c = prescribed_contour("AS", 2)
    assert c[0] < c[1]

def test_prescribed_contour_length():
    assert len(prescribed_contour("AUS", 30)) == 30
    assert len(prescribed_contour("", 10)) == 10

def test_octave_jump_correction():
    "pyin halving artifacts dominate the correlation if left alone."
    f0 = np.array([200.0] * 20 + [100.0] * 3)      # 3 halved frames
    fixed = fix_octave_jumps(f0)
    assert np.allclose(fixed[-3:], 200.0, rtol=0.15)

def test_semitones_are_median_relative():
    "Speaker pitch range must not affect the score."
    a = to_semitones(np.array([100.0, 200.0, 400.0]))
    b = to_semitones(np.array([200.0, 400.0, 800.0]))   # same contour, octave up
    assert np.allclose(a, b)

def test_unaccented_is_skipped():
    "Unaccented text has no prescription to score against."
    assert score("nonexistent.wav", "UUUU").get("skipped") == "unaccented"
    assert score("nonexistent.wav", "").get("skipped") == "unaccented"
