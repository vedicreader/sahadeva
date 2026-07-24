"""Golden tests for the dhvani frontend.

Sandhi and accent bugs are silent quality killers — they do not raise, they just make the
model mispronounce. Every pair below is hand-verified; treat a failure as a real regression
rather than something to re-baseline.
"""
import pytest
from dhvani import (prep, Frontend, SandhiT, SvaraT, to_deva, to_slp1, to_kannada,
                    strip_punct, extract_svara, restore_svara, strip_svara, has_svara)
from dhvani.svara import DIRGHA_SVARITA, SVARITA, ANUDATTA, normalize, contour, to_tags
from dhvani.sandhi import satva, homorganic_anusvara, visarga_sandhi, visarga_echo_final
from dhvani.meter import weight_str, aksharas, syllabify

# ── transliteration ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("deva,slp", [
    ("गणानां त्वा गणपतिं", "gaRAnAM tvA gaRapatiM"),
    ("अग्निमीळे पुरोहितं",  "agnimILe purohitaM"),
    ("तत्सवितुर्वरेण्यं",     "tatsaviturvareRyaM"),
])
def test_deva_to_slp1(deva, slp):
    assert to_slp1(deva, fix_vocalic_rr=False) == slp

def test_kannada_routing():
    "The champion path is Deva -> SLP1 -> Kannada; feeding Deva directly triggers schwa-deletion."
    assert to_kannada(to_slp1("रामः")) == "ರಾಮಃ"

def test_vocalic_rr_fix():
    "ṝ -> repha+ū: IndicF5 mispronounces Kannada ೄ (U+0CC4), so it is patched at the SLP1 layer."
    assert "F" not in to_slp1("पितॄन्", fix_vocalic_rr=True)
    assert "ೄ" not in to_kannada(to_slp1("पितॄन्", fix_vocalic_rr=True))

def test_script_autodetect():
    "Any Brahmic input normalizes to Devanagari before the pipeline runs."
    assert to_deva("ರಾಮಃ") == "रामः"

def test_strip_punct_keeps_pronounced_marks():
    "Daṇḍas go; avagraha and ॐ stay — they are pronounced."
    assert strip_punct("॥ ॐ रामः ॥") == "ॐ रामः"
    assert "ऽ" in strip_punct("मेऽनुकामः")

def test_strip_punct_preserves_svara():
    "Accent marks must survive punctuation stripping — they are not punctuation."
    assert has_svara(strip_punct("मे॒ मय॑श्च ।"))

# ── svara ────────────────────────────────────────────────────────────────────────────
def test_svara_labels_align_to_aksharas():
    "One label per akshara, unmarked syllables reported as UDATTA rather than dropped."
    bare, labels = extract_svara("मे॒ मय॑श्च")
    assert len(labels) == len(aksharas(bare))
    assert labels[0] is SvaraT.ANUDATTA

def test_svara_roundtrip():
    for t in ["मे॒ मय॑श्च मे", "अ॒ग्निमी॑ळे पु॒रोहि॑तं", "रामः"]:
        bare, labels = extract_svara(t)
        assert restore_svara(bare, labels) == t

def test_known_lossy_case_mark_inside_conjunct():
    """Documented limitation, pinned so it cannot drift silently.

    In 'अहीग्॑श्च' the accent mark sits between the virāma and the next consonant, i.e. *inside*
    a conjunct cluster that forms a single akshara. One label per syllable cannot encode where
    within the cluster the glyph sat, so the string does not round-trip exactly. The label
    channel is still correct — that syllable is svarita — and svara is a syllable-level pitch
    property, so nothing the model consumes is lost. 2 of 2,619 corpus lines hit this.
    """
    src = "अहीग्॑श्च"
    bare, labels = extract_svara(src)
    assert len(labels) == len(aksharas(bare))          # alignment holds
    assert SvaraT.SVARITA in labels                    # the accent itself is captured
    assert restore_svara(bare, labels) != src          # exact glyph position is not

def test_avagraha_does_not_displace_mark():
    "Avagraha trails the syllable, so the mark belongs before it: 'मे॒ऽ' not 'मेऽ॒'."
    src = "मे॒ऽमृत"
    bare, labels = extract_svara(src)
    assert restore_svara(bare, labels) == src

def test_taittiriya_vs_unicode_reading():
    """U+0951 is svarita in Taittirīya printing, despite Unicode naming it UDATTA.

    Getting this backwards inverts the melody, so both readings are pinned explicitly.
    """
    assert extract_svara("म॑", tradition="taittiriya")[1][0] is SvaraT.SVARITA
    assert extract_svara("म॑", tradition="unicode")[1][0] is SvaraT.UDATTA
    # anudātta is unambiguous across both
    assert extract_svara("म॒", tradition="unicode")[1][0] is SvaraT.ANUDATTA

def test_dirgha_svarita_is_normalized_out():
    "U+1CDA is absent from the IndicF5 vocab; fold it to U+0951 to keep the tonal category."
    assert DIRGHA_SVARITA not in normalize("म" + DIRGHA_SVARITA)
    assert SVARITA in normalize("म" + DIRGHA_SVARITA)

def test_strip_svara_is_the_ablation_arm():
    assert strip_svara("मे॒ मय॑श्च") == "मे मयश्च"
    assert not has_svara(strip_svara("मे॒ मय॑श्च"))

def test_contour_and_tags():
    _, labels = extract_svara("मे॒ म॑")
    assert contour(labels)[0] == 0.0                 # anudātta low
    assert to_tags(labels) == ["<A>", "<S>"]          # udātta suppressed by default

# ── sandhi ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("src,expect_slp", [
    ("रामः गच्छति",    "rAmo gacCati"),      # utva: aḥ + voiced -> o
    ("देवः अत्र",      "devo 'tra"),         # utva: aḥ + a -> o + avagraha
    ("गुरुः आगच्छति",  "gurur AgacCati"),    # rutva: uḥ + vowel -> r
])
def test_visarga_sandhi(src, expect_slp):
    assert prep(src, sandhi=SandhiT.UTVA_RUTVA_LOPA, echo_final=False).slp == expect_slp

def test_satva_and_homorganic():
    assert satva("रामः च") == "रामश् च"
    assert homorganic_anusvara("सं गच्छति") == "सङ् गच्छति"

def test_echo_final_visarga():
    "Chant convention: clip-final ḥ takes an echo vowel. Also fixes vocoder garble at the tail."
    assert prep("ॐ शान्तिः").slp.endswith("SAntihi")
    assert visarga_echo_final("guruH") == "guruhu"

def test_plain_is_default():
    "PLAIN is the 4.6-MOS champion path — vagdhenu A/B'd resolved sandhi as worse."
    assert Frontend().sandhi is SandhiT.PLAIN
    assert prep("रामः गच्छति").model_text == "ರಾಮಃ ಗಚ್ಛತಿ"

def test_sandhi_refuses_accented_text():
    "saṃhitāpāṭha already carries its sandhi; re-resolving corrupts the received text."
    with pytest.raises(ValueError):
        prep("मे॒ मय॑श्च", sandhi=SandhiT.FULL)
    prep("मे॒ मय॑श्च", sandhi=SandhiT.FULL, allow_sandhi_on_accented=True)   # explicit opt-in

def test_sandhi_lookahead_skips_accent_marks():
    "An accent mark between visarga and its conditioning consonant must not block the rule."
    assert satva("रामः॑ च") == "रामश्॑ च"

# ── meter ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("slp,expect", [
    ("rAma",  "GL"),      # rā long -> G ; ma final short open -> L
    ("agni",  "GL"),      # a + conjunct gn -> G ; ni -> L
    ("rAmaH", "GG"),      # visarga closes the syllable -> G
    ("kaviM", "LG"),      # anusvāra closes -> G
])
def test_metrical_weight(slp, expect):
    assert weight_str(slp) == expect

def test_syllabify_counts():
    assert len(syllabify("gaRapatiM")) == 4       # ga-Ra-pa-tiM

# ── end-to-end on corpus-shaped input ────────────────────────────────────────────────
def test_prepped_invariants():
    p = prep("श-ञ्च॑ मे॒ मय॑श्च मे")
    assert p.accented
    assert p.n_aksharas == len(p.svara) == len(p.svara_str)
    assert len(p.weight_str) == p.n_aksharas
    assert p.model_text and not any(c in p.model_text for c in "।॥|")

def test_keep_svara_arm_toggles_marks_in_model_text():
    src = "मे॒ मय॑श्च"
    assert has_svara(prep(src, keep_svara=True).model_text)
    assert not has_svara(prep(src, keep_svara=False).model_text)
    # the label channel survives either way — that is what makes the A/B honest
    assert prep(src, keep_svara=False).svara_str == prep(src, keep_svara=True).svara_str
