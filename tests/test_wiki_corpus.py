import numpy as np
import pandas as pd
from uabr.data import wiki_corpus as W
from uabr.data import tokenizer as T


def test_normalize_title():
    assert W.normalize_title("henry_Feilden  (Conservative politician)") == "Henry Feilden (Conservative politician)"
    assert W.normalize_title(float("nan")) == ""


def test_clean_article_cuts_trailing_sections():
    t = "Lead text.\n\nCareer\nDid things.\n\n\n\nSee also\nOther\nReferences\nx"
    assert W.clean_article(t) == "Lead text.\n\nCareer\nDid things."


def test_disambiguation():
    assert W.is_disambiguation("Mercury", "Mercury may refer to:\n planet")
    assert not W.is_disambiguation("Mercury (planet)", "Mercury is the smallest planet.")


def test_stable_unit_deterministic():
    assert W.stable_unit("123", 1) == W.stable_unit("123", 1) != W.stable_unit("123", 2)


def test_assign_exposure_balanced():
    rng = np.random.default_rng(0)
    n = 4000
    p = pd.DataFrame({"id": range(n), "s_wiki_title": [f"T{i}" for i in range(n)],
                      "prop": rng.choice(["a", "b", "c"], n), "pop_bin": rng.integers(0, 5, n),
                      "subj_id": range(n)})
    cfg = W.CorpusConfig()
    e = W.assign_exposure(p, {f"T{i}" for i in range(n)}, cfg)
    shares = e["exposure_group"].value_counts(normalize=True)
    assert (shares - 0.25).abs().max() < 0.01
    assert dict(zip(e["exposure_group"], e["extra_copies"])) == {"control": 0, "low": 1, "medium": 4, "high": 16}
    # deterministic
    assert (W.assign_exposure(p, set(p["s_wiki_title"]), cfg)["exposure_group"] == e["exposure_group"]).all()


def test_select_fill_budget_and_excludes_val():
    m = pd.DataFrame({"id": range(100), "u": np.linspace(0, 0.3, 100), "n_chars": 10, "lm_val": [i % 10 == 0 for i in range(100)]})
    s = W.select_fill(m, 205)
    assert s["n_chars"].sum() >= 205 and not s["lm_val"].any() and s["u"].is_monotonic_increasing


def test_tokenizer_specials_and_roundtrip():
    texts = ["George Rankin was an Australian politician. Ünïcode — ok!"] * 50 + ["the cat sat on the mat"] * 50
    tok = T.train_bpe(texts, vocab_size=400, min_frequency=1)
    a = T.encode_batch(tok, [texts[0]])[0]
    assert tok.decode(a.tolist()) == texts[0]
    d = T.wrap_doc(a)
    assert d[0] == T.BOS_ID and d[-1] == T.EOS_ID and d.dtype == np.uint16
