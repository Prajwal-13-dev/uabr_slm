import numpy as np
import pandas as pd
import torch
from uabr.data.fact_sentences import ReinjectConfig, assign_doses, fact_sentences_for, injection_docs, split_sentences
from uabr.eval.probes import answer_span, candidate_sets, score_continuations
from uabr.model.llama import LlamaLM, ModelConfig


def test_split_sentences_keeps_initials_and_abbrev():
    t = "Hamlet is a 1948 film directed by Laurence Olivier. It won awards.\nJohn H. Smith lived in St. Louis. He died."
    s = split_sentences(t)
    assert s == ["Hamlet is a 1948 film directed by Laurence Olivier.", "It won awards.", "John H. Smith lived in St. Louis.", "He died."]


def test_fact_sentences_need_subject_and_answer():
    cfg = ReinjectConfig()
    texts = ["Hamlet is a 1948 film directed by Laurence Olivier. Olivier also starred. Hamlet won four Oscars."]
    out = fact_sentences_for(texts, ["Hamlet"], ["Laurence Olivier", "Olivier"], cfg)
    assert out == ["Hamlet is a 1948 film directed by Laurence Olivier."]


def test_assign_doses_balanced_and_deterministic():
    rng = np.random.default_rng(0)
    s = pd.DataFrame({"title_n": [f"T{i}" for i in range(6000)], "primary_prop": rng.choice(list("abc"), 6000),
                      "pop_bin": rng.integers(0, 5, 6000)})
    cfg = ReinjectConfig()
    a = assign_doses(s, cfg)
    assert len(a) == 4000 and a["title_n"].is_unique
    shares = a["dose_group"].value_counts(normalize=True)
    assert (shares - 0.25).abs().max() < 0.01
    assert (assign_doses(s, cfg)["title_n"] == a["title_n"]).all()


def test_injection_docs_cycle():
    f = pd.DataFrame({"dose": [0, 3, 2], "sentences": [["x"], ["a", "b"], ["c"]]})
    assert injection_docs(f) == ["a", "b", "a", "c", "c"]


def test_score_continuations_matches_unbatched():
    torch.manual_seed(0)
    m = LlamaLM(ModelConfig(vocab_size=50, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=32)).eval()
    p = [1, 5, 6]
    conts = [[7], [8, 9, 10], [11, 12]]
    sums, means = score_continuations(m, p, conts, "cpu")
    for c, s in zip(conts, sums):
        x = torch.tensor([p + c]); lp = torch.log_softmax(m(x)[0][0].float(), -1)
        ref = sum(float(lp[len(p) - 1 + j, t]) for j, t in enumerate(c))
        assert abs(ref - s) < 1e-4


def test_candidates_and_answer_span():
    c = candidate_sets({"r": ["A", "B", "C", "Gold", "gold "]}, {1: ("Gold", ["Gold"])}, {1: "r"}, 2, seed=0)
    assert c[1][0] == "Gold" and len(c[1]) == 3 and "gold " not in c[1][1:]
    assert answer_span("Hamlet was directed by Laurence Olivier.", ["Olivier", "Laurence Olivier"]) == (23, "Laurence Olivier")
    assert answer_span("Olivier directed it.", ["Olivier"]) is None   # answer at position 0 -> no prefix


def test_assign_doses_exact_balance_small_n():
    rng = np.random.default_rng(1)
    s = pd.DataFrame({"title_n": [f"T{i}" for i in range(4448)], "primary_prop": rng.choice(list("abcdefghijklmnop"), 4448),
                      "pop_bin": rng.integers(0, 5, 4448)})
    a = assign_doses(s, ReinjectConfig(n_subjects=400))
    c = a["dose_group"].value_counts()
    assert c.max() - c.min() <= 1
    per = pd.crosstab([a["primary_prop"], a["pop_bin"]], a["dose_group"])
    assert ((per.max(axis=1) - per.min(axis=1)) <= 1).all()
