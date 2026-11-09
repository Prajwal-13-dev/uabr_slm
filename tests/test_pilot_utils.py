import numpy as np
import pandas as pd
import torch
from uabr.eval import prompts as PR
from uabr.eval.generate import answer_mean_logprob, fit_evidence, generate, seed_for
from uabr.eval.stats import clustered_bootstrap_diff, clustered_bootstrap_mean
from uabr.model.llama import LlamaLM, ModelConfig

RELS = ["occupation", "place of birth", "genre", "father", "country", "producer", "director", "capital of",
        "screenwriter", "composer", "color", "religion", "sport", "author", "mother", "capital"]


def test_templates_cover_popqa_relations():
    assert set(RELS) == set(PR.CLOZE_TEMPLATES)
    assert PR.cloze("director", "Hamlet") == "Hamlet was directed by"


def test_demos_distinct_and_train_only():
    tr = pd.DataFrame({"prop": ["sport"] * 6, "subj": list("abcdef"), "in_main": True,
                       "answer": ["football", "football", "Football", "tennis", "golf", "chess"]})
    d = PR.pick_demos(tr, 3, seed=1)["sport"]
    assert len(d) == 3 and len({a.lower() for _, a in d}) == 3
    p = PR.closed_prompt("sport", "Zed", {"sport": d})
    assert p.endswith("Zed plays the sport of") and p.count("\n") == 3
    e = PR.evidence_prompt(["W1: x", "W2: y"], "sport", "Zed", {"sport": d})
    assert e.startswith("W1: x\nW2: y\n\n")


def test_clean_generation():
    assert PR.clean_generation(" Steven Spielberg.\nThe") == "Steven Spielberg"
    assert PR.clean_generation(" Paris, France") == "Paris"
    assert PR.clean_generation(" rock music (genre)") == "rock music"
    assert PR.clean_generation(" St.Louis") == "St.Louis"


class _Tok:  # minimal tokenizer stub: 1 char = 1 id
    class _E:
        def __init__(self, ids): self.ids = ids
    def encode(self, t, add_special_tokens=False): return self._E([3 + (ord(c) % 90) for c in t])


def test_fit_evidence_drops_windows():
    tok = _Tok()
    build = lambda ws: "".join(w + "\n" for w in ws) + "q"
    ids, k = fit_evidence(tok, build, ["a" * 10, "b" * 10, "c" * 10], max_len=30)
    assert k == 2 and len(ids) <= 30
    ids, k = fit_evidence(tok, build, ["a" * 10] * 3, max_len=40)
    assert k == 3


def test_generate_and_logprob_tiny():
    torch.manual_seed(0)
    m = LlamaLM(ModelConfig(vocab_size=100, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=64)).eval()
    out = generate(m, [1, 5, 6], n=4, max_new=5, temperature=0.7, seed=seed_for("q", 1), device="cpu")
    out2 = generate(m, [1, 5, 6], n=4, max_new=5, temperature=0.7, seed=seed_for("q", 1), device="cpu")
    assert out == out2 and len(out) == 4 and all(len(o) <= 5 for o in out)
    g = generate(m, [1, 5, 6], n=1, max_new=5, temperature=0, seed=0, device="cpu")
    lp = answer_mean_logprob(m, [1, 5, 6], [7, 8], "cpu")
    assert lp < 0 and len(g) == 1


def test_bootstrap():
    v = np.r_[np.ones(50), np.zeros(50)]
    r = clustered_bootstrap_mean(v, np.arange(100))
    assert r["lo"] < 0.5 < r["hi"]
    d = clustered_bootstrap_diff(v, np.arange(100), np.r_[np.ones(50, bool), np.zeros(50, bool)], np.r_[np.zeros(50, bool), np.ones(50, bool)])
    assert d["diff"] == 1.0 and d["lo"] == 1.0
