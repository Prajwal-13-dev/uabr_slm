import numpy as np
from uabr.retrieval.bm25 import BM25Config, BM25Index, window_starts, window_text
from uabr.retrieval.eval import answer_in_window, subject_in_window


def test_window_starts_cover_end():
    assert window_starts(10, 60, 30) == [0]
    assert window_starts(60, 60, 30) == [0]
    s = window_starts(100, 60, 30)
    assert s == [0, 30, 40] and s[-1] + 60 == 100
    assert window_starts(120, 60, 30) == [0, 30, 60]


def test_window_text_title():
    cfg = BM25Config()
    assert window_text("Paris", "a b c".split(), 1, cfg) == "Paris: b c"


def _corpus():
    titles = ["George Rankin", "John Mayne", "Filler One", "Filler Two"]
    texts = ["He was an Australian politician and soldier. " * 5,
             "He was a Scottish journalist and poet. " * 5,
             "Bananas are yellow fruit grown in tropical places. " * 20,
             "The river flows through the valley to the sea. " * 20]
    return titles, texts


def test_bm25_retrieves_title_matched_window():
    titles, texts = _corpus()
    cfg = BM25Config(window_words=12, stride_words=6, n_features=2**18, articles_per_task=2)
    ix = BM25Index(cfg).build(titles, texts, n_jobs=1, log=lambda *a: None)
    ids, sc = ix.search("What is George Rankin's occupation?", k=3)
    assert len(ids) == 3 and np.all(np.diff(sc) <= 1e-6)
    assert ix.doc_article[ids[0]] == 0
    w = ix.window(ids[0], titles, texts)
    assert w.startswith("George Rankin:")
    ids2, _ = ix.search("journalist poet", k=1)
    assert ix.doc_article[ids2[0]] == 1
    # buffer is reset between queries -> identical results when repeated
    a, b = ix.search("George Rankin politician", 5), ix.search("George Rankin politician", 5)
    assert (a[0] == b[0]).all() and np.allclose(a[1], b[1])


def test_bm25_parallel_equals_serial(tmp_path):
    titles, texts = _corpus()
    cfg = BM25Config(window_words=12, stride_words=6, n_features=2**18, articles_per_task=1)
    s = BM25Index(cfg).build(titles, texts, n_jobs=1, log=lambda *a: None)
    p = BM25Index(cfg).build(titles, texts, n_jobs=2, log=lambda *a: None)
    assert (s.W != p.W).nnz == 0 and (s.doc_start == p.doc_start).all()
    p.save(str(tmp_path)); l = BM25Index.load(str(tmp_path), cfg)
    assert (l.search("river valley", 3)[0] == p.search("river valley", 3)[0]).all()


def test_eval_flags():
    assert answer_in_window("George Rankin: he was a politician.", ["politician", "pol"])
    assert not answer_in_window("George Rankin: political party", ["pol"])
    assert subject_in_window("x", "George Rankin", "George Rankin", [])
    assert subject_in_window("Other: George James Rankin was", "Other", "George Rankin", ["George James Rankin"])
