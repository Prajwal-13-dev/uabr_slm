import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mock_popqa import make
from uabr.data import popqa as P


def build(scale=1.0):
    cfg = P.SplitConfig()
    raw = make(scale)
    df = P.prepare(raw)
    df = P.add_groups(df)
    df, _ = P.add_pop_bins(df, cfg.n_pop_bins)
    df = P.assign_splits(df, cfg)
    df = P.select_pilot(df, cfg)
    return df, cfg, raw


def test_end_to_end_checks_pass():
    df, cfg, raw = build()
    checks = {c["name"]: c for c in P.split_checks(df, cfg, expected_rows=len(raw))}
    for name in ["no_group_leak", "no_subject_leak", "ood_relations_isolated", "pilot_size",
                 "pilot_in_val_only", "pilot_not_in_test", "gold_aliases_nonempty", "overall_fractions_within_2pp"]:
        assert checks[name]["passed"], (name, checks[name]["detail"])


def test_homonyms_grouped_and_flagged():
    df, _, _ = build()
    h = df[df["question"] == "What is John Smith's occupation?"]
    assert h["ambiguous_question"].all()
    assert h["group_id"].nunique() == 1


def test_deterministic():
    a, _, _ = build(); b, _, _ = build()
    assert (a["split"] == b["split"]).all() and (a["is_pilot"] == b["is_pilot"]).all()


def test_relation_prior():
    df, cfg, _ = build()
    tab, hit = P.relation_majority(df, cfg)
    sport = tab.set_index("relation").loc["sport"]
    assert sport["prior_answer"] in ("association football", "basketball", "baseball")
    assert sport["acc_train"] > 0.15
    assert tab.set_index("relation").loc["religion", "prior_fit_on"] == "ood(oracle)"
    assert set(hit.unique()) <= {0.0, 1.0}


def test_fingerprint_stable():
    raw = make(0.2)
    assert P.fingerprint(raw) == P.fingerprint(raw.sample(frac=1, random_state=3))
