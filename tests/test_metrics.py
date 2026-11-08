from uabr.eval.metrics import alias_contains, alias_em, alias_f1, dedupe_aliases, normalize_answer


def test_normalize():
    assert normalize_answer("The Beyoncé!") == "beyonce"
    assert normalize_answer("  U.S.A. ") == normalize_answer("u.s.a")  # SQuAD drops lone "a"


def test_alias_em_f1():
    g = ["association football", "soccer"]
    assert alias_em("Soccer", g) == 1.0
    assert alias_em("", g) == 0.0
    assert alias_f1("football", g) == 2 * (1 * 0.5) / 1.5
    assert alias_em("politician", ["politician", "pol"]) == 1.0


def test_contains_ignores_short_aliases():
    assert alias_contains("he was a political figure", ["pol"]) == 0.0
    assert alias_contains("he was a politician.", ["politician"]) == 1.0
    assert alias_contains("politicians", ["politician"]) == 0.0  # whole-word only


def test_dedupe():
    assert dedupe_aliases(["Jazz", "jazz", "", None, "the jazz"]) == ["Jazz"]
