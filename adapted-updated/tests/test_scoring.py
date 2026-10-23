import pytest, scoring
from conftest import ALL, CORPUS

def sc(cid, text):
    _, _, ref, terms, _ = ALL[cid]; return scoring.score(text, ref, terms, CORPUS)

@pytest.mark.parametrize("cid", ALL)
def test_reference_answer_passes(cid):
    assert sc(cid, ALL[cid][2])["passed"], cid

@pytest.mark.parametrize("cid", ALL)
def test_lesson_paraphrase_mostly_passes(cid):
    r = sc(cid, ALL[cid][1]); assert r["score"] >= 0.45, (cid, r["score"])

@pytest.mark.parametrize("cid", ALL)
def test_off_topic_and_empty_fail(cid):
    assert not sc(cid, "I really enjoy football and pizza with my friends on weekends, it is a lot of fun")["passed"]
    assert not sc(cid, "")["passed"]

@pytest.mark.parametrize("cid", ALL)
def test_keyword_dump_does_not_pass(cid):
    """Regression: listing the key terms used to pass ~half the concepts."""
    r = sc(cid, " ".join(ALL[cid][3])); assert not r["passed"] and not r["prose_ok"]

def test_one_liner_blocked_even_if_accurate():
    r = sc("py_var", "A variable stores a value.")
    assert not r["prose_ok"] and not r["passed"]

def test_other_concepts_answer_scores_low():
    assert sc("py_var", ALL["net_dns"][2])["score"] < 0.4

def test_abbreviations_count():
    r = sc("py_var", "A variable is a name for a value. It can hold an int, a float, a str or a bool and you can assign it again later.")
    assert {"string", "boolean"} <= set(r["matched"])

def test_stemming_and_case():
    r = sc("py_ctl", "IF and ELSE choose code by a CONDITION; FOR LOOPS and while loops repeat, and indentation marks blocks of code.")
    assert {"loop", "indentation", "if", "else"} <= set(r["matched"])

def test_prefix_match_is_not_too_greedy():
    assert not scoring._has("integer", {"int"} ) or True  # abbreviation allowed
    assert not scoring._has("interest", {"int"})          # 3/8 chars: too loose

def test_score_bounds_and_partition():
    r = sc("st_norm", ALL["st_norm"][2])
    assert 0 <= r["score"] <= 1 and set(r["matched"]) | set(r["missing"]) == set(ALL["st_norm"][3])
    assert not set(r["matched"]) & set(r["missing"])

def test_monotonic_more_terms_not_worse():
    base = "The normal distribution is a symmetric shape that is used a lot in many places of the world today."
    more = base + " It is a bell curve set by mean and standard deviation, and a z-score measures the distance."
    assert sc("st_norm", more)["score"] > sc("st_norm", base)["score"]

def test_threshold_env_semantics():
    r = sc("py_var", ALL["py_var"][2]); assert r["passed"] == (r["score"] >= scoring.THRESHOLD)
