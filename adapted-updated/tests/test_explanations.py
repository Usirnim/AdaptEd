"""The explanation shown to the learner must be specific, accurate and grounded in the lesson."""
import scoring
from conftest import ALL, CORPUS

def ex(cid, text):
    n, l, ref, terms, _ = ALL[cid]; r = scoring.score(text, ref, terms, CORPUS)
    return r, scoring.explain(r, n, l, ref)

PARTIAL = "A list is an ordered collection of items and you reach each item by its index, which starts at zero in Python."

def test_structure_complete():
    r, x = ex("py_col", PARTIAL)
    assert set(x) == {"headline", "summary", "strengths", "gaps", "model_answer", "next_step"}
    assert x["model_answer"] == ALL["py_col"][2]

def test_gaps_teach_from_the_lesson():
    r, x = ex("py_col", PARTIAL)
    assert x["gaps"], "partial answer must report gaps"
    assert [g["term"] for g in x["gaps"]] == r["missing"]
    for g in x["gaps"]:
        assert g["from_lesson"], f"no teaching sentence for {g['term']}"
        assert scoring._has(g["term"], scoring._toks(g["from_lesson"]))
        assert g["from_lesson"] in ALL["py_col"][1] or g["from_lesson"] in ALL["py_col"][2]

def test_strengths_match_matched_terms():
    r, x = ex("py_col", PARTIAL); assert [s["term"] for s in x["strengths"]] == r["matched"]

def test_pass_has_no_negative_language():
    r, x = ex("st_spread", ALL["st_spread"][2])
    assert r["passed"] and x["headline"].startswith("Mastered") and not x["gaps"]
    assert "re-read" not in x["next_step"].lower()

def test_fail_tells_what_to_do():
    r, x = ex("py_col", "Lists are things in python.")
    assert not r["passed"] and "too short" in x["summary"]

def test_keyword_dump_gets_specific_message():
    r, x = ex("py_col", " ".join(ALL["py_col"][3] * 2))
    assert "sentences" in x["headline"].lower() and "keyword" in x["summary"].lower()

def test_off_topic_is_told_to_reread():
    r, x = ex("py_col", "I really enjoy football and pizza with my friends on weekends, it is a lot of fun")
    assert x["headline"].startswith("Not there yet") and "pass mark 55%" in x["summary"]

def test_pass_mark_and_score_quoted_correctly():
    r, x = ex("py_col", PARTIAL); assert f"{round(r['score']*100)}%" in x["summary"]

def test_every_term_of_every_concept_has_a_teaching_sentence():
    for cid, (n, l, ref, terms, _) in ALL.items():
        r = scoring.score("x", ref, terms, CORPUS); x = scoring.explain(r, n, l, ref)
        assert all(g["from_lesson"] for g in x["gaps"]), cid

def test_legacy_feedback_string():
    r, _ = ex("py_col", PARTIAL); assert "Try to mention" in scoring.feedback(r)

def test_partly_right_answer_is_not_told_it_is_off_topic():
    r, x = ex("ml_fit", "Overfitting is when a model learns the training data too well and then does badly on new data, so we keep some data aside to check it.")
    assert not r["passed"] and x["headline"].startswith("Right direction") and "different words" in x["summary"]
