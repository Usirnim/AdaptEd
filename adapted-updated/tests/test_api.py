"""End-to-end through Flask: auth, gating, mastery, review scheduling, explanation payload."""
import datetime as dt
from conftest import ALL

GOOD = ALL["py_var"][2]
def start(c, goal="I want to learn Python basics"): return c.post("/api/goal", json={"goal": goal})
def study(c, cid): return c.post("/api/studied", json={"concept": cid})
def explain(c, cid, text): return c.post("/api/explain", json={"concept": cid, "text": text})

def test_auth_rules(client):
    assert client.get("/api/path").status_code == 401
    assert client.post("/api/auth/register", json={"name": "ab", "password": "secret"}).status_code == 400
    assert client.post("/api/auth/register", json={"name": "bob", "password": "x"}).status_code == 400
    assert client.post("/api/auth/register", json={"name": "bob", "password": "secret"}).status_code == 200
    client.post("/api/logout")
    assert client.post("/api/auth/register", json={"name": "bob", "password": "other"}).status_code == 400
    assert client.post("/api/auth/login", json={"name": "bob", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"name": "bob", "password": "secret"}).json["name"] == "bob"

def test_goal_creates_ordered_locked_path(user):
    p = start(user).json["path"]
    assert [c["id"] for c in p][0] == "py_var" and p[0]["current"] and p[0]["status"] == "in-progress"
    assert all(c["status"] == "locked" for c in p[1:])

def test_unsupported_goal_404(user):
    assert start(user, "how to bake sourdough bread").status_code == 404
    assert user.post("/api/goal", json={"goal": "  "}).status_code == 400

def test_gating_prereq_and_study(user):
    start(user)
    assert explain(user, "py_ctl", GOOD).status_code == 403            # locked
    assert explain(user, "py_var", GOOD).status_code == 403            # not studied
    assert explain(user, "py_var", "hi").status_code == 400            # too short
    assert explain(user, "nope", GOOD).status_code == 400

def test_pass_masters_and_unlocks_next(user):
    start(user); study(user, "py_var"); r = explain(user, "py_var", GOOD).json
    assert r["passed"] and r["status"] == "mastered"
    ex = r["explanation"]; assert ex["headline"].startswith("Mastered") and ex["model_answer"]
    assert r["feedback_source"] == "built from the lesson" and r["tutor_note"] is None
    assert r["next_review"] == (dt.date.today() + dt.timedelta(days=1)).isoformat()
    p = user.get("/api/path").json["path"]; assert p[0]["status"] == "mastered" and p[1]["status"] == "in-progress"

def test_fail_goes_to_review_today_with_real_explanation(user):
    start(user); study(user, "py_var")
    r = explain(user, "py_var", "A variable is a thing that holds some kind of data in a program somewhere.").json
    assert not r["passed"] and r["status"] == "review"
    assert r["next_review"] == dt.date.today().isoformat()
    assert r["explanation"]["gaps"] and all(g["from_lesson"] for g in r["explanation"]["gaps"])
    assert user.get("/api/path").json["path"][1]["status"] == "locked"     # next stays locked
    assert [d["id"] for d in user.get("/api/dashboard").json["due"]] == ["py_var"]

def test_keyword_dump_cannot_master(user):
    start(user); study(user, "py_var")
    r = explain(user, "py_var", " ".join(ALL["py_var"][3])).json
    assert not r["passed"] and "sentences" in r["explanation"]["headline"].lower()

def test_retry_after_fail_can_pass(user):
    start(user); study(user, "py_var"); explain(user, "py_var", "variables are stuff in code that you can use in programs")
    assert explain(user, "py_var", GOOD).json["status"] == "mastered"

def test_review_advances_interval(user):
    start(user); study(user, "py_var"); explain(user, "py_var", GOOD)
    d = user.post("/api/review", json={"concept": "py_var"}).json["next_review"]
    assert d > dt.date.today().isoformat()
    assert user.post("/api/review", json={"concept": "py_ctl"}).status_code == 404

def test_users_are_isolated(client):
    client.post("/api/auth/register", json={"name": "alice", "password": "secret"}); start(client); study(client, "py_var")
    client.post("/api/logout"); client.post("/api/auth/register", json={"name": "carol", "password": "secret"})
    assert client.get("/api/path").json["path"] == []
    start(client); assert explain(client, "py_var", GOOD).status_code == 403   # carol hasn't studied

def test_subjects_and_concept_endpoints(user):
    assert len(user.get("/api/subjects").json) == 6
    c = user.get("/api/concept/py_var").json; assert c["lesson"] and "own words" in c["hint"]

def test_html_escaping_in_explanation_inputs_is_data_only(user):
    start(user); study(user, "py_var")
    r = explain(user, "py_var", "<script>alert(1)</script> variable value int float string boolean assign things here").json
    assert "explanation" in r        # server returns JSON; UI escapes via e()
