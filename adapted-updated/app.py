import os, sys, json, sqlite3, datetime as dt
from flask import Flask, request, session, jsonify, send_from_directory, g
from werkzeug.security import generate_password_hash, check_password_hash
import content, graph, scoring, scheduler, llm
app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")
DB = os.environ.get("DB_PATH", "adapted.db")

def db():
    if "db" not in g: g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db
@app.teardown_appcontext
def close(_): d = g.pop("db", None); d and d.close()

def init():
    c = sqlite3.connect(DB)
    if c.execute("SELECT 1 FROM sqlite_master WHERE name='concepts'").fetchone() and "lesson" not in [r[1] for r in c.execute("PRAGMA table_info(concepts)")]:
        for t in ("concepts", "edges", "paths", "mastery", "scores"): c.execute(f"DROP TABLE {t}")   # upgrade from v1 schema
    c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE, pw TEXT);
    CREATE TABLE IF NOT EXISTS concepts(id TEXT PRIMARY KEY, subject TEXT, name TEXT, ref TEXT, terms TEXT, ord INT, lesson TEXT);
    CREATE TABLE IF NOT EXISTS studied(uid INT, cid TEXT, PRIMARY KEY(uid,cid));
    CREATE TABLE IF NOT EXISTS edges(src TEXT, dst TEXT);
    CREATE TABLE IF NOT EXISTS paths(uid INT PRIMARY KEY, goal TEXT, subject TEXT, path TEXT);
    CREATE TABLE IF NOT EXISTS mastery(uid INT, cid TEXT, status TEXT, idx INT, due TEXT, PRIMARY KEY(uid,cid));
    CREATE TABLE IF NOT EXISTS scores(uid INT, cid TEXT, score REAL, ts TEXT);
    CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY, uid INT, goal TEXT, concepts TEXT, status TEXT DEFAULT 'pending');""")
    if not c.execute("SELECT 1 FROM concepts").fetchone():
        i = 0
        for sub, (desc, cs) in content.SUBJECTS.items():
            for k, (n, l, r, t, pre) in cs.items():
                c.execute("INSERT INTO concepts VALUES(?,?,?,?,?,?,?)", (k, sub, n, r, json.dumps(t), i, l)); i += 1
                c.executemany("INSERT INTO edges VALUES(?,?)", [(p, k) for p in pre])
    c.commit(); c.close()

def load():
    cs = [dict(r, terms=json.loads(r["terms"])) for r in db().execute("SELECT * FROM concepts ORDER BY ord")]
    return graph.build_graph(cs, [tuple(r) for r in db().execute("SELECT src,dst FROM edges")])

def uid():
    if "uid" not in session: raise PermissionError
    return session["uid"]
@app.errorhandler(PermissionError)
def noauth(_): return jsonify(error="Log in first"), 401
today = lambda: dt.date.today()
def hist(u, cid): return [r[0] for r in db().execute("SELECT score FROM scores WHERE uid=? AND cid=? ORDER BY rowid", (u, cid))]

@app.route("/")
def index(): return send_from_directory("static", "index.html")

@app.post("/api/auth/<mode>")
def auth(mode):
    d = request.json; n, p = d.get("name", "").strip(), d.get("password", "")
    if len(n) < 3 or len(p) < 4: return jsonify(error="Username 3+ chars, password 4+ chars"), 400
    if mode == "register":
        try: db().execute("INSERT INTO users(name,pw) VALUES(?,?)", (n, generate_password_hash(p))); db().commit()
        except sqlite3.IntegrityError: return jsonify(error="Username taken"), 400
    u = db().execute("SELECT * FROM users WHERE name=?", (n,)).fetchone()
    if not u or not check_password_hash(u["pw"], p): return jsonify(error="Wrong username or password"), 401
    session["uid"] = u["id"]; session["name"] = n
    return jsonify(name=n)
@app.post("/api/logout")
def logout(): session.clear(); return jsonify(ok=1)
@app.get("/api/me")
def me(): return jsonify(name=session.get("name"))

@app.post("/api/goal")
def goal():
    u = uid(); text = request.json.get("goal", "").strip()
    if not text: return jsonify(error="Enter a goal"), 400
    G = load(); m = graph.match_goal(text, G, {s: v[0] for s, v in content.SUBJECTS.items()})
    if not m:   # REQ-1.5: decided locally, LLM only assists
        props = llm.propose_concepts(text)
        if props:
            db().execute("INSERT INTO proposals(uid,goal,concepts) VALUES(?,?,?)", (u, text, json.dumps(props))); db().commit()
            return jsonify(error="Topic not yet supported. AI-suggested concepts were saved for developer review."), 404
        return jsonify(error="Topic not yet supported. Pick one of the listed subjects."), 404
    sub, targets, sim = m; path = graph.ordered_path(G, targets)
    db().execute("REPLACE INTO paths VALUES(?,?,?,?)", (u, text, sub, json.dumps(path))); db().commit()
    return jsonify(path_for(u))

def path_for(u):
    p = db().execute("SELECT * FROM paths WHERE uid=?", (u,)).fetchone()
    if not p: return dict(path=[])
    G = load(); ids = json.loads(p["path"]); ms = {r["cid"]: r["status"] for r in db().execute("SELECT * FROM mastery WHERE uid=?", (u,))}
    out, cur = [], False
    for i in ids:
        if ms.get(i) == "mastered": st = "mastered"
        elif any(ms.get(q) != "mastered" for q in G.predecessors(i) if q in ids): st = "locked"
        else: st = "in-progress"
        out.append(dict(id=i, name=G.nodes[i]["name"], status=st, review=ms.get(i) == "review"))
    sd = {r[0] for r in db().execute("SELECT cid FROM studied WHERE uid=?", (u,))}
    for c in out: c["studied"] = c["id"] in sd
    nxt = next((c for c in out if c["status"] == "in-progress"), None)
    if nxt: nxt["current"] = True
    return dict(goal=p["goal"], subject=p["subject"], path=out)
@app.get("/api/path")
def path(): return jsonify(path_for(uid()))

@app.post("/api/explain")
def explain():
    u = uid(); cid = request.json.get("concept"); text = request.json.get("text", "").strip()
    G = load()
    if cid not in G or len(text) < 5: return jsonify(error="Write a short explanation first"), 400
    pr = path_for(u); st = {c["id"]: c["status"] for c in pr["path"]}
    if st.get(cid) == "locked" or cid not in st: return jsonify(error="Master the prerequisites first"), 403
    if not next(c for c in pr["path"] if c["id"] == cid)["studied"]: return jsonify(error="Study the lesson first"), 403
    n = G.nodes[cid]; corpus = [G.nodes[x]["ref"] for x in G]
    r = scoring.score(text, n["ref"], n["terms"], corpus)
    db().execute("INSERT INTO scores VALUES(?,?,?,?)", (u, cid, r["score"], dt.datetime.now().isoformat())); db().commit()
    prev = db().execute("SELECT * FROM mastery WHERE uid=? AND cid=?", (u, cid)).fetchone()
    if r["passed"]:     # REQ-2.5 / 3.1
        idx = prev["idx"] if prev and prev["status"] == "mastered" else 0
        days = scheduler.next_interval(idx, hist(u, cid), scoring.THRESHOLD)
        s = "mastered"
    else:               # REQ-2.6 / 3.4
        idx, days, s = 0, scheduler.LOW_D, "review"
    due = (today() + dt.timedelta(days=days)).isoformat()
    db().execute("REPLACE INTO mastery VALUES(?,?,?,?,?)", (u, cid, s, idx, due)); db().commit()
    fb = scoring.feedback(r); ex = scoring.explain(r, n["name"], n["lesson"], n["ref"])
    ai = llm.tutor_feedback(r, n["name"], n["lesson"], text)   # REQ-2.8: assistive only, score is local
    r.update(feedback=fb, explanation=ex, tutor_note=ai, next_review=due, status=s,
             feedback_source="AI tutor note (assistive; score computed locally)" if ai else "built from the lesson")
    return jsonify(r)

@app.get("/api/concept/<cid>")
def concept(cid):
    u = uid(); n = load().nodes[cid]
    return jsonify(name=n["name"], lesson=n["lesson"], subject=n["subject"], hint=f"Explain '{n['name']}' in your own words, without looking at the lesson.")
@app.post("/api/studied")
def studied():
    u = uid(); db().execute("INSERT OR IGNORE INTO studied VALUES(?,?)", (u, request.json.get("concept"))); db().commit(); return jsonify(ok=1)
@app.get("/api/subjects")
def subjects(): return jsonify([dict(name=s, count=len(v[1])) for s, v in content.SUBJECTS.items()])

@app.get("/api/dashboard")
def dash():
    u = uid(); G = load(); rows = db().execute("SELECT * FROM mastery WHERE uid=? ORDER BY due", (u,)).fetchall()
    f = lambda r: dict(id=r["cid"], name=G.nodes[r["cid"]]["name"], due=r["due"], status=r["status"])
    t = today().isoformat()
    return jsonify(due=[f(r) for r in rows if r["due"] <= t], upcoming=[f(r) for r in rows if r["due"] > t],
                   mastered=sum(r["status"] == "mastered" for r in rows))

@app.post("/api/review")
def review():
    u = uid(); cid = request.json.get("concept")
    m = db().execute("SELECT * FROM mastery WHERE uid=? AND cid=?", (u, cid)).fetchone()
    if not m: return jsonify(error="Not scheduled"), 404
    idx = min(m["idx"] + 1, len(scheduler.BASE) - 1) if m["status"] == "mastered" else m["idx"]   # REQ-3.6
    days = scheduler.next_interval(idx, hist(u, cid), scoring.THRESHOLD) if m["status"] == "mastered" else 1
    due = (today() + dt.timedelta(days=days)).isoformat()
    db().execute("UPDATE mastery SET idx=?, due=? WHERE uid=? AND cid=?", (idx, due, u, cid)); db().commit()
    return jsonify(next_review=due)

init()
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "proposals":    # developer review of LLM suggestions
        for r in sqlite3.connect(DB).execute("SELECT id,goal,concepts,status FROM proposals"): print(r)
    else: app.run(debug=True, port=int(os.environ.get("PORT", 5000)))