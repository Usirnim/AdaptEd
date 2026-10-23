"""Embedding/NLP scoring module (REQ-2.x). TF-IDF cosine by default; set USE_SBERT=1 for sentence-transformers."""
import os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
W_SIM = float(os.environ.get("W_SIM", 0.6))      # weight of similarity; coverage gets 1-W_SIM
SIM_CAP = 0.5                                     # TF-IDF cosine of a good answer rarely exceeds ~0.5
THRESHOLD = float(os.environ.get("THRESHOLD", 0.55))
_sb = None

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
MIN_WORDS = 12          # an explanation shorter than this cannot show understanding
MIN_STOP_RATIO = 0.10   # real prose is full of "is/the/a/of"; a keyword dump is not

_IRREG = {"children": "child", "vertices": "vertex", "indices": "index", "indexes": "index"}
def _stem(w):
    w = _IRREG.get(w, w).replace("z", "s")            # generalize == generalise
    for _ in range(2):                                 # iterate so 'headings' and 'heading' meet at 'head'
        if len(w) > 4: w = re.sub(r"(ing|ed|es|s)$", "", w)
    return w
def _words(t): return re.findall(r"[a-z0-9]+", t.lower())
def _toks(t): return {_stem(w) for w in _words(t)}

def _has(term, tk):
    """Does the learner's token set contain `term`? Accepts abbreviations of single-word terms
    ('str' for 'string', 'bool' for 'boolean') so correct answers are not marked down."""
    tt = _toks(term)
    if tt <= tk: return True
    if len(tt) == 1:
        (w,) = tt
        return any((len(k) >= 3 and len(k) * 2 >= len(w) and w.startswith(k))        # str -> string
                   or (len(w) >= 4 and 1 <= len(k) - len(w) <= 3 and k.startswith(w))  # fast -> faster
                   for k in tk)
    return False

def prose_check(text, terms=()):
    """Returns (ok, reason). Blocks keyword-dumping / one-liners from counting as an explanation.
    Words that are themselves key terms ('if', 'for', 'with') don't count as connecting words."""
    w = _words(text)
    if len(w) < MIN_WORDS:
        return False, f"too short ({len(w)} words; aim for at least {MIN_WORDS})"
    tw = {x for t in terms for x in _words(t)}
    ratio = sum(x in ENGLISH_STOP_WORDS and x not in tw for x in w) / len(w)
    if ratio < MIN_STOP_RATIO:
        return False, "reads like a list of keywords rather than sentences"
    return True, ""

def similarity(text, ref, corpus):
    global _sb
    if os.environ.get("USE_SBERT"):
        from sentence_transformers import SentenceTransformer
        _sb = _sb or SentenceTransformer("all-MiniLM-L6-v2")
        e = _sb.encode([text, ref]); return max(0.0, float(cosine_similarity([e[0]], [e[1]])[0][0])), 1.0
    v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit(corpus + [text, ref])
    return float(cosine_similarity(v.transform([text]), v.transform([ref]))[0][0]), SIM_CAP

def score(text, ref, terms, corpus):
    sim, cap = similarity(text, ref, corpus)
    tk = _toks(text)
    hit = [t for t in terms if _has(t, tk)]
    cov = len(hit) / len(terms) if terms else 0
    total = W_SIM * min(1, sim / cap) + (1 - W_SIM) * cov
    ok, why = prose_check(text, terms)
    return dict(score=round(total, 3), similarity=round(sim, 3), coverage=round(cov, 3),
                matched=hit, missing=[t for t in terms if t not in hit],
                prose_ok=ok, prose_issue=why, passed=ok and total >= THRESHOLD)

def feedback(r):
    """Short one-paragraph fallback (kept for compatibility)."""
    if r["passed"]: m = "Good explanation - concept marked as mastered."
    else: m = "Not quite there yet - this concept was added to your review queue."
    if r["missing"]: m += " Try to mention: " + ", ".join(r["missing"][:5]) + "."
    return m

def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z'\"(])", text.strip()) if x.strip()]

def _sentence_for(term, *sources):
    """The sentence from the lesson/reference that actually teaches `term`."""
    for src in sources:
        for snt in _sentences(src):
            if _has(term, _toks(snt)): return snt
    return None

def explain(r, name, lesson, ref):
    """Structured, concept-specific feedback built only from the lesson content and the score
    breakdown (deterministic - no LLM). Tells the learner what they got right, exactly what was
    missing and *why it matters* (the lesson sentence that covers it), and shows a model answer."""
    total, pct = r["score"], round(r["score"] * 100)
    if not r["prose_ok"]:
        head = "Write it as sentences, not a list"
        summary = (f"Your answer is {r['prose_issue']}. A good explanation links ideas together "
                   f"(what {name} is, how it works, why it matters), so it can't be marked as mastered yet.")
    elif r["passed"]:
        head = "Mastered" if total >= 0.8 else "Mastered - solid, with room to sharpen"
        summary = (f"You scored {pct}%. Your wording matched the reference closely "
                   f"({round(r['similarity'] / SIM_CAP * 100) if r['similarity'] < SIM_CAP else 100}% of the target) and you used "
                   f"{len(r['matched'])} of {len(r['matched']) + len(r['missing'])} key terms.")
    elif r["coverage"] >= 0.7:
        head = "Close - you have the vocabulary, but the idea is not connected yet"
        summary = (f"You scored {pct}% (pass mark {round(THRESHOLD * 100)}%). You used most key terms but the way you "
                   f"put them together differs from the lesson. Explain how the pieces relate, not just what they are called.")
    elif len(r["matched"]) >= 2 or r["similarity"] / SIM_CAP >= 0.4:
        head = "Right direction, but key ideas are missing"
        summary = (f"You scored {pct}% (pass mark {round(THRESHOLD * 100)}%). What you wrote is on topic and you "
                   f"used {len(r['matched'])} key term(s), but {len(r['missing'])} important one(s) are missing, listed below. "
                   f"Marking looks for the lesson's vocabulary, so if you described an idea in different words, "
                   f"use the lesson's exact term as well.")
    else:
        head = "Not there yet - re-read the lesson and try again"
        summary = (f"You scored {pct}% (pass mark {round(THRESHOLD * 100)}%). Your answer doesn't yet reflect the lesson. "
                   f"Read it once more, then explain it without looking.")
    got = [{"term": t, "from_lesson": _sentence_for(t, lesson, ref)} for t in r["matched"]]
    gaps = [{"term": t, "from_lesson": _sentence_for(t, lesson, ref)} for t in r["missing"]]
    if r["passed"] and not gaps: nxt = "Nothing missing - try explaining it with an example of your own to lock it in."
    elif r["passed"]: nxt = "To make it stick, work the missing terms below into your own words before your next review."
    else: nxt = "Re-read the lesson, then rewrite your answer covering each missing term below in a full sentence."
    return dict(headline=head, summary=summary, strengths=got, gaps=gaps, model_answer=ref, next_step=nxt)
