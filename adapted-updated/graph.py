"""Concept-graph module: goal matching + topological-sort path (REQ-1.x)."""
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_graph(concepts, edges):
    G = nx.DiGraph()
    for c in concepts: G.add_node(c["id"], **c)
    G.add_edges_from(edges)
    return G

def match_goal(goal, G, subjects, min_sim=0.12):
    """subjects: {name: keywords}. Returns (subject, targets, sim) or None."""
    names = list(subjects)
    nodes = list(G.nodes)
    ndocs = [G.nodes[n]["name"] + " " + " ".join(G.nodes[n]["terms"]) for n in nodes]
    sdocs = [s + " " + subjects[s] + " " + " ".join(G.nodes[n]["name"] for n in nodes if G.nodes[n]["subject"] == s) for s in names]
    v = TfidfVectorizer(stop_words="english").fit(sdocs + ndocs + [goal])
    g = v.transform([goal])
    ss = cosine_similarity(g, v.transform(sdocs))[0]
    best = int(ss.argmax())
    if ss[best] < min_sim: return None
    sub = names[best]
    cand = [n for n in nodes if G.nodes[n]["subject"] == sub]
    cs = cosine_similarity(g, v.transform([ndocs[nodes.index(n)] for n in cand]))[0]
    if cs.max() >= 0.2 and cs.max() > ss[best]: targets = [cand[int(cs.argmax())]]
    else: targets = [n for n in cand if G.out_degree(n) == 0]   # whole subject
    return sub, targets, float(ss[best])

def ordered_path(G, targets):
    keep = set()
    for t in targets: keep |= nx.ancestors(G, t) | {t}
    return list(nx.lexicographical_topological_sort(G.subgraph(keep), key=lambda n: G.nodes[n]["ord"]))