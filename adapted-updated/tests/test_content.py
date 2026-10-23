"""Integrity of the hand-authored curriculum - a typo here silently breaks learners."""
import networkx as nx, pytest
from conftest import ALL
import content

def test_ids_unique_and_prereqs_exist():
    ids = [k for _, (_, cs) in content.SUBJECTS.items() for k in cs]
    assert len(ids) == len(set(ids))
    for k, (*_, pre) in ALL.items():
        assert all(p in ALL for p in pre), k

def test_prereqs_stay_within_subject_and_acyclic():
    G = nx.DiGraph()
    for sub, (_, cs) in content.SUBJECTS.items():
        for k, (*_, pre) in cs.items():
            G.add_node(k)
            for p in pre: assert p in cs, f"{k} -> {p} crosses subjects"; G.add_edge(p, k)
    assert nx.is_directed_acyclic_graph(G)

@pytest.mark.parametrize("cid", ALL)
def test_each_concept_is_well_formed(cid):
    name, lesson, ref, terms, _ = ALL[cid]
    assert name and len(lesson) > 80 and len(ref) > 40 and len(terms) >= 5
    assert len(terms) == len(set(terms))

@pytest.mark.parametrize("cid", ALL)
def test_every_key_term_is_teachable(cid):
    """Each term must appear in the lesson, otherwise the learner is graded on something never taught."""
    import scoring
    _, lesson, ref, terms, _ = ALL[cid]
    tk = scoring._toks(lesson + " " + ref)
    missing = [t for t in terms if not scoring._has(t, tk)]
    assert not missing, f"{cid}: terms not in lesson/reference: {missing}"
