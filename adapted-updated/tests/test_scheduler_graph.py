import pytest, scheduler as S, graph, content

def test_first_review_uses_base_interval():
    assert S.next_interval(0, [0.9]) == 1 and S.next_interval(2, [0.7]) == 7

def test_intervals_grow_with_index():
    vals = [S.next_interval(i, [0.7, 0.7]) for i in range(5)]; assert vals == sorted(vals)

def test_strong_scores_stretch_and_weak_shrink():
    assert S.next_interval(2, [0.85, 0.9, 0.95]) > S.next_interval(2, [0.7, 0.7])
    assert S.next_interval(2, [0.9, 0.8, 0.6]) < S.next_interval(2, [0.7, 0.7])   # declining
    assert S.next_interval(3, [0.6, 0.6]) < 14                                      # weak

def test_bounds_and_overflow_index():
    assert S.MIN_D <= S.next_interval(99, [0.99, 0.99, 0.99]) <= S.MAX_D
    assert S.next_interval(0, [0.0, 0.0]) >= S.MIN_D

def test_trend_empty_and_window():
    assert S.trend([]) == (0.0, 0.0)
    m, s = S.trend([0.0] * 10 + [0.2, 0.4, 0.6, 0.8, 1.0]); assert m == pytest.approx(0.6) and s == pytest.approx(0.2)

def _G():
    cs = [dict(id=k, name=v[0], terms=v[3], subject=s, ord=i) for i, (s, k, v) in
          enumerate((s, k, v) for s, (_, c) in content.SUBJECTS.items() for k, v in c.items())]
    ed = [(p, k) for _, (_, c) in content.SUBJECTS.items() for k, v in c.items() for p in v[4]]
    return graph.build_graph(cs, ed)

def test_path_respects_prerequisites():
    G = _G(); p = graph.ordered_path(G, ["py_err"])
    assert p[-1] == "py_err" and set(p) == {"py_var", "py_ctl", "py_col", "py_fn", "py_err"}
    assert all(p.index(a) < p.index(b) for a, b in G.edges if a in p and b in p)

def test_path_deterministic():
    G = _G(); assert graph.ordered_path(G, ["ml_nn"]) == graph.ordered_path(G, ["ml_nn"])

SUBS = {s: v[0] for s, v in content.SUBJECTS.items()}
@pytest.mark.parametrize("goal,sub", [("I want to learn the basics of Python", "Python Basics"),
    ("teach me DNS and how the internet works", "Computer Networks"), ("learn about neural networks and overfitting", "Machine Learning Basics"),
    ("hypothesis testing and p-values", "Statistics Basics"), ("css flexbox and html", "Web Development"),
    ("stacks queues and hash tables", "Data Structures")])
def test_goal_matching(goal, sub):
    m = graph.match_goal(goal, _G(), SUBS); assert m and m[0] == sub

def test_nonsense_goal_rejected():
    assert graph.match_goal("how to bake sourdough bread", _G(), SUBS) is None

def test_specific_goal_targets_one_concept():
    assert graph.match_goal("learn about DNS", _G(), SUBS)[1] == ["net_dns"]
