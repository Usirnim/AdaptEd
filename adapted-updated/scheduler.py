"""Adaptive review scheduler (REQ-3.x): base spaced intervals scaled by a rolling performance trend."""
BASE = [1, 3, 7, 14, 30]; MIN_D, MAX_D = 1, 60; LOW_D = 0; WINDOW = 5

def trend(scores):
    """scores: oldest->newest list in [0,1]. Returns (mean, slope) over the rolling window."""
    s = scores[-WINDOW:]
    if not s: return 0.0, 0.0
    return sum(s) / len(s), (s[-1] - s[0]) / max(1, len(s) - 1)

def next_interval(idx, scores, thr=0.55):
    mean, slope = trend(scores)
    f = 1.0
    if len(scores) >= 2:
        if slope < -0.05 or mean < thr + 0.1: f = 0.6        # declining / weak -> sooner
        elif mean >= 0.8 and slope >= 0: f = 1.5             # consistently high -> later
    return max(MIN_D, min(MAX_D, round(BASE[min(idx, len(BASE) - 1)] * f)))