# src/odds.py
import numpy as np

def odds_to_probs(o_home, o_draw, o_away):
    """
    Convert decimal odds to implied probabilities and remove vig by simple normalization.
    Returns a list [p_home, p_draw, p_away] (may contain nan).
    """
    try:
        implied = np.array([1.0/float(o_home), 1.0/float(o_draw), 1.0/float(o_away)])
    except Exception:
        return [float("nan"), float("nan"), float("nan")]
    s = implied.sum()
    if s == 0:
        return [float("nan"), float("nan"), float("nan")]
    return (implied / s).tolist()
