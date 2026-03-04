"""
authentication.py — Fixed version

KEY FIX: Use top-K mean similarity instead of single max.

Previous version took max(sims) across all templates.
With 200 templates, one noisy template could give a false high score.
More importantly, a dim-light auth frame might not match the best
template exactly — taking the mean of top 5 matches is more robust.
"""

import numpy as np
from scipy.spatial.distance import cosine
from config import MATCH_THRESHOLD, SMOOTHING_WINDOW


def authenticate(sig, db, score_history):
    best_sim  = 0.0
    best_user = "None"

    for user, templates in db.items():
        sims = [1 - cosine(sig, t) for t in templates]

        # Top-5 mean is more robust than single max:
        # - Ignores occasional bad templates
        # - Still responds quickly to a good match
        top_k = sorted(sims, reverse=True)[:5]
        user_score = float(np.mean(top_k))

        if user_score > best_sim:
            best_sim  = user_score
            best_user = user

    score_history.append(best_sim)
    if len(score_history) > SMOOTHING_WINDOW:
        score_history.pop(0)

    current_score = float(np.mean(score_history))

    if current_score > MATCH_THRESHOLD:
        return True, best_user, current_score
    else:
        return False, "UNKNOWN USER", current_score