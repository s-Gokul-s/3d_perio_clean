import numpy as np
from scipy.spatial.distance import cosine
from config import MATCH_THRESHOLD, SMOOTHING_WINDOW

def authenticate(sig, db, score_history):
    best_sim = 0
    best_user = "None"

    for user, templates in db.items():
        sims = [1 - cosine(sig, t) for t in templates]
        max_s = max(sims)

        if max_s > best_sim:
            best_sim = max_s
            best_user = user

    score_history.append(best_sim)
    if len(score_history) > SMOOTHING_WINDOW:
        score_history.pop(0)

    current_score = np.mean(score_history)

    if current_score > MATCH_THRESHOLD:
        return True, best_user, current_score
    else:
        return False, "UNKNOWN USER", current_score
