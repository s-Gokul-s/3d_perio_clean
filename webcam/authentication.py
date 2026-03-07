import numpy as np
from scipy.spatial.distance import cosine
from config import MATCH_THRESHOLD, SMOOTHING_WINDOW

# Margin required between best and second-best identity
MARGIN_THRESHOLD = 0.05


def authenticate(sig, db, score_history):

    user_scores = {}

    # ------------------------------------------------
    # Compute similarity for each enrolled user
    # ------------------------------------------------
    for user, templates in db.items():

        sims = [1 - cosine(sig, t) for t in templates]

        # Use top-3 templates instead of max
        top_scores = sorted(sims, reverse=True)[:3]

        user_scores[user] = np.mean(top_scores)

    # ------------------------------------------------
    # Sort users by similarity
    # ------------------------------------------------
    sorted_users = sorted(
        user_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_user, best_score = sorted_users[0]

    # second best identity
    if len(sorted_users) > 1:
        second_best_score = sorted_users[1][1]
    else:
        second_best_score = 0

    margin = best_score - second_best_score

    # ------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------
    score_history.append(best_score)

    if len(score_history) > SMOOTHING_WINDOW:
        score_history.pop(0)

    current_score = np.mean(score_history)

    # ------------------------------------------------
    # Final decision
    # ------------------------------------------------
    if current_score > MATCH_THRESHOLD and margin > MARGIN_THRESHOLD:
        return True, best_user, current_score
    else:
        return False, "UNKNOWN USER", current_score