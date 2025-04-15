import numpy as np
from collections import defaultdict, deque
from scipy.sparse import dok_matrix


# ==========================
# --- t-CoMem Module ---
# ==========================


class tCoMem:
    def __init__(self, srcs, dsts, ts_list, current_time, time_window=1_000_000, co_occurrence_weight=0.5):
        self.node_to_recent_dests = defaultdict(lambda: deque(maxlen=100))  # deque saves memory and is faster
        self.node_to_co_occurrence = defaultdict(lambda: defaultdict(int))  # Tracks co-occurrence of (u, v)
        self.time_window = time_window
        self.co_occurrence_weight = co_occurrence_weight

        for u, v, t in zip(srcs, dsts, ts_list):
            if 0 <= current_time - t <= self.time_window:
                self.update(u, v, t)

    def update(self, u, v, t):
        self.node_to_recent_dests[u].append((t, v))
        self.node_to_co_occurrence[u][v] += 1
        self.node_to_co_occurrence[v][u] += 1

    def updateOld(self, u, v, t):
        self.node_to_recent_dests[u].append((t, v))

        # Co-occurrence as sparse dicts
        #self.node_to_co_occurrence.setdefault(u, {}).setdefault(v, 0)
        #self.node_to_co_occurrence.setdefault(v, {}).setdefault(u, 0)

        self.node_to_co_occurrence[u][v] += 1
        self.node_to_co_occurrence[v][u] += 1

    def get_poptrack_score(self, v, poptrack_model):
        return poptrack_model.get_score(v)

    def get_score(self, u, v, current_time, poptrack_model):
        score = 0.0
        recent = self.node_to_recent_dests.get(u, [])
        valid_recent = [(ts, nbr) for ts, nbr in recent if 0 <= current_time - ts <= self.time_window]
        co_occurrence_score = self.node_to_co_occurrence.get(u, {}).get(v, 0)

        for ts, nbr in valid_recent:
            decay = np.exp(-(current_time - ts) / self.time_window)
            pop_score = self.get_poptrack_score(nbr, poptrack_model)
            score += decay * pop_score

        co_occurrence_influence = self.co_occurrence_weight * (co_occurrence_score / (1 + co_occurrence_score))
        score += co_occurrence_influence

        return score / (1 + score) if score > 0 else 0.0
  

# ==========================
# --- PopTrack Module ---
# ==========================


class PopTrack:
    def __init__(self, num_nodes, decay=0.995):
        self.decay = decay
        self.popularity = np.zeros(num_nodes, dtype=np.float32)  # switch to float32 to save 50% memory

    def update_batch_old(self, dest_nodes): #, timestamps=None, src_nodes=None):
        for dst in dest_nodes:
            dst = int(dst)
            self.popularity[dst] += 1.0
        self.popularity *= self.decay
    
    def update_batch(self, dest_nodes):
        np.add.at(self.popularity, dest_nodes, 1.0)
        self.popularity *= self.decay

    def predict_batch(self, K=100):
        top_k_indices = np.argsort(self.popularity)[::-1][:K]
        top_k_scores = self.popularity[top_k_indices]
        return top_k_indices, top_k_scores

    def get_score(self, node):
        return self.popularity[int(node)]

# ==========================
# --- Scoring Functions ---
# ==========================

def edgebank_score(u, v, edgebank):
    return 1.0 if (u, v) in edgebank else 0.0

def edgebank_freq_score(u, v, edgebank):
    return edgebank.get((u, v), 0)

def poptrack_score(u, v, poptrack_vector, default=0.1):
    v = int(v)
    return poptrack_vector[v] if v < len(poptrack_vector) else default

def full_interpolated_score(u, v, t, edgebank, poptrack_model, top_k_nodes, step_mem):
    """
    Interpolated score with weights smoothly adapted based on EdgeBank confidence.
    """

    if (u,v) in edgebank:
        alpha, beta, delta = 0.5, 0.2, 0.3 
    else: 
        alpha, beta, delta = 0.2, 0.3, 0.5

    # Base signals
    eb_score = edgebank_score(u, v, edgebank)
    pop_score = 1.0 if v in top_k_nodes else 0.0
    step_score = step_mem.get_score(u, v, t, poptrack_model)

    return alpha * eb_score + beta * pop_score + delta * step_score



def full_interpolated_score_conf(u, v, t, edgebank, poptrack_model, top_k_nodes, step_mem):
    """
    Interpolated score with weights smoothly adapted based on EdgeBank confidence.
    """

    # Base signals
    eb_score = edgebank_score(u, v, edgebank)
    pop_score = 1.0 if v in top_k_nodes else 0.0
    step_score = step_mem.get_score(u, v, t, poptrack_model)

    # Use eb_score itself as a soft confidence value 
    eb_conf = eb_score

    # Define weights when confident in EdgeBank vs not
    weights_if_confident = np.array([0.5, 0.2, 0.3])  # [alpha, beta, delta]
    weights_if_not = np.array([0.2, 0.3, 0.5])

    # Blend the two based on EdgeBank confidence
    blended_weights = eb_conf * weights_if_confident + (1 - eb_conf) * weights_if_not
    blended_weights /= blended_weights.sum()  # Normalize just in case

    alpha, beta, delta = blended_weights

    # alpha, beta, delta = 0,1,0

    return alpha * eb_score + beta * pop_score + delta * step_score
