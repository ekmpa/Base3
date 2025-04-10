import numpy as np
from collections import defaultdict
import math

# ==========================
# --- PopTrack Module ---
# ==========================

from collections import defaultdict, deque
import numpy as np

class STePMemory:
    def __init__(self, time_window=1_000_000, co_occurrence_weight=0.5):
        self.node_to_recent_dests = defaultdict(lambda: deque(maxlen=100))  # deque saves memory and is faster
        self.node_to_co_occurrence = defaultdict(dict)  # nested dict still better than defaultdict of defaultdicts
        self.time_window = time_window
        self.co_occurrence_weight = co_occurrence_weight
        self.poptrack_scores_cache = {}

    def update(self, u, v, t):
        self.node_to_recent_dests[u].append((t, v))

        # Co-occurrence as sparse dicts
        self.node_to_co_occurrence.setdefault(u, {}).setdefault(v, 0)
        self.node_to_co_occurrence.setdefault(v, {}).setdefault(u, 0)

        self.node_to_co_occurrence[u][v] += 1
        self.node_to_co_occurrence[v][u] += 1

    def get_poptrack_score(self, v, poptrack_model):
        if v not in self.poptrack_scores_cache:
            self.poptrack_scores_cache[v] = poptrack_model.get_score(v)
        return self.poptrack_scores_cache[v]

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
    
# rename to tCoMem and Base3
class STePMemorynotOpt:
    def __init__(self, time_window=1_000_000, co_occurrence_weight=0.5):
        self.node_to_recent_dests = defaultdict(list)
        self.node_to_co_occurrence = defaultdict(lambda: defaultdict(int))  # Tracks co-occurrence of (u, v)
        self.time_window = time_window
        self.co_occurrence_weight = co_occurrence_weight  # Weight for co-occurrence in the final score
        self.poptrack_scores_cache = {}  # Cache for poptrack scores

    def update(self, u, v, t):
        """ Update recent destinations and co-occurrence for a given (u, v, t) interaction """
        self.node_to_recent_dests[u].append((t, v))
        self.node_to_co_occurrence[u][v] += 1  # Increment co-occurrence for (u, v)
        self.node_to_co_occurrence[v][u] += 1  

    def get_poptrack_score(self, v, poptrack_model):
        """ Retrieve and cache the PopTrack score for a node to avoid redundant computations """
        if v not in self.poptrack_scores_cache:
            self.poptrack_scores_cache[v] = poptrack_model.get_score(v)
        return self.poptrack_scores_cache[v]

    def get_score(self, u, v, current_time, poptrack_model):
        """ 
        Calculate the score based on recent interactions and co-occurrence
        """
        score = 0.0
        
        # Get the recent interactions for u within the time window, without redundant lookups
        recent = self.node_to_recent_dests.get(u, [])
        valid_recent = [(ts, nbr) for ts, nbr in recent if 0 <= current_time - ts <= self.time_window]
        
        # Initialize a variable for co-occurrence influence
        co_occurrence_score = self.node_to_co_occurrence[u].get(v, 0)

        # Compute the score based on temporal decay and poptrack popularity
        for ts, nbr in valid_recent:
            decay = np.exp(-(current_time - ts) / self.time_window)
            pop_score = self.get_poptrack_score(nbr, poptrack_model)  # Use cached poptrack score
            score += decay * pop_score
        
        # Normalize and consider both co-occurrence and recent interaction influence
        co_occurrence_influence = self.co_occurrence_weight * (co_occurrence_score / (1 + co_occurrence_score))
        score += co_occurrence_influence
        
        # Normalize final score between [0, 1] to avoid extreme values
        return score / (1 + score) if score > 0 else 0.0

class PopTrack:
    def __init__(self, num_nodes, decay=0.995):
        self.decay = decay
        self.popularity = np.zeros(num_nodes, dtype=np.float32)  # switch to float32 to save 50% memory

        # Use numpy arrays instead of lists
        self.history_sources = np.empty(0, dtype=np.int32)
        self.history_destinations = np.empty(0, dtype=np.int32)
        self.history_timestamps = np.empty(0, dtype=np.float64)

    def update_batch(self, dest_nodes, timestamps=None, src_nodes=None):
        for dst in dest_nodes:
            dst = int(dst)
            self.popularity[dst] += 1.0
        self.popularity *= self.decay

        if src_nodes is not None and timestamps is not None:
            self.history_sources = np.concatenate([self.history_sources, np.array(src_nodes, dtype=np.int32)])
            self.history_destinations = np.concatenate([self.history_destinations, np.array(dest_nodes, dtype=np.int32)])
            self.history_timestamps = np.concatenate([self.history_timestamps, np.array(timestamps, dtype=np.float64)])

    def predict_batch(self, K=100):
        top_k_indices = np.argsort(self.popularity)[::-1][:K]
        top_k_scores = self.popularity[top_k_indices]
        return top_k_indices, top_k_scores

    def get_score(self, node):
        return self.popularity[int(node)]
    
class PopTrackNotOpt:
    def __init__(self, num_nodes, decay=0.995):
        self.decay = decay
        self.popularity = np.zeros(int(num_nodes))  # corresponds to P in paper

        # Optional: store history (for THAS or other models)
        self.history_sources = []
        self.history_destinations = []
        self.history_timestamps = []
        #self.co_tracker = CoOccurrenceTracker()

    def update_batch(self, dest_nodes, timestamps=None, src_nodes=None):
        """
        - Increment P[dst] by 1 (no recency boost)
        - Apply decay *after* the update
        """
        for dst in dest_nodes:
            dst = int(dst)
            self.popularity[dst] += 1.0  # paper uses fixed increment

        # Apply decay (after update) as per original algorithm
        self.popularity *= self.decay

        # Optional: store history for hybrid models (like THAS)
        if src_nodes is not None and timestamps is not None:
            self.history_sources.extend(src_nodes)
            self.history_destinations.extend(dest_nodes)
            self.history_timestamps.extend(timestamps)

        #if self.co_tracker is not None and src_nodes is not None and dest_nodes is not None:
        #    for u, v in zip(src_nodes, dest_nodes):
        #        self.co_tracker.update(u, v)

    def predict_batch(self, K=100):
        """
        Return the top-K popular destination nodes based on raw decayed values
        """
        top_k_indices = np.argsort(self.popularity)[::-1][:K]
        top_k_scores = self.popularity[top_k_indices]
        return top_k_indices, top_k_scores

    def get_score(self, node):
        """
        Return the raw popularity score for a given node (unscaled)
        """
        node = int(node)
        return self.popularity[node]

# ==========================
# --- THAS Memory Module ---
# ==========================

class THASMemory:
    def __init__(self, time_window=1000000):
        self.time_window = float(time_window)
        self.node_history = defaultdict(list)

    def add_interaction(self, u, v, t):
        t = float(t)
        self.node_history[u].append((t, v))
        self.node_history[v].append((t, u))

def thas_memory(sources_list, destinations_list, timestamps_list, current_time, time_window=1_000_000):
    mem = THASMemory(time_window)
    for u, v, t in zip(sources_list, destinations_list, timestamps_list):
        if t < current_time:  # Only include past edges
            mem.add_interaction(u, v, t)
    return mem

def ind_thas_score(u, v, t, hist: THASMemory, time_window=2_000_000, time_decay=0.0025):
    """
    Improved THAS:
    - Extended time window
    - Stronger decay for very old edges
    - Normalized output
    """
    raw_score = 0.0
    short_window = 0.1 * time_window

    #print(f"[DEBUG] node_history[{u}] =", hist.node_history.get(u, None))
    neighbors = [
        (ts1, nbr1) for ts1, nbr1 in hist.node_history.get(u, [])
        if 0 <= t - ts1 <= time_window
    ]

    for ts1, nbr1 in neighbors:
        # 1-hop
        if nbr1 == v:
            decay = math.exp(-time_decay * (t - ts1))
            if t - ts1 < short_window:
                decay *= 2  # soft boost
            raw_score += decay

        # 2-hop
        for ts2, nbr2 in hist.node_history.get(nbr1, []):
            if nbr2 == v and 0 <= t - ts2 <= time_window:
                decay = math.exp(-time_decay * (t - ts2))
                raw_score += 0.5 * decay  # softer 2-hop weight

    # Normalize to a bounded score between 0–1
    normalized_score = raw_score / (1 + raw_score)
    return normalized_score

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

def full_interpolated_score2(u, v, t, edgebank, poptrack_model, top_k_nodes, step_mem):
    """
    Final improved score:
    - EdgeBank (optional memory match)
    - PopTrack (destination popularity)
    - THAS (recent temporal 1–2 hop influence)
    """

    if (u,v) in edgebank:
        alpha, beta, delta = 0.5, 0.2, 0.3 
    else: 
        alpha, beta, delta = 0.2, 0.3, 0.5

    pop_score = 1.0 if v in top_k_nodes else 0.0 
    eb_score = edgebank_score(u, v, edgebank)
    step_score = step_mem.get_score(u, v, t, poptrack_model)
    #thas_score = ind_thas_score(u, v, t, thas_mem)
    #co_score = co_tracker.get_score(u, v) if co_tracker else 0.0
    #print(f"[DEBUG] THAS score u={u}, v={v} at time {t}: {thas_score:.4f}")

    #alpha, beta, gamma, delta = 0,0,0,1

    return alpha * eb_score + beta * pop_score + delta * step_score

def full_interpolated_score(u, v, t, edgebank, poptrack_model, top_k_nodes, step_mem):
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