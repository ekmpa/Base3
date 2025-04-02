# this used to be in interpobas.py. 
# brings the results down. might explore more.


# explore GraRep (Graph Representations via Global Structural Information)
	#	Uses higher-order adjacency powers (e.g., A, A², A³…) to capture multi-hop structure
	#	Based on matrix factorization


from collections import defaultdict
import math

class THASMemory:
    """
    Tracks recent interactions within a specified time window for THAS.
    """
    def __init__(self, time_window=100000):
        self.time_window = float(time_window)  # Ensure time_window is a float
        self.node_history = defaultdict(list)

    def add_interaction(self, u, v, t):
        t = float(t)  # Convert timestamp to Python float
        self.node_history[u].append((t, v))
        self.node_history[v].append((t, u))

    #def get_recent_neighbors(self, node, current_time):
    #    current_time = float(current_time)
    #    neighbors = [v for v in self.node_history[node]] # if current_time - t <= self.time_window]
    #    return neighbors


def ind_thas_score(u, v, t, hist: THASMemory, time_window=10000, time_decay=0.05):
    """
    Inductive THAS: uses only recent 1- and 2-hop neighbors of u
    to estimate influence on v.
    """
    influence_score = 0.0
    visited = set()
    recent_neighbors = [
        (ts, nbr) for ts, nbr in hist.node_history.get(u, [])
        if 0 <= t - ts <= time_window
    ]

    short_window = 0.1 * time_window  # Short window for recent interactions

    for ts1, nbr1 in recent_neighbors:
        if nbr1 == v:
            time_weight = math.exp(-time_decay * (t - ts1))
            if t - ts1 < short_window:
                time_weight *= 3  # boost very recent interactions
            influence_score += time_weight
        visited.add(nbr1)
        

        # Explore 2-hop neighbors of u
        for ts2, nbr2 in hist.node_history.get(nbr1, []):
            if nbr2 == v and 0 <= t - ts2 <= time_window:
                time_weight = math.exp(-time_decay * (t - ts2))
                influence_score += 0.5 * time_weight  # 2-hop decayed

    return influence_score #1 / (1 + influence_score)

# and this was in baseline.py:

def thas_memory(sources_list, destinations_list, timestamps_list, time_window=10000):
    """
    Generates the memory of THAS using the THASMemory class.
    """
    thas_mem = THASMemory(time_window)
    for u, v, t in zip(sources_list, destinations_list, timestamps_list):
        thas_mem.add_interaction(u, v, t)
    return thas_mem

# then when you prepare the memories you'd have
# thas_hist = thas_memory(srcs, dsts, ts_list) 
# and as an arg to predict_links

# and in full_interpolated_score you'd have
# alpha * ind_thas_score(u, v, t, hist)