import os
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score
import math

# --- Modules ---

class TemporalCentrality:
    """
    Approximates a lightweight, temporal, influence-based centrality.
    Node centrality grows more when connecting with already-important nodes.
    """
    def __init__(self):
        self.centrality = defaultdict(float)

    def update(self, u, v, t, decay_factor=0.99, influence_boost=0.1):
        cu = self.centrality[u]
        cv = self.centrality[v]

        self.centrality[u] = cu * decay_factor
        self.centrality[v] = cv * decay_factor

        # Boost based on neighbor influence (like mini PageRank)
        self.centrality[u] += 1 + influence_boost * cv
        self.centrality[v] += 1 + influence_boost * cu

    def get(self, node, default=0.01):
        return self.centrality.get(node, default)

class PopTrack:
    def __init__(self, decay=0.99):
        self.popularity = defaultdict(float)
        self.decay = decay

    def update(self, u, v):
        self.popularity[u] = self.popularity[u] * self.decay + 1
        self.popularity[v] = self.popularity[v] * self.decay + 1

    def get(self, node):
        return self.popularity[node]

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
    
from collections import defaultdict
import numpy as np

class PopTrackInterpolated:
    def __init__(self, num_nodes, decay=0.94):
        self.decay = decay
        self.popularity = np.zeros(int(num_nodes)) 

    def predict_batch(self, K=50):
        """
        Returns top-K popular node indices and their corresponding popularity scores.
        """
        top_k_indices = np.argsort(self.popularity)[::-1][:K]
        top_k_scores = self.popularity[top_k_indices]
        return top_k_indices, top_k_scores

    def update_batch(self, dest_nodes):
        """
        Apply update to the popularity vector: increment and decay.
        """
        for dst in dest_nodes:
            dst = int(dst)
            self.popularity[dst] += 1.0
        self.popularity *= self.decay  # decay all scores

def get_top_k(P, K):
    return np.argsort(P)[::-1][:K]

# --- Scoring Functions ---

def ind_thas_score(u, v, t, hist: THASMemory, time_window=10000, time_decay=0.99):
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

    return 1 / (1 + influence_score)

def edgebank_score(u, v, edgebank):
    return 1.0 if (u,v) in edgebank else 0.0

def edgebank_freq_score(u, v, edgebank):
    if (u,v) in edgebank:
        return edgebank[(u,v)]
    else: 
        return 0

def poptrack_score(u, v, poptrack_vector, default=0.1):
    v = int(v)  # 🔧 ensure it's a valid index
    if v < len(poptrack_vector):
        return poptrack_vector[v]
    else:
        return default
#def poptrack_score(u, v, poptrack):
#    return poptrack.get(v, 0.1)#math.log1p(poptrack.get(v, 0))

def full_interpolated_score(u, v, t, hist, edgebank, poptrack,
                            alpha=0.3, beta=0.5, gamma=0.2):
    if (u,v) not in edgebank:
        # Likely inductive
        alpha, beta, gamma = 0.2, 0.2, 0.6
    else:
        # Historical edge
        alpha, beta, gamma = 0.3, 0.5, 0.2
     # and even: 
     # - more to THAS if in recent time window
     # - more to Poptrack if popular
     # - less to edgebank if unseen 
    
    #alpha, beta, gamma = 1,0,0 # test next

    
    return (
        alpha * ind_thas_score(u, v, t, hist)
        + beta * edgebank_score(u, v, edgebank) # right now: running with freq
        + gamma * poptrack #poptrack_score(u, v, poptrack)
        
        #+ delta * inductive_boost
    )

# explore GraRep (Graph Representations via Global Structural Information)
	#	Uses higher-order adjacency powers (e.g., A, A², A³…) to capture multi-hop structure
	#	Based on matrix factorization


