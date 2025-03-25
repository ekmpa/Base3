import os
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score

# --- Modules ---

class EdgeTracker:
    def __init__(self):
        self.all_edges = set()
        self.edge_by_time = defaultdict(set)
        self.test_edges = set()

    def add(self, u, v, t):
        self.all_edges.add((u, v))
        self.edge_by_time[t].add((u, v))

    def is_edge_active(self, u, v, t):
        return (u, v) in self.edge_by_time[t]

    def was_edge_seen(self, u, v, current_time):
        for ts in range(current_time):
            if (u, v) in self.edge_by_time[ts]:
                return True
        return False
    
class InteractionHistory:
    def __init__(self, time_window=100000):
        self.time_window = time_window
        self.node_history = defaultdict(list)

    def add_interaction(self, u, v, t):
        self.node_history[u].append((t, v))
        self.node_history[v].append((t, u))

    def get_recent_neighbors(self, node, current_time):
        current_time = float(current_time)
        neighbors = [v for t, v in self.node_history[node] if current_time - t <= self.time_window]
        if not neighbors:
            print(f"⚠️ No recent neighbors for node {node} at time {current_time}. Last seen at:")
            for t, v in self.node_history[node][-5:]:  # just show last 5
                print(f"  → {v} @ t={t}, Δt={current_time - t}")
        return neighbors

from collections import defaultdict

class TemporalCentrality:
    """
    Approximates a lightweight, temporal, influence-based centrality.
    Node centrality grows more when connecting with already-important nodes.
    """
    def __init__(self):
        self.centrality = defaultdict(float)

    def update(self, u, v, t, decay_factor=0.99, influence_boost=0.1):
        """
        Updates the centrality scores for nodes u and v at time t.
        Each node gains more centrality when connecting to a high-centrality neighbor.
        """
        cu = self.centrality[u]
        cv = self.centrality[v]

        # Decay existing scores
        self.centrality[u] = cu * decay_factor
        self.centrality[v] = cv * decay_factor

        # Boost based on neighbor influence (like mini PageRank)
        self.centrality[u] += 1 + influence_boost * cv
        self.centrality[v] += 1 + influence_boost * cu

    def get(self, node):
        return self.centrality.get(node)
    
class EdgeBankInf:
    def __init__(self):
        self.memory = set()

    def update(self, u, v):
        self.memory.add((u, v))

    def predict(self, u, v):
        return 1 if (u, v) in self.memory else 0
    
class EdgeBankTW:
    def __init__(self, window):
        self.window_size = window
        self.edge_buffer = []
        self.edge_set = set()

    def update(self, u, v, t):
        self.edge_buffer.append((u, v, t))
        self.edge_set.add((u, v))

        while self.edge_buffer and t - self.edge_buffer[0][2] > self.window_size:
            old_u, old_v, old_t = self.edge_buffer.pop(0)
            self.edge_set.discard((old_u, old_v))

    def exists(self, u, v):
        return (u, v) in self.edge_set
    
    def predict(self, u, v, t):
        return 1 if (u, v) in self.edge_set else 0


class PopTrack:
    def __init__(self, decay=0.99):
        self.popularity = defaultdict(float)
        self.decay = decay

    def update(self, u, v):
        self.popularity[u] = self.popularity[u] * self.decay + 1
        self.popularity[v] = self.popularity[v] * self.decay + 1

    def get(self, node):
        return self.popularity[node]


# --- Scoring Functions ---

class THASMemory:
    """
    Tracks recent interactions within a specified time window for THAS.
    """
    def __init__(self, time_window=100000000):
        self.time_window = float(time_window)  # Ensure time_window is a float
        self.node_history = defaultdict(list)

    def add_interaction(self, u, v, t):
        """
        Adds an interaction between nodes u and v at time t.
        Ensures that the timestamp t is stored as a Python float.
        """
        t = float(t)  # Convert timestamp to Python float
        self.node_history[u].append((t, v))
        self.node_history[v].append((t, u))

    def get_recent_neighbors(self, node, current_time):
        current_time = float(current_time)
        neighbors = [v for v in self.node_history[node]] # if current_time - t <= self.time_window]
        return neighbors
    
# --- Scoring Functions ---

def thas_score(u, v, t, hist: THASMemory, centrality, hop_decay=0.5, centrality_threshold=0.1, max_hops=3):
    """
    Influence score from u to v using multi-hop propagation (PageRank-style).
    """
    visited = set()
    queue = [(u, 0, 1.0)]  # (current_node, depth, influence_weight)
    influence_score = 0.0

    while queue:
        current, depth, influence = queue.pop(0)
        if depth >= max_hops:
            continue

        neighbors = [
            nbr for ts, nbr in hist.node_history.get(current, [])
            if 0 <= t - ts <= hist.time_window and nbr not in visited
        ]

        for nbr in neighbors:
            visited.add(nbr)
            # Add influence only if centrality is strong enough
            c_score = centrality.get(nbr)
            if c_score >= centrality_threshold:
                propagated = influence * hop_decay * c_score
                if nbr == v:
                    influence_score += propagated
                else:
                    queue.append((nbr, depth + 1, propagated))

    #if influence_score > 0:
    #    print(f"[{u}→{v}] Propagated score: {influence_score:.4f}")

    return influence_score

def edgebank_score(u, v, edgebank):
    return 1.0 if (u,v) in edgebank else 0.0

def poptrack_score(u, v, poptrack):
    return poptrack.get(u, 0) * poptrack.get(v, 0)


def full_interpolated_score(u, v, t, hist, centrality, edgebank, poptrack,
                            alpha=1, beta=0, gamma=0):
    return (
        alpha * thas_score(u, v, t, hist, centrality)
        + beta * edgebank_score(u, v, edgebank)
        + gamma * poptrack_score(u, v, poptrack)
    )

#def poptrack_score(u, v, poptrack):
#    return poptrack.get(u) * poptrack.get(v)

def sample_random_negative(u, all_nodes, hist, num_samples=100):
    negatives = []
    neighbors = set(v for _, v in hist.node_history[u])
    while len(negatives) < num_samples:
        v = random.choice(all_nodes)
        if v != u and v not in neighbors:
            negatives.append(v)
    return negatives


def sample_historical_negative(t, tracker, num_samples=100):
    seen_edges = set()
    for ts in range(t):
        seen_edges |= tracker.edge_by_time[ts]

    current_edges = tracker.edge_by_time[t]
    candidate_negatives = list(seen_edges - current_edges)
    sampled = random.sample(candidate_negatives, min(num_samples, len(candidate_negatives)))
    return sampled


def sample_inductive_negative(t, tracker, train_cutoff_time, num_samples=100):
    test_only_edges = tracker.all_edges - set().union(*[tracker.edge_by_time[ts] for ts in range(train_cutoff_time)])
    current_edges = tracker.edge_by_time[t]
    candidate_negatives = list(test_only_edges - current_edges)
    sampled = random.sample(candidate_negatives, min(num_samples, len(candidate_negatives)))
    return sampled

