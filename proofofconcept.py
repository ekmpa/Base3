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
    
class TemporalCentrality:
    """
    Tracks the temporal centrality of nodes, updating their scores based on interactions over time.
    """
    def __init__(self):
        self.centrality = defaultdict(float)

    def update(self, u, v, t, decay_factor=0.99):
        """
        Updates the centrality scores for nodes u and v at time t.
        """
        self.centrality[u] = self.centrality[u] * decay_factor + 1
        self.centrality[v] = self.centrality[v] * decay_factor + 1

    def get(self, node):
        """
        Returns the centrality score for a given node.
        """
        return self.centrality.get(node, 0.0)
    
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


class THASMemory:
    """
    Tracks recent interactions within a specified time window for THAS.
    """
    def __init__(self, time_window=100000):
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
        """
        Returns the recent neighbors of a node within the time window.
        Ensures that current_time is a Python float for compatibility.
        """
        current_time = float(current_time)  # Ensure current_time is a Python float
        return [v for t, v in self.node_history[node] if current_time - t <= self.time_window]


# --- Scoring Functions ---

def thas_score(a, b, current_time, hist, centrality):
    if not isinstance(hist, THASMemory):
        thas_mem = THASMemory(time_window=100000)  # Default time window
        for node, interactions in hist.node_history.items():
            for t, neighbor in interactions:
                thas_mem.add_interaction(node, neighbor, float(t))
        hist = thas_mem
    hubs_a = hist.get_recent_neighbors(a, current_time)
    hubs_b = hist.get_recent_neighbors(b, current_time)
    shared_hubs = set(hubs_a).intersection(hubs_b)

    score = 0.0
    for h in shared_hubs:
        times_a = [current_time - t for t, v in hist.node_history[a] if v == h]
        times_b = [current_time - t for t, v in hist.node_history[b] if v == h]
        
        # Skip if either list is empty
        if not times_a or not times_b:
            continue
        
        rec_a = 1.0 / max((1 + min(times_a)), 0.1)
        rec_b = 1.0 / max((1 + min(times_b)), 0.1)
        c = centrality.get(h)
        score += rec_a * rec_b * c
    return score

def edgebank_score(u, v, edgebank):
    return 1.0 if edgebank.exists(u, v) else 0.0


def poptrack_score(u, v, poptrack):
    return poptrack.get(u) * poptrack.get(v)


def full_interpolated_score(u, v, t, hist, centrality, edgebank, poptrack,
                            alpha=0.4, beta=0.3, gamma=0.3):
    return (
        alpha * thas_score(u, v, t, hist, centrality)
        + beta * edgebank_score(u, v, edgebank)
        + gamma * poptrack_score(u, v, poptrack)
    )


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

