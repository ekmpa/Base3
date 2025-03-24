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
        return [v for t, v in self.node_history[node] if current_time - t <= self.time_window]


class TemporalCentrality:
    def __init__(self):
        self.centrality = defaultdict(float)

    def update(self, u, v, t, decay_factor=0.99):
        self.centrality[u] = self.centrality[u] * decay_factor + 1
        self.centrality[v] = self.centrality[v] * decay_factor + 1

    def get(self, node):
        return self.centrality.get(node, 0.0)
    
class EdgeBankInf:
    def __init__(self):
        self.memory = set()

    def update(self, u, v):
        self.memory.add((u, v))

    def predict(self, u, v):
        return 1 if (u, v) in self.memory else 0
    
class EdgeBankTW:
    def __init__(self, window_size):
        self.window_size = window_size
        self.edge_buffer = []
        self.edge_set = set()

    def update(self, u, v, t):
        self.edge_buffer.append((u, v, t))
        self.edge_set.add((u, v))

        while self.edge_buffer and t - self.edge_buffer[0][2] > self.window_size:
            old_u, old_v, old_t = self.edge_buffer.pop(0)
            self.edge_set.discard((old_u, old_v))

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

def thas_score(a, b, current_time, hist, centrality):
    hubs_a = hist.get_recent_neighbors(a, current_time)
    hubs_b = hist.get_recent_neighbors(b, current_time)
    shared_hubs = set(hubs_a).intersection(hubs_b)

    score = 0.0
    for h in shared_hubs:
        times_a = [current_time - t for t, v in hist.node_history[a] if v == h]
        times_b = [current_time - t for t, v in hist.node_history[b] if v == h]
        if not times_a or not times_b:
            continue
        rec_a = 1.0 / (1 + min(times_a))
        rec_b = 1.0 / (1 + min(times_b))
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


def sample_negative(u, all_nodes, hist, num_samples=10):
    negatives = []
    neighbors = set(v for _, v in hist.node_history[u])
    while len(negatives) < num_samples:
        v = random.choice(all_nodes)
        if v != u and v not in neighbors:
            negatives.append(v)
    return negatives


# --- Synthetic Mini-Dataset for Demo (can be replaced with real one) ---

interactions = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 4), (4, 1, 5), (2, 4, 6)]
all_nodes = list(set([u for u, v, t in interactions] + [v for u, v, t in interactions]))

# --- Run Baseline ---

hist = InteractionHistory(time_window=10)
centrality = TemporalCentrality()
edgebank = EdgeBankTW(window=10)
poptrack = PopTrack()

scores = []
labels = []

for (u, v, t) in sorted(interactions, key=lambda x: x[2]):
    # Positive
    score_pos = full_interpolated_score(u, v, t, hist, centrality, edgebank, poptrack)
    scores.append(score_pos)
    labels.append(1)

    # Negative
    negatives = sample_negative(u, all_nodes, hist, num_samples=3)
    for v_neg in negatives:
        score_neg = full_interpolated_score(u, v_neg, t, hist, centrality, edgebank, poptrack)
        scores.append(score_neg)
        labels.append(0)

    # Update
    hist.add_interaction(u, v, t)
    centrality.update(u, v, t)
    edgebank.update(u, v, t)
    poptrack.update(u, v)

# Evaluate
auc = roc_auc_score(labels, scores)
auc
