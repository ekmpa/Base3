import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime, timedelta
import random
from sklearn.metrics import roc_auc_score

class InteractionHistory:
    def __init__(self, time_window):
        self.time_window = time_window
        self.node_history = defaultdict(list)  # node_id -> [(timestamp, neighbor)]

    def add_interaction(self, u, v, t):
        self.node_history[u].append((t, v))
        self.node_history[v].append((t, u))

    def get_recent_neighbors(self, node, current_time):
        return [v for t, v in self.node_history[node]
                if current_time - t <= self.time_window]

class TemporalCentrality:
    def __init__(self):
        self.centrality = defaultdict(float)  # node -> score

    def update(self, u, v, t, decay_factor=0.99):
        self.centrality[u] = self.centrality[u] * decay_factor + 1
        self.centrality[v] = self.centrality[v] * decay_factor + 1

    def get(self, node):
        return self.centrality.get(node, 0.0)


def thas_score(a, b, current_time, hist: InteractionHistory, centrality: TemporalCentrality):
    hubs_a = hist.get_recent_neighbors(a, current_time)
    hubs_b = hist.get_recent_neighbors(b, current_time)
    shared_hubs = set(hubs_a).intersection(hubs_b)
    
    score = 0.0
    for h in shared_hubs:
        rec_a = 1.0 / (1 + min([current_time - t for t, v in hist.node_history[a] if v == h]))
        rec_b = 1.0 / (1 + min([current_time - t for t, v in hist.node_history[b] if v == h]))
        c = centrality.get(h)
        score += rec_a * rec_b * c
    return score

def recency_score(a, b, current_time, hist):
    # Most recent direct interaction
    times = [current_time - t for t, v in hist.node_history[a] if v == b]
    return 1.0 / (1 + min(times)) if times else 0.0

def common_neighbors_score(a, b, hist, current_time):
    na = set(hist.get_recent_neighbors(a, current_time))
    nb = set(hist.get_recent_neighbors(b, current_time))
    return len(na.intersection(nb))

# def edgebank 

# def poptrack 


def interpolated_score(a, b, current_time, hist, centrality, alpha=0.5, beta=0.3, gamma=0.2):
    return (alpha * thas_score(a, b, current_time, hist, centrality) +
            beta * edgebank_score(a, b, current_time, hist) +
            gamma * poptrack_score(a, b, hist, current_time))


def sample_negative(u, all_nodes, hist, num_samples=10):
    # change to add time awareness, popularity awareness, dynamic negatives.. 
    negatives = []
    neighbors = set(v for _, v in hist.node_history[u])
    while len(negatives) < num_samples:
        v = random.choice(all_nodes)
        if v != u and v not in neighbors:
            negatives.append(v)
    return negatives


for (u, v, t) in interactions:
    pos_score = interpolated_score(u, v, t, hist, centrality)
    scores.append(pos_score)
    labels.append(1)

    # Negative samples
    negatives = sample_negative(u, all_nodes, hist)
    for v_neg in negatives:
        neg_score = interpolated_score(u, v_neg, t, hist, centrality)
        scores.append(neg_score)
        labels.append(0)

    # Update history after scoring
    hist.add_interaction(u, v, t)
    centrality.update(u, v, t)


print("AUC:", roc_auc_score(labels, scores))

# For Hits@K and MRR
rank = rank_of_true_link_among_negatives(pos_score, neg_scores)
