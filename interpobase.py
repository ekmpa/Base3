import numpy as np
from collections import defaultdict

# --- PopTrack Modules ---

class PopTrack: # rem ? 
    def __init__(self, decay=0.99):
        self.popularity = defaultdict(float)
        self.decay = decay

    def update(self, u, v):
        self.popularity[u] = self.popularity[u] * self.decay + 1
        self.popularity[v] = self.popularity[v] * self.decay + 1

    def get(self, node):
        return self.popularity[node]

class PopTrackInterpolated:
    def __init__(self, num_nodes, decay=0.94):
        self.decay = decay
        self.popularity = np.zeros(int(num_nodes)) 

    def predict_batch(self, K=100):
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

# --- Scoring Functions ---

def edgebank_score(u, v, edgebank):
    return 1.0 if (u,v) in edgebank else 0.0

def edgebank_freq_score(u, v, edgebank):
    if (u,v) in edgebank:
        return edgebank[(u,v)]
    else: 
        return 0

def poptrack_score(u, v, poptrack_vector, default=0.1):
    v = int(v)  
    if v < len(poptrack_vector):
        return poptrack_vector[v]
    else:
        return default

def full_interpolated_score(u, v, edgebank, poptrack,
                            alpha=0.3, beta=0.7) : 
    
    return (
        + alpha * edgebank_score(u, v, edgebank) # right now: running with freq
        + beta * poptrack 
    )


