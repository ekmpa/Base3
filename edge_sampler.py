"""
Negative edge sampler
From the DBG github; 
https://github.com/fpour/DGB/blob/main/
"""

import numpy as np
import random

class RandEdgeSamplerFast:
    def __init__(self, src_list, dst_list, seed=None):
        self.neg_sample = 'rnd'
        self.src_list = np.unique(np.array(src_list, dtype=np.int32))
        self.dst_list = np.unique(np.array(dst_list, dtype=np.int32))

        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else np.random

    def edge_hash(self, u, v):
        return (int(u) << 32) + int(v)

    def sample(self, size, pos_src, pos_dst):
        pos_hash_set = set(self.edge_hash(u, v) for u, v in zip(pos_src, pos_dst))

        neg_src = []
        neg_dst = []
        attempts = 0
        max_attempts = size * 2

        while len(neg_src) < size and attempts < max_attempts:
            u = self.random_state.choice(self.src_list)
            v = self.random_state.choice(self.dst_list)
            if self.edge_hash(u, v) not in pos_hash_set:
                neg_src.append(u)
                neg_dst.append(v)
            attempts += 1

        if len(neg_src) < size:
            print(f"[WARN] Only generated {len(neg_src)} negative edges after {attempts} attempts.")

        return neg_src, neg_dst

    def reset_random_state(self):
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

class RandEdgeSampler_original(object):
    """
    from TGN code
    """

    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'  # negative edge sampling method: random edges
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSamplero(object):

    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'  # negative edge sampling method: random edges
        self.src_list = np.unique(np.array(src_list, dtype=np.int32))
        self.dst_list = np.unique(np.array(dst_list, dtype=np.int32))

        # Directly generate set from generator expression (no intermediate list!)
        self.possible_edges = set(
            (src, dst) for src in self.src_list for dst in self.dst_list
        )

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size, pos_src, pos_dst):
        current_pos_e = set([(pos_src[i], pos_dst[i]) for i in range(len(pos_src))])
        current_not_pos_e = list(self.possible_edges - current_pos_e)
        if self.seed is None:
            e_index = np.random.randint(0, len(current_not_pos_e), size)
        else:
            e_index = self.random_state.randint(0, len(current_not_pos_e), size)
        return [current_not_pos_e[idx][0] for idx in e_index], [current_not_pos_e[idx][1] for idx in e_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

# TO TEST FOR MEM!
class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.neg_sample = 'rnd'
        self.src_list = np.unique(np.array(src_list, dtype=np.int32))
        self.dst_list = np.unique(np.array(dst_list, dtype=np.int32))

        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else np.random

    def sample(self, size, pos_src, pos_dst):
        # Set of positive edges for fast collision checking
        current_pos_e = set(zip(pos_src, pos_dst))

        neg_src = []
        neg_dst = []
        attempts = 0
        max_attempts = size * 1  # to prevent infinite loops

        while len(neg_src) < size and attempts < max_attempts:
            # Sample a candidate edge
            u = self.random_state.choice(self.src_list)
            v = self.random_state.choice(self.dst_list)
            if (u, v) not in current_pos_e:
                neg_src.append(u)
                neg_dst.append(v)
            attempts += 1

        if len(neg_src) < size:
            print(f"[WARN] Only generated {len(neg_src)} negative edges after {attempts} attempts.")

        return neg_src, neg_dst

    def reset_random_state(self):
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

class RandEdgeSampler_adversarial_original(object):
    """
    Adversarial Random Edge Sampling as Negative Edges
    """

    def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """
        if not (NEG_SAMPLE == 'hist_nre' or NEG_SAMPLE == 'induc_nre'):
            raise ValueError("Undefined Negative Edge Sampling Strategy!")

        self.seed = None
        self.neg_sample = NEG_SAMPLE
        self.rnd_sample_ratio = rnd_sample_ratio
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)
        self.ts_end = max(self.ts_list_distinct)
        self.ts_test_split = last_ts_train_val
        self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_test_split)

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
        return edges of a specific time interval
        """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, current_split_start_ts, current_split_end_ts):
        if self.neg_sample == 'hist_nre':
            negative_src_l, negative_dst_l = self.sample_hist_NRE(size, current_split_start_ts, current_split_end_ts)
        elif self.neg_sample == 'induc_nre':
            negative_src_l, negative_dst_l = self.sample_induc_NRE(size, current_split_start_ts, current_split_end_ts)
        else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        return negative_src_l, negative_dst_l

    def sample_hist_NRE(self, size, current_split_start_ts, current_split_end_ts):
        """
        method one:
        "historical adversarial sampling": (~ historical non repeating edges)
        randomly samples among previously seen edges that are not repeating in current batch,
        fill in any remaining with randomly sampled
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)
        num_smp_rnd = int(self.rnd_sample_ratio * size)
        num_smp_from_hist = size - num_smp_rnd
        if num_smp_from_hist > len(non_repeating_e_src_l):
            num_smp_from_hist = len(non_repeating_e_src_l)
            num_smp_rnd = size - num_smp_from_hist

        replace = len(self.src_list_distinct) < num_smp_rnd
        rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(self.dst_list_distinct) < num_smp_rnd
        rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

        replace = len(non_repeating_e_src_l) < num_smp_from_hist
        nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

        negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
        negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])

        return negative_src_l, negative_dst_l

    def sample_induc_NRE(self, size, current_split_start_ts, current_split_end_ts):
        """
        method two:
        "inductive adversarial sampling": (~ inductive non repeating edges)
        considers only edges that have been seen (in red region),
        fill in any remaining with randomly sampled
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        induc_adversarial_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)
        induc_adv_src_l, induc_adv_dst_l = [], []
        if len(induc_adversarial_e) > 0:
            for e in induc_adversarial_e:
                induc_adv_src_l.append(e[0])
                induc_adv_dst_l.append(e[1])
            induc_adv_src_l = np.array(induc_adv_src_l)
            induc_adv_dst_l = np.array(induc_adv_dst_l)

        num_smp_rnd = size - len(induc_adversarial_e)

        if num_smp_rnd > 0:
            replace = len(self.src_list_distinct) < num_smp_rnd
            rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
            replace = len(self.dst_list_distinct) < num_smp_rnd
            rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)

            negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], induc_adv_src_l])
            negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], induc_adv_dst_l])
        else:
            rnd_induc_hist_index = np.random.choice(len(induc_adversarial_e), size=size, replace=False)
            negative_src_l = induc_adv_src_l[rnd_induc_hist_index]
            negative_dst_l = induc_adv_dst_l[rnd_induc_hist_index]

        return negative_src_l, negative_dst_l

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_adversarial(object):
    """
    Adversarial Random Edge Sampling as Negative Edges
    """

    def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NEG_SAMPLE, seed=None, rnd_sample_ratio=0):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """
        if NEG_SAMPLE not in ['hist_nre', 'induc_nre', 'rp_ns']:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        
        self.seed = None
        self.neg_sample = NEG_SAMPLE
        self.rnd_sample_ratio = rnd_sample_ratio
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)
        self.ts_end = max(self.ts_list_distinct)
        self.ts_test_split = last_ts_train_val
        self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_test_split)
        self.possible_edges = set([(src, dst) for src in np.unique(src_list) for dst in np.unique(dst_list)])

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
        return edges of a specific time interval
        """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, pos_src, pos_dst, current_split_start_ts, current_split_end_ts):
        if self.neg_sample == 'hist_nre':
            negative_src_l, negative_dst_l = self.sample_hist_NRE(size, pos_src, pos_dst, current_split_start_ts,
                                                                  current_split_end_ts)
        elif self.neg_sample == 'induc_nre':
            negative_src_l, negative_dst_l = self.sample_induc_NRE(size, pos_src, pos_dst, current_split_start_ts,
                                                                   current_split_end_ts)
        else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        return negative_src_l, negative_dst_l

    def sample_random_not_positive_e(self, num_smp_rnd, pos_src, pos_dst):
        current_pos_e = set([(pos_src[i], pos_dst[i]) for i in range(len(pos_src))])
        current_not_pos_e = list(self.possible_edges - current_pos_e)
        replace = len(self.dst_list_distinct) < num_smp_rnd
        e_index = np.random.choice(len(current_not_pos_e), size=num_smp_rnd, replace=replace)
        neg_e_src = np.array([current_not_pos_e[idx][0] for idx in e_index])
        neg_e_dst = np.array([current_not_pos_e[idx][1] for idx in e_index])
        return neg_e_src, neg_e_dst

    def sample_hist_NRE(self, size, pos_src, pos_dst, current_split_start_ts, current_split_end_ts):
        """
        method one:
        "historical adversarial sampling": (~ historical non repeating edges)
        randomly samples among previously seen edges that are not repeating in current batch,
        fill in any remaining with randomly sampled
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)
        num_smp_rnd = int(self.rnd_sample_ratio * size)
        num_smp_from_hist = size - num_smp_rnd
        if num_smp_from_hist > len(non_repeating_e_src_l):
            num_smp_from_hist = len(non_repeating_e_src_l)
            num_smp_rnd = size - num_smp_from_hist

        # select random negative edges
        if num_smp_rnd > 0:
            neg_e_src, neg_e_dst = self.sample_random_not_positive_e(num_smp_rnd, pos_src, pos_dst)
        else:
            neg_e_src, neg_e_dst = [], []

        # select historical negative edge
        replace = len(non_repeating_e_src_l) < num_smp_from_hist
        nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)

        negative_src_l = np.concatenate([neg_e_src, non_repeating_e_src_l[nre_e_index]])
        negative_dst_l = np.concatenate([neg_e_dst, non_repeating_e_dst_l[nre_e_index]])

        return negative_src_l, negative_dst_l
    
    def reset_random_state(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def sample_induc_NRE(self, size, pos_src, pos_dst, current_split_start_ts, current_split_end_ts):
        """
        method two:
        "inductive adversarial sampling": (~ inductive non repeating edges)
        considers only edges that have been seen (in red region),
        fill in any remaining with randomly sampled
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        induc_adversarial_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)
        induc_adv_src_l, induc_adv_dst_l = [], []
        if len(induc_adversarial_e) > 0:
            for e in induc_adversarial_e:
                induc_adv_src_l.append(e[0])
                induc_adv_dst_l.append(e[1])
            induc_adv_src_l = np.array(induc_adv_src_l)
            induc_adv_dst_l = np.array(induc_adv_dst_l)

        num_smp_rnd = size - len(induc_adversarial_e)

        if num_smp_rnd > 0:
            neg_e_src, neg_e_dst = self.sample_random_not_positive_e(num_smp_rnd, pos_src, pos_dst)
            negative_src_l = np.concatenate([neg_e_src, induc_adv_src_l])
            negative_dst_l = np.concatenate([neg_e_dst, induc_adv_dst_l])
        else:
            rnd_induc_hist_index = np.random.choice(len(induc_adversarial_e), size=size, replace=False)
            negative_src_l = induc_adv_src_l[rnd_induc_hist_index]
            negative_dst_l = induc_adv_dst_l[rnd_induc_hist_index]

        return negative_src_l, negative_dst_l

import numpy as np

def recently_popular_negative_sampling(
    size,
    src_nodes_all,
    dst_nodes_all,
    ts_all,
    current_time,
    pos_src,
    pos_dst,
    time_window=1e9,
    power=0.75,
    pop_ratio=0.9,
    seed=None
):
    """
    Recently Popular Negative Sampling (RP-NS) with fallback rate tracking
    Returns:
    - negative_srcs, negative_dsts: sampled negative edge arrays
    - was_fallback: True if fallback to uniform was used
    """

    if seed is not None:
        np.random.seed(seed)

    dst_nodes_distinct = np.unique(dst_nodes_all)
    src_nodes_distinct = np.unique(src_nodes_all)

    # Step 1: Get recently active destination nodes
    recent_mask = (ts_all >= (current_time - time_window)) & (ts_all <= current_time)
    recent_dsts = dst_nodes_all[recent_mask]

    if len(recent_dsts) == 0:
        # Fallback to uniform
        negative_srcs = np.random.choice(src_nodes_distinct, size=size, replace=True)
        negative_dsts = np.random.choice(dst_nodes_distinct, size=size, replace=True)
        return negative_srcs, negative_dsts, True  # was_fallback = True

    # Step 2: Compute popularity distribution
    unique_dsts, counts = np.unique(recent_dsts, return_counts=True)
    popularity_probs = counts ** power
    popularity_probs = popularity_probs / popularity_probs.sum()

    # Step 3: Sample from popularity + uniform
    n_pop = int(size * pop_ratio)
    n_uni = size - n_pop

    sampled_dst_pop = np.random.choice(unique_dsts, size=n_pop, p=popularity_probs, replace=True)
    sampled_dst_uni = np.random.choice(dst_nodes_distinct, size=n_uni, replace=True)
    negative_dsts = np.concatenate([sampled_dst_pop, sampled_dst_uni])
    negative_srcs = np.random.choice(src_nodes_distinct, size=size, replace=True)

    return negative_srcs, negative_dsts, False  # was_fallback = False

class LazyRandEdgeSampler:
    def __init__(self, src_list, dst_list, seed=None):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else None
        self.neg_sample = 'rnd' 

    def reset_random_state(self):
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size, pos_src=None, pos_dst=None, start_ts=None, end_ts=None):
        if self.random_state is not None:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        else:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]