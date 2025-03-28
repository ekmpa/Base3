"""
Proposed baseline
Edgebank code is from the DBG github; 
https://github.com/fpour/DGB/blob/main/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
from sklearn.metrics import *
from tqdm import tqdm
import math
from collections import defaultdict, Counter
from edge_sampler import RandEdgeSampler, RandEdgeSampler_adversarial
from load_data import Data, get_data
from args_parser import parse_args_edge_bank
from evaluation import *
from proofofconcept import *
import csv
import os

"""
np settings
"""
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(0)
random.seed(0)


def predict_links(memory, edge_set, poptrack_mem, thas_hist, centrality, current_time, edgebank_per_node):
    """
    Predict whether each edge in edge_set is an actual or a dummy edge based on a 3-factor interpolation:
    - EdgeBank (memory)
    - PopTrack (node popularity)
    - THAS (multi-hub measure)
    """
    source_nodes, destination_nodes = edge_set
    pred = []

    for i in range(len(destination_nodes)):
        u, v = source_nodes[i], destination_nodes[i]
        score = full_interpolated_score(u,v,current_time, thas_hist,centrality, memory,poptrack_mem, edgebank_per_node)
        pred.append(score)

    return np.array(pred)


def edge_bank_unlimited_memory(sources_list, destinations_list):
    """
    generates the memory of EdgeBank
    The memory stores every edges that it has seen
    """
    # generate memory
    mem_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) not in mem_edges:
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
    # print("Info: EdgeBank memory mode: >> Unlimited Memory <<")
    # print(f"Info: Memory contains {len(mem_edges)} edges.")
    return mem_edges

def edge_bank_infin_freq(sources_list, destinations_list):
    """
    generates the memory of EdgeBank_inf with frequency scores not just 
    The memory stores every edges that it has seen
    """
    # generate memory
    mem_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) not in mem_edges:
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
        else: 
            mem_edges[(sources_list[e_idx], destinations_list[e_idx])] += 1
    # print("Info: EdgeBank memory mode: >> Unlimited Memory <<")
    # print(f"Info: Memory contains {len(mem_edges)} edges.")
    # normalize to 0-1 scale
    if mem_edges:
        max_freq = max(mem_edges.values())
        if max_freq > 0:
            for edge in mem_edges:
                mem_edges[edge] /= max_freq

    return mem_edges


def edge_bank_repetition_based_memory(sources_list, destinations_list):
    """
    in memory, save edges that has repeated more than a threshold
    """
    # frequency of seeing each edge
    all_seen_edges = {}
    for e_idx in range(len(sources_list)):
        if (sources_list[e_idx], destinations_list[e_idx]) in all_seen_edges:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] += 1
        else:
            all_seen_edges[(sources_list[e_idx], destinations_list[e_idx])] = 1
    n_repeat = np.array(list(all_seen_edges.values()))
    # repeat_occur = Counter(n_repeat)  # contains something like this: {"n_repeat_e": number of times happens in
    # all_seen_edges dictionary}

    # NOTE: different values can be set to the threshold with manipulating the repeat_occur dictionary
    threshold = np.mean(n_repeat)
    # print("Info: repetition of an edge: max:", np.max(n_repeat), "; min:", np.min(n_repeat))
    # print("Info: Threshold is set equal to the average number of times an edge repeats. Threshold value:", threshold)
    mem_edges = {}
    for edge, n_e_repeat in all_seen_edges.items():
        if n_e_repeat >= threshold:
            mem_edges[edge] = 1

    # print("Info: EdgeBank memory mode: >> Repetition-based Memory <<")
    # print(f"Info: Memory contains {len(mem_edges)} edges.")

    return mem_edges


def time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time, end_time):
    """
    returns a memory that contains all edges seen during a time window
    """
    mem_mask = np.logical_and(timestamps_list <= end_time, timestamps_list >= start_time)
    src_in_window = sources_list[mem_mask]
    dst_in_window = destinations_list[mem_mask]
    mem_edges = edge_bank_unlimited_memory(src_in_window, dst_in_window)
    return mem_edges


def edge_bank_time_window_memory(sources_list, destinations_list, timestamps_list, window_mode, memory_span=0.15):
    """
    only saves the edges seen the time time interval equal to the last time window in timestamps_list
    """
    # print("Info: Total number of edges:", len(sources_list))
    if window_mode == 'fixed':
        window_start_ts = np.quantile(timestamps_list, 1 - memory_span)
        window_end_ts = max(timestamps_list)
    elif window_mode == 'avg_reoccur':
        e_ts_l = {}
        for e_idx in range(len(sources_list)):
            curr_edge = (sources_list[e_idx], destinations_list[e_idx])
            if curr_edge not in e_ts_l:
                e_ts_l[curr_edge] = []
            e_ts_l[curr_edge].append(timestamps_list[e_idx])

        sum_t_interval = 0
        for e, ts_list in e_ts_l.items():
            if len(ts_list) > 1:
                ts_interval_l = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
                sum_t_interval += np.mean(ts_interval_l)
        avg_t_interval = sum_t_interval / len(e_ts_l)
        window_end_ts = max(timestamps_list)
        window_start_ts = window_end_ts - avg_t_interval

    # print("Info: Time window mode:", window_mode)
    # print(f"Info: start window: {window_start_ts}, end window: {window_end_ts}, "
    #       f"interval: {window_end_ts - window_start_ts}")
    mem_edges = time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time=window_start_ts,
                                        end_time=window_end_ts)
    # print("Info: EdgeBank memory mode: >> Time Window Memory <<")
    # print(f"Info: Memory contains {len(mem_edges)} edges.")

    return mem_edges




def edge_bank_link_pred_end_to_end(history_data, positive_edges, negative_edges, memory_opt):
    """
    Combined baseline link prediction (EdgeBank + PopTrack + THAS)
    """
    srcs = history_data.sources
    dsts = history_data.destinations
    ts_list = history_data.timestamps
    pos_sources, pos_destinations = positive_edges
    neg_sources, neg_destinations = negative_edges
    assert (len(srcs) == len(dsts))
    assert (len(pos_sources) == len(pos_destinations))
    assert (len(neg_sources) == len(neg_destinations))

    # Initialize centrality
    centrality = TemporalCentrality()

    # Update centrality for all edges in history
    for u, v, t in zip(srcs, dsts, ts_list):
        centrality.update(u, v, t)

    # Generate memories
    mem_edges = edge_bank_unlimited_memory(srcs, dsts)  
    #mem_edges = edge_bank_infin_freq(srcs, dsts)  

    poptrack_mem = poptrack_memory(srcs, dsts, ts_list)
    thas_hist = thas_memory(srcs, dsts, ts_list, time_window=100)

    
    edgebank_by_node = build_edgebank_by_node(srcs, dsts)
    
    # Predict links
    pos_pred = predict_links(mem_edges, positive_edges, poptrack_mem, thas_hist, centrality, max(ts_list), edgebank_by_node)
    neg_pred = predict_links(mem_edges, negative_edges, poptrack_mem, thas_hist, centrality, max(ts_list), edgebank_by_node)

    return pos_pred, neg_pred


def edge_bank_link_pred_batch(train_val_data, test_data, rand_sampler, args):
    """
    EdgeBank link prediction: batch based
    """
    assert rand_sampler.seed is not None
    rand_sampler.reset_random_state()

    TEST_BATCH_SIZE = args['batch_size']
    num_test_instance = len(test_data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    agg_pred_score, agg_true_label = [], []
    val_ap, val_auc_roc, measures_list = [], [], []

    # for k in tqdm(range(num_test_batch)):
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        sources_batch = test_data.sources[s_idx:e_idx]
        destinations_batch = test_data.destinations[s_idx:e_idx]
        timestamps_batch = test_data.timestamps[s_idx:e_idx]
        # edge_idxs_batch = test_data.edge_idxs[s_idx: e_idx]
        positive_edges = (sources_batch, destinations_batch)

        size = len(sources_batch)  # number of negative edges
        if rand_sampler.neg_sample != 'rnd':
            src_negative_samples, dst_negative_samples = rand_sampler.sample(size, sources_batch, destinations_batch,
                                                                             timestamps_batch[0],
                                                                             timestamps_batch[-1])
        else:
            src_negative_samples, dst_negative_samples = rand_sampler.sample(size, sources_batch, destinations_batch)
            src_negative_samples = sources_batch  # similar to what baselines do

        negative_edges = (src_negative_samples, dst_negative_samples)

        pos_label = np.ones(size)
        neg_label = np.zeros(size)
        true_label = np.concatenate([pos_label, neg_label])
        agg_true_label = np.concatenate([agg_true_label, true_label])

        if args['learn_through_time']:
            history_data = Data(np.concatenate([train_val_data.sources, test_data.sources[: s_idx]]),
                                np.concatenate([train_val_data.destinations, test_data.destinations[: s_idx]]),
                                np.concatenate([train_val_data.timestamps, test_data.timestamps[: s_idx]]),
                                np.concatenate([train_val_data.edge_idxs, test_data.edge_idxs[: s_idx]]),
                                np.concatenate([train_val_data.labels, test_data.labels[: s_idx]]))
        else:
            history_data = train_val_data

        # Prepare memory options
        memory_opt = {
            'm_mode': args['m_mode'],
            'w_mode': args['w_mode']
        }

        # performance evaluation
        pos_pred, neg_pred = edge_bank_link_pred_end_to_end(history_data, positive_edges, negative_edges, memory_opt)
        pred_score = np.concatenate([pos_pred, neg_pred])
        agg_pred_score = np.concatenate([agg_pred_score, pred_score])
        assert (len(pred_score) == len(true_label)), "Lengths of predictions and true labels do not match!"

        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc_roc.append(roc_auc_score(true_label, pred_score))

        # extra performance measures
        measures_dict = extra_measures(true_label, pred_score)
        measures_list.append(measures_dict)
    measures_df = pd.DataFrame(measures_list)
    avg_measures_dict = measures_df.mean()

    return np.mean(val_ap), np.mean(val_auc_roc), avg_measures_dict


def poptrack_memory(sources, destinations, timestamps, decay_base=0.9):
    """
    Generates the memory of PopTrack.
    Tracks the popularity of nodes based on their interactions.
    """
    popularity = defaultdict(float)
    last_update_time = defaultdict(lambda: timestamps[0])  # initialize to first timestamp

    for u, v, t in zip(sources, destinations, timestamps):
        for node in [u, v]:
            delta_t = t - last_update_time[node]
            decay = decay_base ** delta_t  # exponential decay over time gap
            popularity[node] *= decay
            popularity[node] += 1
            last_update_time[node] = t

    return popularity

def build_edgebank_by_node(sources_list, destinations_list):
    """
    Builds a dict where each node maps to a Counter of its neighbors and interaction counts.
    """
    edgebank = defaultdict(Counter)
    for u, v in zip(sources_list, destinations_list):
        edgebank[u][v] += 1
        edgebank[v][u] += 1  # If edges are undirected; remove if directed
    return edgebank

def thas_memory(sources_list, destinations_list, timestamps_list, time_window=100000):
    """
    Generates the memory of THAS using the THASMemory class.
    """
    thas_mem = THASMemory(time_window)
    for u, v, t in zip(sources_list, destinations_list, timestamps_list):
        thas_mem.add_interaction(u, v, t)
    return thas_mem

def main():
    """
    EdgeBank main execution procedure
    """
    print("===========================================================================")
    cm_args = parse_args_edge_bank()
    print("===========================================================================")
    # arguments
    network_name = cm_args.data
    val_ratio = cm_args.val_ratio
    test_ratio = cm_args.test_ratio
    n_runs = cm_args.n_runs
    NEG_SAMPLE = cm_args.neg_sample
    learn_through_time = True  # similar to memory of TGN
    args = {'network_name': network_name,
            'n_runs': n_runs,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'm_mode': cm_args.mem_mode,
            'w_mode': cm_args.w_mode,
            'learn_through_time': learn_through_time,
            'batch_size': 200,
            'neg_sample': NEG_SAMPLE}

    # path
    common_path = f'{Path(__file__).parents[0]}/data'
    # ebank_log_file = "{}/ebank_logs/EdgeBank_{}_self_sup.log".format(common_path, network_name)

    # load data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_data(common_path, network_name, val_ratio, test_ratio)

    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    tr_val_data = Data(np.concatenate([train_data.sources, val_data.sources]),
                       np.concatenate([train_data.destinations, val_data.destinations]),
                       np.concatenate([train_data.timestamps, val_data.timestamps]),
                       np.concatenate([train_data.edge_idxs, val_data.edge_idxs]),
                       np.concatenate([train_data.labels, val_data.labels]))

    # define negative edge sampler
    if NEG_SAMPLE != 'rnd':
        print("INFO: Negative Edge Sampling: {}".format(NEG_SAMPLE))
        test_rand_sampler = RandEdgeSampler_adversarial(full_data.sources, full_data.destinations, full_data.timestamps,
                                                        val_data.timestamps[-1], NEG_SAMPLE, seed=2)
    else:
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

    results_file = "all_stats.csv"
    write_header = not os.path.exists(results_file)

    # executing different runs
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "neg_sample", "run", "auc_roc", "ap"] + ["avg_" + k for k in ["precision", "recall", "f1", "acc"]])  # Adjust keys if needed

        for i_run in range(n_runs):
            print("INFO:root:****************************************")
            for k, v in args.items():
                print(f"INFO:root:{k}: {v}")
            print(f"INFO:root:Run: {i_run}")

            start_time_run = time.time()
            inherent_ap, inherent_auc_roc, avg_measures_dict = edge_bank_link_pred_batch(
                tr_val_data, test_data, test_rand_sampler, args)

            print(f'INFO:root:Test statistics: Old nodes -- auc_inherent: {inherent_auc_roc}')
            print(f'INFO:root:Test statistics: Old nodes -- ap_inherent: {inherent_ap}')
            for measure_name, measure_value in avg_measures_dict.items():
                print(f'INFO:root:Test statistics: Old nodes -- {measure_name}: {measure_value}')

            elapse_time = time.time() - start_time_run
            print(f'INFO:root:EdgeBank: Run: {i_run}, Elapsed time: {elapse_time}')
            print("INFO:root:****************************************")

            # Save to CSV
            row = {
                "dataset": network_name,
                "neg_sample": NEG_SAMPLE,
                "run": i_run,
                "auc_inherent": inherent_auc_roc,
                "ap_inherent": inherent_ap
            }

            for key, value in avg_measures_dict.items():
                row[key] = value

            fieldnames = list(row.keys())

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False  # Don't write again for next run
            writer.writerow(row)

    print("===========================================================================")

if __name__ == '__main__':
    main()


