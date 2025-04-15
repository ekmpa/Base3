"""
Proposed Base3 baseline combines the logic 
of EdgeBank and PopTrack, augmented with t-CoMems. 

See README for more.

Edgebank code is from the DBG github; 
https://github.com/fpour/DGB/blob/main/
"""
import os
import gc
import csv
import math
import time
import random

import numpy as np
import pandas as pd
from sklearn.metrics import *

from load_data import Data, get_data
from args_parser import parse_args_edge_bank
from edge_sampler import (
    RandEdgeSampler,
    RandEdgeSampler_adversarial,
    recently_popular_negative_sampling,
    LazyRandEdgeSampler,
    RandEdgeSamplerFast
)

from evaluation import *  
from Base3 import *  
#import psutil

#import py_tgb as tgb
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

# numpy settings
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(0)
random.seed(0)

def predict_links(edgebank_mem, edge_set, poptrack_model, timestamps_batch,
                  poptrack_K, step_mem):

    source_nodes, destination_nodes = edge_set
    pred = []

    top_k_nodes, _ = poptrack_model.predict_batch(K=poptrack_K)
    top_k_nodes = set(top_k_nodes)

    for i in range(len(destination_nodes)):
        u, v = source_nodes[i], destination_nodes[i]
        t = timestamps_batch[i]

        score = full_interpolated_score(u, v, t, edgebank_mem, poptrack_model, top_k_nodes, step_mem) 

        pred.append(score)

    poptrack_model.update_batch(
        dest_nodes=destination_nodes,
        #timestamps=timestamps_batch,
        #src_nodes=source_nodes 
    )

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


def edge_bank_time_window_memory(sources_list, destinations_list, timestamps_list, window_mode, memory_span=0.01):
    """
    only saves the edges seen the time time interval equal to the last time window in timestamps_list
    """
    #print("Timestamps max:", max(timestamps_list))
    
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

    mem_edges = time_window_edge_memory(sources_list, destinations_list, timestamps_list, start_time=window_start_ts,
                                        end_time=window_end_ts)
    
    return mem_edges


def pred_end_to_end(history_data, positive_edges, negative_edges, memory_opt, poptrack_K):
    srcs = history_data.sources
    dsts = history_data.destinations
    ts_list = history_data.timestamps
    pos_sources, pos_destinations = positive_edges
    neg_sources, neg_destinations = negative_edges
    assert (len(srcs) == len(dsts))
    assert (len(pos_sources) == len(pos_destinations))
    assert (len(neg_sources) == len(neg_destinations))

    #print("Gen memories")

    # Generate memories
    mem_edges = edge_bank_time_window_memory(
        srcs, dsts, ts_list,
        window_mode="fixed", #memory_opt.get("w_mode", "fixed"),
        memory_span=0.01 #memory_opt.get("eb_mem_span", 0.01) 
    )
    #mem_edges = edge_bank_unlimited_memory(srcs, dsts)  
    #mem_edges = edge_bank_infin_freq(srcs, dsts)  
    
    # Initialize PopTrack and t-CoMem model
    num_nodes = max(
        max(srcs), max(dsts), 
        max(pos_sources), max(pos_destinations),
        max(neg_sources), max(neg_destinations)
    ) + 1

    current_time = max(np.max(pos_destinations), np.max(neg_destinations))

    step_mem = tCoMem(srcs, dsts, ts_list, current_time, num_nodes, time_window=1_000_000)
    
    poptrack_model = PopTrack(num_nodes=num_nodes)
    poptrack_model.update_batch(dsts) 

    #print("done memories, going to pred")

    # Pretrain poptrack_model on history before test
    #for u, v, t in zip(srcs, dsts, ts_list):
     #   poptrack_model.update_batch([v], timestamps=[t], src_nodes=[u])

    pos_pred = predict_links(mem_edges, positive_edges, poptrack_model,
                            positive_edges[1], poptrack_K, step_mem)
    neg_pred = predict_links(mem_edges, negative_edges, poptrack_model,
                            negative_edges[1], poptrack_K, step_mem)#pos_pred = predict_links(mem_edges, positive_edges, poptrack_model, max(ts_list), poptrack_K)

    return pos_pred, neg_pred


def pred_batch(train_val_data, test_data, rand_sampler, args, evaluator):
    """
    Batch-based link prediction
    """
    if rand_sampler is not None:
        assert rand_sampler.seed is not None
        rand_sampler.reset_random_state()

    TEST_BATCH_SIZE = args['batch_size']
    POPTRACK_K = int(args['K'])

    num_test_instance = len(test_data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    agg_pred_score, agg_true_label = [], []
    val_ap, val_auc_roc, measures_list = [], [], []

    fallback_count = 0
    total_batches = 0

    # tgb metrics
    tgbs = []

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
        if rand_sampler is None and args['neg_sample'] == 'rp_ns':
            src_negative_samples, dst_negative_samples, was_fallback = recently_popular_negative_sampling(
                size=size,
                src_nodes_all=train_val_data.sources,
                dst_nodes_all=train_val_data.destinations,
                ts_all=train_val_data.timestamps,
                current_time=timestamps_batch[-1],
                pos_src=sources_batch,
                pos_dst=destinations_batch,
                seed=2
            )
            fallback_count += int(was_fallback)
            total_batches += 1
        elif rand_sampler.neg_sample != 'rnd':
            src_negative_samples, dst_negative_samples = rand_sampler.sample(
                size, sources_batch, destinations_batch,
                timestamps_batch[0], timestamps_batch[-1]
            )
        else:
            src_negative_samples, dst_negative_samples = rand_sampler.sample(
                size, sources_batch, destinations_batch
            )
            #src_negative_samples = sources_batch
        
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

        #print("going into pred")

        # performance evaluation
        pos_pred, neg_pred = pred_end_to_end(history_data, positive_edges, negative_edges, memory_opt, POPTRACK_K)

        batch_size = len(pos_pred)
        num_neg = len(neg_pred) // batch_size
        neg_scores = neg_pred.reshape((batch_size, num_neg))

        #print("going into eval")

        tgb_metrics = evaluator.eval({
            "y_pred_pos": pos_pred,
            "y_pred_neg": neg_pred,
            "eval_metric": ["hits@", "mrr"],
        })
        #print("EVAL", tgb_metrics)

        # Concatenate predictions and labels
        pred_score = np.concatenate([pos_pred, neg_pred])
        agg_pred_score = np.concatenate([agg_pred_score, pred_score])
        assert len(pred_score) == len(true_label), "Lengths of predictions and true labels do not match!"

        # Standard metrics
        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc_roc.append(roc_auc_score(true_label, pred_score))

        # Extra measures + add MRR
        measures_dict = extra_measures(true_label, pred_score)
        measures_list.append(measures_dict)
        tgbs.append(tgb_metrics)

        # Average TGB evaluator metrics across batches
    tgb_averaged = pd.DataFrame(tgbs).mean().to_dict()

    # Average your own custom metrics (AP, AUROC, etc.)
    avg_measures_dict = pd.DataFrame(measures_list).mean().to_dict()

    # Merge all metrics together into one dictionary
    merged_metrics = {**tgb_averaged, **avg_measures_dict}

    if args['neg_sample'] == 'rp_ns' and total_batches > 0:
        rate = 100 * fallback_count / total_batches
        print(f"RP-NS fallback used in {fallback_count}/{total_batches} batches ({rate:.2f}%)")

    return np.mean(val_ap), np.mean(val_auc_roc), merged_metrics

def truncate_mask(mask, pct=1.0, keep='first'):
    """Truncate a mask by keeping only a pct% of True indices.

    Args:
        mask: boolean array.
        pct: float ∈ (0, 1].
        keep: 'first' or 'last' — whether to keep the beginning or end.
    """
    true_indices = np.where(mask)[0]
    n_keep = int(len(true_indices) * pct)

    if keep == 'first':
        selected = true_indices[:n_keep]
    elif keep == 'last':
        selected = true_indices[-n_keep:]
    else:
        raise ValueError("`keep` must be 'first' or 'last'")

    truncated = np.zeros_like(mask, dtype=bool)
    truncated[selected] = True
    return truncated

def main():
    print("===========================================================================")
    cm_args = parse_args_edge_bank()
    print("===========================================================================")

    network_name = cm_args.data
    val_ratio = cm_args.val_ratio
    test_ratio = cm_args.test_ratio
    n_runs = cm_args.n_runs
    NEG_SAMPLE = cm_args.neg_sample
    learn_through_time = True

    args = {
        'network_name': network_name,
        'n_runs': n_runs,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'm_mode': cm_args.mem_mode,
        'w_mode': cm_args.w_mode,
        'K': cm_args.k_val,
        'learn_through_time': learn_through_time,
        'batch_size': 200, # untested 200,
        'neg_sample': NEG_SAMPLE
    }

    print(f"INFO: Loading TGB dataset: {network_name}")
    os.environ["TGB_AUTOMATIC_DOWNLOAD"] = "1"
    tgb_dataset = LinkPropPredDataset(name=network_name, root="datasets", preprocess=True)

    full_data = tgb_dataset.full_data
    full_data = {
        key: val.astype(np.int32) if val.dtype == np.int64 else val
        for key, val in tgb_dataset.full_data.items()
    }
    #print("DEBUG: Keys in full_data:", full_data.keys())

    # Use TGB official masks
    train_mask = tgb_dataset.train_mask
    val_mask = tgb_dataset.val_mask
    test_mask = tgb_dataset.test_mask

    # Truncate: keep *last* of train, but *first* of val/test
    train_mask = truncate_mask(train_mask, pct=.2, keep='last')
    val_mask = truncate_mask(val_mask, pct=.2, keep='first')
    test_mask = truncate_mask(test_mask, pct=.2, keep='first')

    def extract_data(mask):
        return Data(
            sources=full_data['sources'][mask],
            destinations=full_data['destinations'][mask],
            timestamps=full_data['timestamps'][mask],
            edge_idxs=full_data['edge_idxs'][mask],
            labels=full_data['edge_label'][mask]
        )

    train_data = extract_data(train_mask)
    val_data = extract_data(val_mask)
    test_data = extract_data(test_mask)

    # Combined sets
    #tr_val_data = Data(
     #   sources=np.concatenate([train_data.sources, val_data.sources]),
      #  destinations=np.concatenate([train_data.destinations, val_data.destinations]),
       # timestamps=np.concatenate([train_data.timestamps, val_data.timestamps]),
        #edge_idxs=np.concatenate([train_data.edge_idxs, val_data.edge_idxs]),
    #    labels=np.concatenate([train_data.labels, val_data.labels])
    #)


    tr_val_data = Data(
        sources=np.concatenate([train_data.sources, val_data.sources]),
        destinations=np.concatenate([train_data.destinations, val_data.destinations]),
        timestamps=np.concatenate([train_data.timestamps, val_data.timestamps]),
        edge_idxs=np.concatenate([train_data.edge_idxs, val_data.edge_idxs]),
        labels=np.concatenate([train_data.labels, val_data.labels])
    )

    #full_data_combined = Data(
    #    sources=full_data['sources'],
    #    destinations=full_data['destinations'],
    #    timestamps=full_data['timestamps'],
    #    edge_idxs=full_data['edge_idxs'],
    #    labels=full_data['edge_label']
    #)
    
    #print("DEBUG: Full data size:", len(full_data_combined.sources))
    

    print("DEBUG: Creating sampler")

    
    #print(f"[DEBUG] Memory used: {psutil.virtual_memory().used / 1024 ** 3:.2f} GiB")


    # Don’t build full_data_combined — just extract sampler inputs
    all_sources = full_data['sources']
    all_destinations = full_data['destinations']
    all_timestamps = full_data['timestamps']

    # idx = np.random.choice(len(all_sources), size=500_000, replace=False)
    #all_sources_trunc = all_sources[idx]
    #all_destinations_trunc = all_destinations[idx]
    #all_timestamps_trunc = all_timestamps[idx]

    # Optional: Free large full_data dictionary to reclaim memory
    #del full_data
    #gc.collect()

    #print(f"[DEBUG] Memory used: {psutil.virtual_memory().used / 1024 ** 3:.2f} GiB")

    if NEG_SAMPLE == 'rp_ns':
        print("INFO: Using Recently Popular Negative Sampling (RP-NS)")
        test_rand_sampler = None
    elif NEG_SAMPLE != 'rnd':
        print(f"INFO: Negative Edge Sampling: {NEG_SAMPLE}")
        test_rand_sampler = RandEdgeSampler_adversarial(
            all_sources, all_destinations, all_timestamps,
            val_data.timestamps[-1], NEG_SAMPLE, seed=2
        )
    else:
        print(f"INFO: Negative Edge Sampling: {NEG_SAMPLE}")
        test_rand_sampler = RandEdgeSamplerFast(all_sources, all_destinations, seed=2)

    results_file = "experiments.csv"
    write_header = True

    # Sampling
    #if NEG_SAMPLE == 'rp_ns':
     #   print("INFO: Using Recently Popular Negative Sampling (RP-NS)")
      #  test_rand_sampler = None
    #elif NEG_SAMPLE != 'rnd':
     #   print(f"INFO: Negative Edge Sampling: {NEG_SAMPLE}")
      #  test_rand_sampler = RandEdgeSampler_adversarial(
       #     full_data_combined.sources, full_data_combined.destinations, full_data_combined.timestamps,
        #    val_data.timestamps[-1], NEG_SAMPLE, seed=2
        #)
    #else:
        #full_data_combined.sources = full_data_combined.sources[:500_000]
        #full_data_combined.destinations = full_data_combined.destinations[:500_000]
        # below 2 -> for RAM? untested
        #full_data_combined.sources = full_data_combined.sources.astype(np.int32)
        #full_data_combined.destinations = full_data_combined.destinations.astype(np.int32)
      #  test_rand_sampler = RandEdgeSampler(full_data_combined.sources, full_data_combined.destinations, seed=2)
        
    #print(f"[DEBUG] Memory used: {psutil.virtual_memory().used / 1024 ** 3:.2f} GiB")

    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            fieldnames = ["dataset", "neg_sample", "run","K",
                            "val_mrr", "val_ap", "val_au_roc_score",
                            "test_mrr", "test_ap", "test_au_roc_score"
                            ]
                
            writer.writerow(fieldnames)
        write_header = False

        for i_run in range(n_runs):
            print("INFO:root:****************************************")
            for k, v in args.items():
                print(f"INFO:root:{k}: {v}")
            print(f"INFO:root:Run: {i_run}")

            start_time_run = time.time()

            #print(f"[DEBUG] Memory before pred_batch used: {psutil.virtual_memory().used / 1024 ** 3:.2f} GiB")

            # Evaluate on validation set
            val_ap, val_auc_roc, val_measures_dict = pred_batch(
                train_data, val_data, test_rand_sampler, args, evaluator=Evaluator(name=network_name)
            )

            print(f'INFO: Validation -- MRR: {val_measures_dict.get("mrr"):.4f}')

            # Evaluate on test set
            test_ap, test_auc_roc, test_measures_dict = pred_batch(
                tr_val_data, test_data, test_rand_sampler, args, evaluator=Evaluator(name=network_name)
            )

            print(f'INFO: Test -- MRR: {test_measures_dict.get("mrr"):.4f}')

            elapsed_time = time.time() - start_time_run
            print(f'INFO: Run: {i_run}, Elapsed time: {elapsed_time:.2f}s')
            print("INFO:****************************************")

            row = {
                "dataset": network_name,
                "neg_sample": NEG_SAMPLE,
                "run": i_run,
                "K": args['K'], 
                "val_mrr": val_measures_dict.get("mrr"),
                "val_ap": val_ap,
                "val_au_roc_score": val_auc_roc,
                "test_mrr": test_measures_dict.get("mrr"),
                "test_ap": test_ap,
                "test_au_roc_score": test_auc_roc
            }

            fieldnames = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()
                write_header = False

            writer.writerow(row)

    print("===========================================================================")


if __name__ == '__main__':
    main()


