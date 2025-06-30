"""
This is a script from TGB's examples' structure, on which to run Base3. 
It is more efficient on larger graphs.
It can be used in the bash experiments script in the same way as baseline.py.
"""

import timeit
import numpy as np
import os
import os.path as osp
import math
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.utils.utils import set_random_seed, save_results
from Base3 import *
import torch
from edge_sampler import RandEdgeSampler_adversarial

# ==================
def get_args():
    parser = argparse.ArgumentParser('*** TGB: Base3***')
    parser.add_argument('-d', '--data', type=str, default='tgbl-coin')
    parser.add_argument('--bs', type=int, default=200)
    parser.add_argument('--k_value', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--mem_mode', type=str, default='time_window', choices=['time_window', 'unlim_mem'])
    parser.add_argument('--w_mode', type=str, default='fixed', choices=['fixed', 'avg_reoccur'])
    parser.add_argument('--mem_span', type=float, default=1.0)
    parser.add_argument('--co_occurrence_weight', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='manual')
    parser.add_argument('--neg_sample', type=str, default='rnd', choices=['rnd', 'hist_nre', 'induc_nre'])
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

# ==================
def evaluate_dynamic(train_src, train_dst, train_ts, 
                     pos_src, pos_dst, pos_ts, neg_sampler, split_mode, 
                     batch_size, memory_opt, poptrack_K, method, evaluator, metric, static_negatives, neg_sample):

    

    num_batches = math.ceil(len(pos_src) / batch_size)
    perf_list = []

    edgebank_memory = edge_bank_time_window_memory(
        train_src, train_dst, train_ts,
        window_mode=memory_opt['w_mode'],
        memory_span=memory_opt['mem_span']
    )

    coMem = tCoMem(co_occurrence_weight=memory_opt['co_weight'])
    for u, v, t in zip(train_src, train_dst, train_ts):
        coMem.update(u, v, t)

    num_nodes = int(max(train_src.max(), train_dst.max(), pos_src.max(), pos_dst.max()) + 1)

    poptrack_model = PopTrack(num_nodes=num_nodes)
    poptrack_model.update_batch(train_dst)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pos_src))
        
        batch_src = pos_src[start_idx:end_idx]
        batch_dst = pos_dst[start_idx:end_idx]
        batch_ts = pos_ts[start_idx:end_idx]

        if static_negatives is not None:
            neg_batch_list = static_negatives[start_idx:end_idx]
        else:
            if neg_sample in ['hist_nre', 'induc_nre']:
                neg_srcs, neg_dsts = neg_sampler.sample(
                    size=len(batch_src),
                    pos_src=batch_src,
                    pos_dst=batch_dst,
                    current_split_start_ts=batch_ts[0],
                    current_split_end_ts=batch_ts[-1]
                )
                neg_batch_list = [[dst] for dst in neg_dsts]
        #neg_batch_list = static_negatives[start_idx:end_idx]

        top_k_nodes = poptrack_model.predict_batch(K=poptrack_K)

        for idx, neg_batch in enumerate(neg_batch_list):
            u, v, t = batch_src[idx], batch_dst[idx], batch_ts[idx]

            query_src = np.full(len(neg_batch) + 1, u, dtype=int)
            query_dst = np.concatenate(([v], neg_batch))

            preds = []
            for src_i, dst_i in zip(query_src, query_dst):
                score = full_interpolated_score(
                    src_i, dst_i, t,
                    edgebank_memory, poptrack_model,
                    top_k_nodes, coMem,
                    method=method
                )
                preds.append(score)

            pos_score = preds[0]
            neg_scores = preds[1:]

            input_dict = {
                "y_pred_pos": np.array([pos_score]),
                "y_pred_neg": np.array(neg_scores),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memories after batch
        for u_i, v_i, t_i in zip(batch_src, batch_dst, batch_ts):
            edgebank_memory[(u_i, v_i)] = t_i
            coMem.update(u_i, v_i, t_i)
            poptrack_model.update_batch([v_i])

        # Expand train history after each batch
        train_src = np.concatenate([train_src, batch_src])
        train_dst = np.concatenate([train_dst, batch_dst])
        train_ts = np.concatenate([train_ts, batch_ts])

    return np.mean(perf_list)

# ==================
# Start
start_overall = timeit.default_timer()

args, _ = get_args()

SEED = args.seed
set_random_seed(SEED)
BATCH_SIZE = args.bs
K_VALUE = args.k_value
MEM_MODE = args.mem_mode
MEM_SPAN = args.mem_span
W_MODE = args.w_mode
CO_WEIGHT = args.co_occurrence_weight
METHOD = args.method
DATA = args.data
MODEL_NAME = 'Base3_Rework'
NEG_SAMPLE = args.neg_sample

print(f"INFO: Loading TGB dataset: {DATA}")
os.environ["TGB_AUTOMATIC_DOWNLOAD"] = "1"
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)

data = dataset.full_data
metric = dataset.eval_metric

train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask


# Prepare history
hist_src = data['sources'][train_mask]
hist_dst = data['destinations'][train_mask]
hist_ts = data['timestamps'][train_mask]

# Validation data
val_src = data['sources'][val_mask]
val_dst = data['destinations'][val_mask]
val_ts = data['timestamps'][val_mask]
#val_neg = [dataset.ns_sampler.eval_set['val'][(int(src), int(dst), int(ts))] for src, dst, ts in zip(val_src, val_dst, val_ts)]

# Test data
test_src = data['sources'][test_mask]
test_dst = data['destinations'][test_mask]
test_ts = data['timestamps'][test_mask]
#test_neg = [dataset.ns_sampler.eval_set['test'][(int(src), int(dst), int(ts))] for src, dst, ts in zip(test_src, test_dst, test_ts)]

evaluator = Evaluator(name=DATA)

# Load negatives
if args.neg_sample == 'rnd':
    dataset.ns_sampler.eval_set['val'] = torch.load("tgbl-review_val_ns_v2.pkl.pt")
    dataset.ns_sampler.eval_set['test'] = torch.load("tgbl-review_test_ns_v2.pkl.pt")

    val_neg = [dataset.ns_sampler.eval_set['val'][(int(src), int(dst), int(ts))]
               for src, dst, ts in zip(val_src, val_dst, val_ts)]

    test_neg = [dataset.ns_sampler.eval_set['test'][(int(src), int(dst), int(ts))]
                for src, dst, ts in zip(test_src, test_dst, test_ts)]

    adversarial_sampler = None

elif args.neg_sample in ['hist_nre', 'induc_nre']:
    adversarial_sampler = RandEdgeSampler_adversarial(
        src_list=hist_src,
        dst_list=hist_dst,
        ts_list=hist_ts,
        last_ts_train_val=val_ts[-1],
        NEG_SAMPLE=args.neg_sample,
        seed=args.seed
    )

    val_neg = None  # handled dynamically batch-wise
    test_neg = None
else:
    raise ValueError(f"Unknown neg_sample mode: {args.neg_sample}")


memory_opt = {
    'w_mode': W_MODE,
    'mem_span': MEM_SPAN,
    'co_weight': CO_WEIGHT
}

print("==========================================================")
print(f"============*** {MODEL_NAME}: {MEM_MODE}: {DATA} ***==============")
print("==========================================================")

# Validation evaluation
print("INFO: Start validation evaluation...")
start_val = timeit.default_timer()
val_perf = evaluate_dynamic(
    hist_src, hist_dst, hist_ts,
    val_src, val_dst, val_ts,
    #neg_sampler=dataset.negative_sampler,
    neg_sampler=adversarial_sampler,
    split_mode="val",
    batch_size=BATCH_SIZE,
    memory_opt=memory_opt,
    poptrack_K=K_VALUE,
    method=METHOD,
    evaluator=evaluator,
    metric=metric,
    static_negatives=val_neg,
    neg_sample=NEG_SAMPLE
)
end_val = timeit.default_timer()
print(f"INFO: Validation {metric}: {val_perf:.4f}")
print(f"Validation elapsed time (s): {end_val - start_val:.4f}")

# Test evaluation
print("INFO: Start test evaluation...")
start_test = timeit.default_timer()
test_perf = evaluate_dynamic(
    np.concatenate([hist_src, val_src]),
    np.concatenate([hist_dst, val_dst]),
    np.concatenate([hist_ts, val_ts]),
    test_src, test_dst, test_ts,
    #neg_sampler=dataset.negative_sampler,
    neg_sampler=adversarial_sampler,
    split_mode="test",
    batch_size=BATCH_SIZE,
    memory_opt=memory_opt,
    poptrack_K=K_VALUE,
    method=METHOD,
    evaluator=evaluator,
    metric=metric,
    static_negatives=test_neg,
    neg_sample=NEG_SAMPLE
)
end_test = timeit.default_timer()
print(f"INFO: Test {metric}: {test_perf:.4f}")
print(f"Test elapsed time (s): {end_test - start_test:.4f}")

# Save results
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{MEM_MODE}_{DATA}_results.json'

save_results({
    'model': MODEL_NAME,
    'memory_mode': MEM_MODE,
    'data': DATA,
    'run': 1,
    'seed': SEED,
    'val_' + metric: float(val_perf),
    'test_' + metric: float(test_perf),
    'test_time': end_test - start_test,
    'tot_train_val_time': 'NA',
    'method': METHOD,
}, results_filename)

print(f"Overall elapsed time: {timeit.default_timer() - start_overall:.2f}s")
print("==============================================================")
