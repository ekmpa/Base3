"""
Parsing arguments of different scripts
Initial code from the DBG github; https://github.com/fpour/DGB/blob/main/
"""

import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser('*** Base3 ***')

    # Dataset
    parser.add_argument('-d', '--data', type=str, default='tgbl-coin',
                        help='Name of the dataset to evaluate (default: tgbl-coin).')

    # Execution and model parameters
    parser.add_argument('--bs', type=int, default=200,
                        help='Batch size for processing edges (default: 200).')
    parser.add_argument('--k_value', type=int, default=1000,
                        help='Top-K value for PopTrack popularity scoring (default: 1000).')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed for reproducibility (default: 2).')

    # EdgeBank memory settings
    parser.add_argument('--mem_mode', type=str, default='time_window', 
                        choices=['time_window', 'unlim_mem'],
                        help='Memory mechanism for EdgeBank.')
    parser.add_argument('--w_mode', type=str, default='fixed', 
                        choices=['fixed', 'avg_reoccur'],
                        help='Window selection mode for time_window memory.')

    # Memory and co-occurrence tuning
    parser.add_argument('--mem_span', type=float, default=0.1,
                        help='Time span for memory (as a fraction of total history).')
    parser.add_argument('--co_occurrence_weight', type=float, default=1.0,
                        help='Weight assigned to co-occurrence scores in t-CoMem.')

    # Interpolation strategy
    parser.add_argument('--method', type=str, default='multi_conf',
                        help='Scoring strategy to use (e.g., PopTrack, EdgeBank, t-CoMem, multi_conf, etc.).')

    parser.add_argument('--neg_sample', type=str, default='rnd', choices=['rnd', 'hist_nre', 'induc_nre', 'rp_ns'],
                        help='Strategy for the negative edge sampling.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv
