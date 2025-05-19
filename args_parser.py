"""
Parsing arguments of different scripts
Initial code from the DBG github; https://github.com/fpour/DGB/blob/main/
"""

import argparse
import sys


def parse_args():

    # argument passing
    parser = argparse.ArgumentParser(description='Base3: ' \
    'An interpolation-based baseline for dynamic link prediction.')
    
    # parameters for data construction 
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='test ratio.')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='whether using different new nodes for validation and test.')
    
    # experiment parameters
    parser.add_argument('-d', '--data', type=str, default='tgbl-wiki', help='name of the network dataset.')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs.')

    # parameters for model construction
    # ---------------------------
    # EdgeBank
    parser.add_argument('--mem_mode', type=str, default='unlim_mem', help='How memory of EdgeBank works.',
                        choices=['unlim_mem', 'repeat_freq', 'time_window'])
    parser.add_argument('--w_mode', type=str, default='fixed',
                        help='In time interval-based memory, how to select time window size.',
                        choices=['fixed', 'avg_reoccur'])
    parser.add_argument('--mem_span', type=float, default=0.01,
                        help='Memory span for EdgeBank.')
    # PopTrack
    parser.add_argument('--k_val', type=int, default=100, 
                        help='K value for PopTrack.')
    # t-CoMem
    parser.add_argument('--co_occurence_weight', type=float, default=1.0,
                        help="Co-occurence weight parameter for t-CoMem")
    
    # Interpolation strategy
    parser.add_argument('--method', type=str, default='EB_conf', choices=['EB_conf', 'uniform', 'multi_conf', 't-CoMem', 'EdgeBank', 'PopTrack']
                        help='Strategy for interpolation weighting.'
                        )

    # parameters for negative sampling strategy
    parser.add_argument('--neg_sample', type=str, default='rnd', choices=['rnd', 'hist_nre', 'induc_nre', 'rp_ns'],
                        help='Strategy for the negative edge sampling.')
    

    try:
        args = parser.parse_args()
        print("Info: Arguments:\n", args)
    except:
        parser.print_help()
        sys.exit()

    return args