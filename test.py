import numpy as np
from args_parser import parse_args_edge_bank
from edge_sampler import RandEdgeSampler, RandEdgeSampler_adversarial
from load_data import Data, get_data
from edge_bank_baseline import *
from pathlib import Path
from proofofconcept import PopTrack  # Import PopTrack


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

    # Initialize PopTrack
    poptrack = PopTrack()

    # executing different runs
    for i_run in range(n_runs):
        print("INFO:root:****************************************")
        for k, v in args.items():
            print("INFO:root:{}: {}".format(k, v))
        print ("INFO:root:Run: {}".format(i_run))
        start_time_run = time.time()
        inherent_ap, inherent_auc_roc, avg_measures_dict = edge_bank_link_pred_batch(
            tr_val_data, test_data, test_rand_sampler, args
        )
        print('INFO:root:Test statistics: Old nodes -- auc_inherent: {}'.format(inherent_auc_roc))
        print('INFO:root:Test statistics: Old nodes -- ap_inherent: {}'.format(inherent_ap))
        # extra performance measures
        # Note: just prints out for the Test set! in transductive setting
        for measure_name, measure_value in avg_measures_dict.items():
            print ('INFO:root:Test statistics: Old nodes -- {}: {}'.format(measure_name, measure_value))

        elapse_time = time.time() - start_time_run
        print('INFO:root:EdgeBank: Run: {}, Elapsed time: {}'.format(i_run, elapse_time))
        print('INFO:root:****************************************')

    print("===========================================================================")


if __name__ == '__main__':
    main()