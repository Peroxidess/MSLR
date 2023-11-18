import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='adult_class', # Classification task needs to add "_class"
                        help='{pima} {thyroid} {arrhythmia}')
    parser.add_argument('--nrun', type=int, default=1,
                        help='total number of runs[default: 1]')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='cross-validation fold, 1 refer not CV [default: 1]')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='proportion of test sets divided from training set, '
                             '0 refer dataset has its own test set [default: 0.2]')
    parser.add_argument('--val_ratio', type=float, default=0.,
                        help='proportion of test sets divided from training set [default: 0.2]')
    parser.add_argument('--method_mvi', type=str, default='s',
                        help='missing values imputation method [default: "s"]')
    parser.add_argument('--dict_ManualRatio', type=dict, default={'drop_NanRatio': 1},
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--missing_ratio', type=float, default=1.,
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--Flag_LoadMetric', type=bool, default=False, metavar='N',
                        help='overload metric training before[default: False]')
    parser.add_argument('--Flag_DataPreprocessing', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_MVI', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_downstream', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--method_name_RL', type=str, default='MSLR', metavar='N',
                        help='[default: MSLR]')

    para_MSLR = {'lr': 5e-3, 'weight_decay': 3e-3, 'batch_size': 256, 'epoch': 1.1, 'epoch_alpha': 2.5, 'epoch_max': 65, 'alpha': 0.05, 'beta': 0.05}
    para_DAE = {'lr': 0.01, 'weight_decay': 2e-4, 'batch_size': 256, 'epochs': 200, 'noise_a': 1e-2, 'alpha': 0.05, 'bin_list': [61, 41, 29, 11, 2]}
    para_ = {'para_DAE': para_DAE, 'para_MSLR': para_MSLR}
    parser.add_argument('--RL_para', type=dict, default=para_, metavar='N')

    para_classifier = {'batch_size': 256, 'epoch': 1400, 'lr': 5e-4, 'weight_decay': 1e-4}
    parser.add_argument('--classifier_para', type=dict, default=para_classifier, metavar='N')

    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    args = parser.parse_args()

    return args
