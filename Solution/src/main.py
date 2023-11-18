import os
import copy
import time
import pandas as pd
import arguments
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from model.evaluate import Eval_Regre, Eval_Class, rec_evalution
from preprocess.representation_learning import RepresentationLearning
from model.baseline import Baseline, MLP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def model_tra_eval(train_set, train_label, val_set, val_label, test_set, test_label,
                   target, co_col, ca_col, task_name, nor_std, param_fit, param_init={}, seed=0):
    train_set: pd.DataFrame
    train_label: pd.DataFrame
    val_set: pd.DataFrame
    val_label: pd.DataFrame
    test_set: pd.DataFrame
    val_label: pd.DataFrame
    target: dict

    metric_all = pd.DataFrame([])
    imp_feat_ = pd.DataFrame([])
    name_model_list = [
        ['MLP', 'Base'],
    ]
    for name_model in name_model_list:
        model_method = eval(name_model[0])(name_model[1],
                   train_set, train_label,
                   val_set, val_label,
                   test_set, test_label, target, co_col, ca_col, task_name, seed,
                   param_init,
                   param_fit=param_fit
                                           )
        pred_tra, pred_val, pred_test, model = model_method.grid_fit_pred()
        imp_feat_ = model_method.imp_feat()
        if 'class' in task_name:
            metric = Eval_Class
        else:
            metric = Eval_Regre
        for index_, values in pred_tra.iteritems():
            metric_tra = metric(train_label.loc[:, index_].values.reshape(-1, 1),
                                pred_tra.loc[:, index_].values.reshape(-1, 1), train_set,
                                'train', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_val = metric(val_label.loc[:, index_].values.reshape(-1, 1),
                                pred_val.loc[:, index_].values.reshape(-1, 1), val_set,
                                'val', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_test = metric(test_label.loc[:, index_].values.reshape(-1, 1),
                                 pred_test.loc[:, index_].values.reshape(-1, 1), test_set,
                                 'test', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_single = pd.concat([metric_test, metric_val, metric_tra], axis=1)
            metric_all = pd.concat([metric_all, metric_single], axis=0)
    return metric_all, pd.DataFrame(pred_tra, columns=train_label.columns, index=train_label.index), \
               pd.DataFrame(pred_test, columns=test_label.columns, index=test_label.index), imp_feat_, model


def run(train_data, test_data, target, args, trial) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=args.seed, shuffle=False)

    metric_df_all = pd.DataFrame([])
    pred_train_df_all = pd.DataFrame([])
    pred_test_df_all = pd.DataFrame([])
    metric_AL_AllFlod = pd.DataFrame([])
    kf = KFold(n_splits=args.n_splits)
    for k, (train_index, val_index) in enumerate(kf.split(train_set)):
        print(f'KFlod {k}')
        metric_all_fold = pd.DataFrame([])
        train_set_cv = train_set.iloc[train_index]
        val_set_cv = train_set.iloc[val_index]
        test_set_cv = copy.deepcopy(test_set)

        dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, None, seed=args.seed,
                               flag_label_onehot=False,
                               flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                               flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                               flag_feat_emb=False, flag_RUS=False, flag_confusion=False, flaq_save=False)
        if args.Flag_DataPreprocessing:
            train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()

        col_drop = dp.features_ex(train_set_cv, args.dict_ManualRatio['drop_NanRatio']) # Drop useless features (high deletion rate, small variance, etc.)
        train_set_cv.drop(columns=col_drop, inplace=True)
        col_ = train_set_cv.columns
        val_set_cv = val_set_cv[col_]
        test_set_cv = test_set_cv[col_]
        ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
        co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

        train_label = train_set_cv[[x for x in target.values()]]
        val_label = val_set_cv[[x for x in target.values()]]
        test_label = test_set_cv[[x for x in target.values()]]
        train_x = train_set_cv.drop(columns=target.values())
        val_x = val_set_cv.drop(columns=target.values())
        test_x = test_set_cv.drop(columns=target.values())

        print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')

        # missing values imputation start
        mvi = MVI(train_x.shape[1], co_col, ca_col, args.task_name, target, args.seed, method=args.method_mvi)
        train_x_filled = mvi.fit_transform(train_x)
        val_x_filled = mvi.transform(val_x)
        test_x_filled = mvi.transform(test_x)
        # missing values imputation End

        # Dimension reduction Start
        metric_all_ = pd.DataFrame([])
        print(f'method_name_RL {args.method_name_RL}')
        represent = RepresentationLearning(train_x_filled.shape[1], args.method_name_RL)
        train_x_rec, train_x_hidden = represent.fit_transform(train_x_filled, args.RL_para, val_x_filled)
        val_x_rec, val_x_hidden = represent.transform(val_x_filled)
        text_x_rec, test_x_hidden = represent.transform(test_x_filled)
        MetricRec_tra= rec_evalution(train_x_filled, train_x_rec, args.method_name_RL, 'train')
        metric_all_ = pd.concat([metric_all_, MetricRec_tra], axis=1)
        MetricRec_val= rec_evalution(val_x_filled, val_x_rec, args.method_name_RL, 'val')
        metric_all_ = pd.concat([metric_all_, MetricRec_val], axis=1)
        MetricRec_test = rec_evalution(test_x_filled, text_x_rec, args.method_name_RL, 'test')
        metric_all_ = pd.concat([metric_all_, MetricRec_test], axis=1)
        # Dimension reduction End

        train_x_ger = train_x_hidden
        val_x_ger = val_x_hidden
        test_x_ger = test_x_hidden

        metric, pred_train_df, pred_test_df, imp_feat, model = model_tra_eval(train_x_ger, train_label,
                                                                              val_x_ger, val_label,
                                                                              test_x_ger, test_label,
                                                                              target, co_col, ca_col,
                                                                              args.task_name, nor,
                                                                              param_fit=args.classifier_para,
                                                                              seed=args.seed)

        metric.index = metric_all_.index
        metric_all_ = pd.concat([metric_all_, metric], axis=1)
        print(metric['test auc_'])
        metric_all_fold = pd.concat([metric_all_fold, metric_all_], axis=1)
        metric_df_all = pd.concat([metric_df_all, metric_all_fold], axis=0)
    # metric_df_all.to_csv(f'./{args.task_name}_RL_metric_{args.method_name_RL}.csv')
    return metric_df_all, pred_train_df_all, pred_test_df_all, metric_AL_AllFlod


if __name__ == "__main__":
    args = arguments.get_args()

    test_prediction_all = pd.DataFrame([])
    train_prediction_all = pd.DataFrame([])
    history_df_all = pd.DataFrame([])
    metric_df_all = pd.DataFrame([])
    metric_AL_Allrun = pd.DataFrame([])

    for trial in range(args.nrun):
        print('rnum : {}'.format(trial))
        args.seed = (trial * 55) % 2022 + 1 # a different random seed for each run

        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name, args.seed)

        # run model
        # input: train_data
        # output: metric, train_prediction, test_prediction
        metric_df, train_prediction, test_prediction, metric_AL_AllFlod = run(train_data, test_data, target, args, trial)

        metric_df_all = pd.concat([metric_df_all, metric_df], axis=0)
        test_prediction_all = pd.concat([test_prediction_all, test_prediction], axis=1)
        train_prediction_all = pd.concat([train_prediction_all, train_prediction], axis=1)
        metric_AL_Allrun = pd.concat([metric_AL_Allrun, metric_AL_AllFlod], axis=1)

        local_time = time.strftime("%m_%d_%H_%M", time.localtime())
    metric_df_all.to_csv(f'./{args.task_name}_{local_time}.csv', index_label=['index'])

    # print metric
    metric_df_all['model'] = metric_df_all.index
    metric_mean = metric_df_all.groupby('model').mean()
    metric_mean_test = metric_mean.filter(regex=r'test')
    metric_mean_val = metric_mean.filter(regex=r'val')
    print(metric_mean)
    print('mean test auc_: ', metric_mean_test['test auc_'])
    print('mean val auc_: ', metric_mean_val['val auc_'])
    pass
pass
