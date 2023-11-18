import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, multilabel_confusion_matrix, \
    precision_score, recall_score, f1_score, r2_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, \
    classification_report, matthews_corrcoef, fowlkes_mallows_score
    # , mean_absolute_percentage_error
import seaborn as sns
plt.rc('font', family='Times New Roman')


def mean_absolute_percentage_error(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class Evaluate:
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        self.data = copy.deepcopy(data)
        self.eval_type = eval_type
        self.task_name = task_name
        self.name_clf = name_clf
        self.nor_flag = nor_flag
        self.nor_std = nor_std
        self.true, self.pred = self.nor_inver(true, pred)

    def nor_inver(self, true, pred):
        if self.nor_flag:
            data_inver = []
            for labelorpred in [true, pred]:
                self.data.loc[:, 'label0'] = labelorpred
                data_inver.append(self.nor_std.inverse_transform(self.data))
            true = data_inver[0][:, -1].reshape(-1, 1)
            pred = data_inver[1][:, -1].reshape(-1, 1)
        return true, pred


class Eval_Class(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)
        if pred.shape[1] > 1:
            self.pred = np.argmax(pred, axis=1)
        else:
            self.pred = pred
        if len(true.shape) < 2 or true.shape[1] > 1:
            self.true = np.argmax(true, axis=1)
        else:
            self.true = true

    def eval(self):
        if np.unique(self.true).shape[0] < 3:
            auc_ = 0
            if 'class' in self.task_name:
                fpr, tpr, thresholds = roc_curve(self.true, self.pred)
                auc_ = auc(fpr, tpr)
                youden = (1 - fpr) + tpr - 1
                index_max = np.argmax(youden)
                threshold_max = thresholds[index_max]
                if threshold_max > 1:
                    threshold_max = 0.5
        else:
            auc_ = 0
            threshold_max = 0.5
        pred = np.array([(lambda x: 0 if x < threshold_max else 1)(i) for i in self.pred])
        acc = accuracy_score(self.true, pred)
        acc_balanced = balanced_accuracy_score(self.true, pred)
        pre_weighted = precision_score(self.true, pred, average='weighted')
        pre_macro = precision_score(self.true, pred, average='macro')
        recall_weighted = recall_score(self.true, pred, average='weighted')
        recall_macro = recall_score(self.true, pred, average='macro')
        f1_weighted = f1_score(self.true, pred, average='weighted')
        f1_macro = f1_score(self.true, pred, average='macro')
        mcc = matthews_corrcoef(self.true, pred)
        fms = fowlkes_mallows_score(self.true.reshape(-1, ), pred.reshape(-1, ))
        con_mat = confusion_matrix(self.true, pred)
        metric_dict = dict(zip([
                                '{} acc'.format(self.eval_type),
                                '{} acc_balanced'.format(self.eval_type),
                                '{} pre_weighted'.format(self.eval_type),
                                '{} pre_macro'.format(self.eval_type),
                                '{} recall_weighted'.format(self.eval_type),
                                '{} recall_macro'.format(self.eval_type),
                                '{} f1_weighted'.format(self.eval_type), '{} f1_macro'.format(self.eval_type),
                                '{} auc_'.format(self.eval_type),
                                '{} mcc'.format(self.eval_type),
                                '{} fms'.format(self.eval_type),
                                ],
                               [acc, acc_balanced, pre_weighted, pre_macro, recall_weighted, recall_macro, f1_weighted,
                                f1_macro, auc_, mcc, fms]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


class Eval_Regre(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)

    def eval(self):
        r2 = r2_score(self.true, self.pred)
        mse = mean_squared_error(self.true, self.pred)
        mae = mean_absolute_error(self.true, self.pred)
        mape = mean_absolute_percentage_error(self.true, self.pred)
        metric_dict = dict(zip(['{} r2'.format(self.eval_type), '{} mae'.format(self.eval_type),
                                '{} mse'.format(self.eval_type), '{} mape'.format(self.eval_type)],
                               [r2, mae, mse, mape]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


def rec_evalution(data_raw, data_rec, name_clf=0, name_metric='test'):
    mms = MinMaxScaler(feature_range=(0.1, 10))
    data_nor = mms.fit_transform(data_raw)
    data_rec_nor = mms.transform(data_rec)
    mse_re = mean_squared_error(data_raw, data_rec)
    rmse_re = np.sqrt(mse_re)
    mae_re = mean_absolute_error(data_raw, data_rec)
    mape_re = mean_absolute_percentage_error(data_nor, data_rec_nor)
    r2_re = r2_score(data_raw, data_rec)
    metric_dict = dict(zip([f'{name_metric} mae', f'{name_metric} mse', f'{name_metric} rmse', f'{name_metric} mape', f'{name_metric} r2'],
                           [mae_re, mse_re, rmse_re, mape_re, r2_re]))
    metric_df = pd.DataFrame([metric_dict], index=[name_clf])
    print(f'{name_metric} metric{metric_df}')
    return metric_df
