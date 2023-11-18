import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as Data
from preprocess.get_dataset import MyDataset


class Baseline():
    def __init__(self, model_name, train_set, train_label, val_set, val_label, test_set, test_label, target,
                 co_col, ca_col, task_name, seed, param_init={}, param_fit={}):
        self.target = target
        self.co_col = co_col
        self.ca_col = ca_col
        self.task_name = task_name
        self.seed = seed
        self.train_set, self.train_label = self.input_process(train_set, train_label)
        self.val_set, self.val_label = self.input_process(val_set, val_label)
        self.test_set, self.test_label = self.input_process(test_set, test_label)
        if 'class' in task_name:
            self.output_dim = len(np.unique(self.train_label.values))
        else:
            self.output_dim = 1
        self.model = self.model_bulid(model_name, param_init)
        self.param_grid = {}
        self.param_fit = param_fit

    def input_process(self, input_x, input_y) -> (pd.DataFrame, pd.DataFrame):
        input_x: pd.DataFrame
        input_y: pd.DataFrame
        input_y = input_y[[self.target['label1']]]
        return input_x, input_y

    def model_bulid(self, model_name, param_init):
        model = eval(model_name)(**param_init)
        return model

    def grid_fit_pred(self):
        clf = GridSearchCV(self.model, self.param_grid)
        clf.fit(self.train_set, self.train_label, **self.param_fit)
        print('Best parameters found by grid search are:', clf.best_params_)
        self.model = clf.best_estimator_
        self.model.fit(self.train_set, self.train_label, **self.param_fit)
        pred_tra = self.model.predict(self.train_set).reshape(-1, self.train_label.shape[1])
        pred_val = self.model.predict(self.val_set).reshape(-1, self.train_label.shape[1])
        pred_test = self.model.predict(self.test_set).reshape(-1, self.train_label.shape[1])
        pred_tra_df = pd.DataFrame(pred_tra, index=self.train_label.index, columns=self.train_label.columns)
        pred_val_df = pd.DataFrame(pred_val, index=self.val_label.index, columns=self.val_label.columns)
        pred_test_df = pd.DataFrame(pred_test, index=self.test_label.index, columns=self.test_label.columns)
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        try:
            feat_dict = dict(zip(self.train_set.columns, self.model.coef_.reshape(self.train_set.columns.shape[0], -1)))
            if self.model.coef_.shape[0] == 1:
                feat_dict = [feat_dict]
        except:
            feat_dict = dict(zip(self.train_set.columns, self.model.feature_importances_.reshape(self.train_set.columns.shape[0], -1)))
            if self.model.feature_importances_.shape[0] == 1:
                feat_dict = [feat_dict]
        else:
            pass
        return pd.DataFrame.from_dict(feat_dict, orient='columns').T.sort_values(by=0, ascending=False)


class MLP():
    def __init__(self, name_model, train_set, train_label, val_set, val_label, test_set, test_label, target, co_col, ca_col, task_name, seed, param_init={}, param_fit={}):
        self.train_set = train_set
        self.train_label = train_label
        self.val_set = val_set
        self.val_label = val_label
        self.test_set = test_set
        self.test_label = test_label
        self.param_fit = param_fit
        self.target = target
        self.seed = seed
        self.model = self.model_bulid(param_init)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def model_bulid(self, param_init):
        model = self.MLP_base(dim_input=self.train_set.shape[1], seed=self.seed, **param_init)
        return model

    def input_process(self, input_x, input_y):
        data_labeled = MyDataset(input_x, input_y)
        return data_labeled

    def grid_fit_pred(self, batch_size=self.param_fit['batch_size']):
        train_data_set = self.input_process(self.train_set, self.train_label.iloc[:, 0])
        optim_ = optim.Adam(self.model.parameters(), lr=self.param_fit['lr'], weight_decay=self.param_fit['weight_decay'])
        train_data_loader = Data.DataLoader(train_data_set, batch_size=batch_size, worker_init_fn=np.random.seed(self.seed))
        total_loss_list = []
        best_val_loss, es = 10., 0
        for iter_count in range(self.param_fit['epoch']):
            total_loss_ = 0.
            for train_data_batch, train_y_batch in train_data_loader:
                self.model.train()
                optim_.zero_grad()
                y, features = self.model(train_data_batch)
                label_onehot = torch.ones(train_y_batch.shape[0],
                            2
                            ).scatter_(1, train_y_batch.view(-1, 1), 0)
                CELoss = self.model.loss(y, label_onehot)
                total_loss = CELoss
                total_loss.backward()
                optim_.step()
                total_loss_ += total_loss.detach().numpy()

            self.model.eval()
            pred_val = self.model.predict(self.val_set.values)
            val_Loss = self.model.loss(torch.Tensor(pred_val[:, 0]),
                                        torch.Tensor(self.val_label.iloc[:, 0].astype('float32').values.reshape(-1, )))
            if val_Loss < best_val_loss:
                best_val_loss = val_Loss
                es = 0
            else:
                es += 1
            if iter_count > 150 and es > 15:
                print(f'Early stopping with epoch{iter_count} val loss {best_val_loss}')
                break

            total_loss_list.append(total_loss_)
        self.model.eval()
        pred_tra = self.model.predict(self.train_set.values)
        pred_val = self.model.predict(self.val_set.values)
        pred_test = self.model.predict(self.test_set.values)
        pred_tra_df = pd.DataFrame(pred_tra[:, 0], index=self.train_label.index, columns=[self.target['label1']])
        pred_val_df = pd.DataFrame(pred_val[:, 0], index=self.val_label.index, columns=[self.target['label1']])
        pred_test_df = pd.DataFrame(pred_test[:, 0], index=self.test_label.index, columns=[self.target['label1']])
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        return None

    class MLP_base(nn.Module):
        def __init__(self, dim_input, seed=2022):
            super(MLP.MLP_base, self).__init__()
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.dim_input = dim_input
            self.num_classes = 2
            self.dim_list = [dim_input, max(dim_input // 1, 4), max(dim_input // 2, 4), max(dim_input // 3, 3), max(dim_input // 4, 2), self.num_classes]
            self.fc1 = nn.Linear(self.dim_list[0], self.dim_list[1])
            self.fc2 = nn.Linear(self.dim_list[1], self.dim_list[2])
            self.fc3 = nn.Linear(self.dim_list[2], self.dim_list[4])
            self.fc4 = nn.Linear(self.dim_list[4], self.dim_list[4])
            self.linear = nn.Linear(self.dim_list[4], self.dim_list[-1])

            self.list_feat = [self.dim_list[1], self.dim_list[2], self.dim_list[4]]
            self.weight_init()

        def weight_init(self):
            for key_, block in self._modules.items():
                if key_ == 'loss_pred':
                    continue
                try:
                    for m in self._modules[key_]:
                        self.kaiming_init(m)
                except:
                    self.kaiming_init(block)

        def forward(self, x):
            out1 = self.fc1(x)
            out1_act = F.gelu(out1)
            out2 = self.fc2(out1_act)
            out2_act = F.gelu(out2)
            out3 = self.fc3(out2_act)
            out3_act = F.gelu(out3)
            out4 = self.fc4(out3_act)
            prob = self.linear(out4)
            prob = torch.sigmoid(prob)
            return prob, [out1, out2, out3, out4]

        def loss(self, x, y):
            loss_CE = nn.BCELoss()
            CE_loss = loss_CE(x, y)
            loss_ = CE_loss
            return loss_

        def predict(self, data):
            pred = self.forward(torch.Tensor(data))
            prob = pred[0].detach().numpy()
            return prob

        @staticmethod
        def kaiming_init(m):
            if isinstance(m, (nn.Linear)):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)