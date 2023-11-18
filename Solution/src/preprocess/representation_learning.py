import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model.ae import CAutoEncoder, AutoEncoder
import torch
import torch.optim as optim
import torch.utils.data as Data


class RepresentationLearning:
    def __init__(self, dim_input, method='MSLR', seed=2022):
        self.method = method
        self.model = ZSR(dim_input, name_method=method, seed=seed)

    def fit_transform(self, train_x, para_, val_x=None):
        col_ = train_x.filter(regex=r'dense|sparse').columns
        col_sp = train_x.columns.drop(col_)
        data_x, data_z = self.model.fit_transform(train_x[col_], para_['para_MSLR'], para_['para_DAE'], val_x)
        data_x_all = pd.concat([data_x, train_x[col_sp]], axis=1)
        data_z_all = pd.concat([data_z, train_x[col_sp]], axis=1)
        return data_x_all, data_z_all

    def transform(self, x):
        col_ = x.filter(regex=r'dense|sparse').columns
        col_sp = x.columns.drop(col_)
        data_rec, z_ = self.model.transform(x[col_])
        data_x_all = pd.concat([data_rec, x[col_sp]], axis=1)
        data_z_all = pd.concat([z_, x[col_sp]], axis=1)
        return data_x_all, data_z_all


class Cluster:
    def __init__(self, seed=2022, n_clusters=1000):
        self.model = KMeans(n_clusters=n_clusters)

    def fit_transform(self, train_x):
        self.model.fit_transform(train_x)
        centers_ = self.model.cluster_centers_
        centers_df = pd.DataFrame(centers_, columns=train_x.columns)
        centers_df['cluster_y'] = list(range(centers_.shape[0]))
        data_x = pd.DataFrame([])
        for sample_cluster in self.model.labels_:
            df_new = centers_df[centers_df['cluster_y'] == sample_cluster]
            data_x = pd.concat([data_x, df_new], axis=0)
        data_x.index = train_x.index
        data_ = data_x.drop(columns=['cluster_y'])
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(train_x.values, data_.values)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        return data_

    def transform(self, x):
        labels_ = self.model.predict(x)
        centers_ = self.model.cluster_centers_
        centers_df = pd.DataFrame(centers_, columns=x.columns)
        centers_df['cluster_y'] = list(range(centers_.shape[0]))
        data_x = pd.DataFrame([])
        for sample_cluster in labels_:
            df_new = centers_df[centers_df['cluster_y'] == sample_cluster]
            data_x = pd.concat([data_x, df_new], axis=0)
        data_x.index = x.index
        data_ = data_x.drop(columns=['cluster_y'])
        mse_re, mae_re, mape_re, r2_re, near_mae_re, x_new_output = metric_rec(x.values, data_.values)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        return data_


class MSLR():
    def __init__(self, dim_input, flag_model_load=False, name_method='MSLR', seed=2022):
        self.seed = seed
        self.dim_input = dim_input
        self.flag_model_load = flag_model_load
        self.name_method = name_method
        if flag_model_load == True:
            self.model = torch.load('./MSLR.pth')
        else:
            self.model = CAutoEncoder(dim_input, z_dim=max(dim_input//8, 3))
        self.ae = AutoEncoder(dim_input, z_dim=max(dim_input//8, 3))

    def input_process(self, data, data_rec):
        data_dataloader = self.MyDataset(data, data_rec)
        return data_dataloader

    def fit_transform(self, x, para, para_DAE, val_x=None):
        noise_s = np.random.normal(size=x.shape)
        if 'noise_none' in self.name_method:
            x_noise = x
        else:
            x_noise = x + noise_s
        x_noise = self.preprocessing(x_noise, val_x, para_DAE)
        if not self.flag_model_load:
            x_lr_list = []
            step = 1
            bin_list = para['bin_list']
            for bs in bin_list:
                if 'cluster' in self.name_method:
                    x_lr = self.scale_fuzz_cluster(x_noise, bs, para['alpha'])
                else:
                    x_lr = self.scale_fuzz(x_noise, bs, para['alpha'])
                x_lr_list.append(x_lr)
            x_lr_list.append(x_noise)
            len_x_lr = len(x_lr_list)
            count_all = 0
            epoch = para['epoch']
            for index_ in range(0, len_x_lr - 1, step): # x_lr_list: lr--->hr
                x_lr_re = x_lr_list[index_]
                x_hr_re = x_lr_list[index_ + 1]
                data_dataloader = self.input_process(x_lr_re, x_hr_re)
                optim_ = optim.Adam(self.model.parameters(), lr=para['lr'], weight_decay=para['weight_decay'])
                data_loader = Data.DataLoader(data_dataloader, batch_size=para['batch_size'], worker_init_fn=np.random.seed(self.seed))
                epoch = epoch ** para['epoch_alpha']
                epoch_ = min(math.ceil(epoch), para['epoch_max'])
                for iter_count in range(epoch_):
                    count_all += 1
                    total_vae_loss_ = 0
                    self.model.train()
                    for data_lr, data_hr in data_loader:
                        optim_.zero_grad()
                        x_rec, z = self.model(data_lr, torch.tensor(1/(index_+1)))
                        rec_loss = self.model.loss(data_hr, x_rec)
                        rec_loss.backward()
                        optim_.step()
                    total_vae_loss_ += rec_loss.detach().numpy()
                    if iter_count % 5 == 0:
                        self.model.eval()
                        x_hr_raw = val_x.values
                        x_tensor = torch.Tensor(val_x.values)
                        x_hr_rec, z_hr = self.model(x_tensor, torch.tensor(0))
                        x_np = x_hr_rec.detach().numpy()
                        mse_re_eval, mae_re_eval, mape_re_eval, r2_re_eval, near_mae_re_eval, x_new_output_eval = metric_rec(x_hr_raw, x_np)
                        print(f'index_{index_} count{iter_count} val total_vae_loss_ {total_vae_loss_} mse_re {mse_re_eval.round(4)} mae_re {mae_re_eval.round(4)} mape_re {mape_re_eval}')

                        torch.save(self.model, 'MSLR.pth')

        self.model.eval()
        x_ = x_noise
        x_tensor = torch.Tensor(x_.values)
        x_hr, z_hr = self.model(x_tensor, torch.tensor(0))
        self.model.train()
        x_np = x_hr.detach().numpy()
        z_np = z_hr.detach().numpy()
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    def transform(self, x):
        x_tensor = torch.Tensor(x.values)
        x_hr, z_hr = self.model(x_tensor, torch.tensor(0))
        x_np = x_hr.detach().numpy()
        z_np = z_hr.detach().numpy()
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    def scale_fuzz_cluster(self, data, bins_stride=7, alpha=0.05):
        model = Cluster(n_clusters=bins_stride)
        fuzzy_data = model.fit_transform(data)
        fuzzy_data_ = fuzzy_data * (1 - alpha) + data * alpha
        return pd.DataFrame(fuzzy_data_, index=data.index, columns=data.columns)

    def scale_fuzz(self, data, bins_stride=7, alpha=0.05):
        data_fuzz = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col_weight = np.histogram(data_col, bins=bins, weights=data_col)[0]
            histogram_data_col = np.histogram(data_col, bins=bins)

            histogram_per = pd.Series(histogram_data_col[1])
            histogram_post = pd.Series(histogram_data_col[1]).shift()
            histogram_ave_ = (histogram_per + histogram_post)/2
            histogram_ave_ = histogram_ave_.dropna(axis=0, how='any')

            data_col_bins = pd.cut(data_col, bins=bins, labels=histogram_ave_, ordered=False)
            data_col_bins_df = pd.Series(data_col_bins)
            set_data_col_bins = data_col_bins_df.unique()
            if data_fuzz is None:
                data_fuzz = data_col_bins_df
            else:
                data_fuzz = np.vstack((data_fuzz, data_col_bins_df))
        data_fuzz = np.transpose(data_fuzz)
        data_fuzz_ = data_fuzz * (1 - 'alpha') + data.values * alpha
        data_fuzz_df = pd.DataFrame(data_fuzz_, index=data.index, columns=data.columns)
        return data_fuzz_df

    def preprocessing(self, x, val_x, para):
        data_dataloader = self.input_process(x, x)
        optim_ = optim.Adam(self.model.parameters(), lr=para['lr'], weight_decay=para['weight_decay'])
        data_loader = Data.DataLoader(data_dataloader, batch_size=para['batch_size'], worker_init_fn=np.random.seed(self.seed))
        for epoch_ in range(para['epochs']):
            total_ae_loss_ = 0
            self.ae.train()
            for data_x, data_rec in data_loader:
                optim_.zero_grad()
                noise = torch.randn(size=data_x.shape)
                x_rec, z = self.ae(data_x + para['noise_a'] * noise)
                rec_ae_loss = self.model.loss(data_rec, x_rec)
                rec_ae_loss.backward()
                optim_.step()
            total_ae_loss_ += rec_ae_loss.detach().numpy()

            self.ae.eval()
            val_x_rec, val_z = self.ae(torch.Tensor(val_x.values))
            rec_val_loss = self.model.loss(torch.Tensor(val_x.values), val_x_rec)

        x_rec_tra, _, _, _ = self.ae(torch.tensor(x.values, dtype=torch.float32))
        x_rec_tra = pd.DataFrame(x_rec_tra.detach().numpy(), index=x.index, columns=x.columns)
        x_rec_sparse = x_rec_tra.filter(regex=r'sparse')
        x_sparse = x.filter(regex=r'sparse')
        x_noise = copy.deepcopy(x)
        x_noise[x_noise.filter(regex=r'sparse').columns] = para['beta'] * x_rec_sparse + (1 - para['beta']) * x_sparse
        return x_noise

    class MyDataset(Data.Dataset):
        def __init__(self,
                     data,
                     data_rec,
                     random_seed=0):
            self.rnd = np.random.RandomState(random_seed)
            data = data.astype('float32')
            data_rec = data_rec.astype('float32')

            list_data = []
            for index_, values_ in data.iterrows():
                x = data.loc[index_].values
                x_rec = data_rec.loc[index_].values
                list_data.append((x, x_rec))

            self.shape = data.shape
            self.data = list_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data = self.data[idx]
            return data


def metric_rec(data, data_rec):
    mms = MinMaxScaler(feature_range=(0.1, 1))
    data_nor = mms.fit_transform(data)
    data_rec_nor = mms.transform(data_rec)
    mse_re = mean_squared_error(data, data_rec)
    mae_re = mean_absolute_error(data, data_rec)
    mape_re = mean_absolute_percentage_error(data_nor, data_rec_nor)
    r2_re = r2_score(data, data_rec)
    return mse_re, mae_re, mape_re, r2_re, None, None
