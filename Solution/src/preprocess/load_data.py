import numpy as np
import pandas as pd


def data_load(task_name, seed):
    test_data = pd.DataFrame([])
    if 'thyroid' in task_name:
        file_name_tra = '../DataSet/UCI_thyroid/ann-train.csv'
        file_name_test = '../DataSet/UCI_thyroid/ann-test.csv'
        data = pd.read_csv(file_name_tra, sep=' ', header=None).dropna(how='all', axis=1)
        data_test = pd.read_csv(file_name_test, sep=' ', header=None).dropna(how='all', axis=1)
        train_data = data.rename(columns={data.shape[1] - 1: 'label1'})
        test_data = data_test.rename(columns={data.shape[1] - 1: 'label1'})
        xx = train_data['label1'].value_counts()
        train_data['label1'][train_data['label1'] == 3] = 0
        test_data['label1'][test_data['label1'] == 3] = 0
        train_data['label1'][(train_data['label1'] == 2) | (train_data['label1'] == 1)] = 1
        test_data['label1'][(test_data['label1'] == 2) | (test_data['label1'] == 1)] = 1
        target_dict = {'label1': 'label1'}
    elif 'diabetic' in task_name:
        file_name_tra = '../DataSet/UCI_dataset_diabetes/diabetic_data.csv'
        data = pd.read_csv(file_name_tra, index_col=[0]).dropna(how='all', axis=1)
        data_ = data.replace({'?': np.nan})
        data_.rename(columns={'diabetesMed': 'label1'}, inplace=True)
        data_['label1'] = data_['label1'].replace({'Yes': 1, 'No': 0})
        for col_ in ['age', 'weight']:
            data_[col_] = data_[col_].replace({'>200': '200-200'})
            list_age = data_[col_].str.replace(r'\[|\)', '').str.split('-', expand=True).astype(float)
            data_[col_] = (list_age[0] + list_age[1]) / 2
        target_dict = {'label1': 'label1'}
        data_dup = data_.drop_duplicates(subset=['patient_nbr'])
        train_data = data_dup.sample(frac=0.01)
        # train_data = data_dup
        xx = train_data['label1'].value_counts()
        col_drop = data_dup.filter(regex=r'diag|change|specialty|admission|discharge|time|num|patient_nbr|readmitted|payer_code').columns
        train_data.drop(columns=col_drop, inplace=True)
        pass
    elif 'arrhythmia' in task_name:
        file_name_tra = '../DataSet/UCI_arrhythmia/arrhythmia.csv'
        data = pd.read_csv(file_name_tra, header=None).dropna(how='all', axis=1)
        train_data = data.rename(columns={data.shape[1] - 1: 'label1'})
        unknow_index = train_data['label1'][train_data['label1'] == 16].index
        train_data.drop(index=unknow_index, inplace=True)
        train_data['label1'][train_data['label1'] != 1] = 0
        train_data[train_data == '?'] = np.nan
        target_dict = {'label1': 'label1'}
    elif 'pima' in task_name:
        file_name_tra = '../DataSet/PimaIndiansdiabetes/PimaIndiansdiabetes.csv'
        data = pd.read_csv(file_name_tra)
        train_data = data.rename(columns={'Outcome': 'label1'})
        target_dict = {'label1': 'label1'}
    elif 'adult' in task_name:
        file_name_tra = '../DataSet/UCI_adult/adult_train.csv'
        file_name_test = '../DataSet/UCI_adult/adult_test.csv'
        data_tra = pd.read_csv(file_name_tra, header=None)
        data_test = pd.read_csv(file_name_test, header=None)
        data_tra = data_tra.replace({'?': np.nan})
        data_test = data_test.replace({'?': np.nan})
        train_data = data_tra.rename(columns={data_tra.shape[1] - 1: 'label1'})
        test_data = data_test.rename(columns={data_test.shape[1] - 1: 'label1'})
        train_data['label1'] = train_data['label1'].replace({'>50K': 1, '<=50K': 0})
        train_data = train_data.sample(frac=0.3)
        test_data['label1'] = test_data['label1'].replace({'>50K': 1, '<=50K': 0})
        target_dict = {'label1': 'label1'}
    elif 'breast' in task_name:
        file_name_tra = '../DataSet/UCI_breast_cancer/breast-cancer-wisconsin.csv'
        data = pd.read_csv(file_name_tra, header=None, index_col=[0]).dropna(how='all', axis=1)
        data.reset_index(drop=True, inplace=True)
        data_drop_dup = data.drop_duplicates()
        train_data = data_drop_dup.rename(columns={data_drop_dup.shape[1]: 'label1'})
        xx = train_data['label1'].value_counts()
        train_data['label1'][train_data['label1'] == 2] = 0
        train_data['label1'][(train_data['label1'] == 4)] = 1
        # train_data = train_data.sample(frac=0.1)
        target_dict = {'label1': 'label1'}
    elif 'wine' in task_name:
        file_name = '../DataSet/wine/winequality_white.csv'
        target_dict = {'label1': 'label1'}
        data = pd.read_csv(file_name)
        train_data = data.rename(columns={'quality': 'label1'})
        test_data = pd.DataFrame([])

    elif 'heart' in task_name:
        path_root = '../DataSet/UCI_heart_disease/'
        path_file = 'yanxishe.csv'
        data = pd.read_csv(path_root+path_file, index_col=['id'])
        train_data = data.rename(columns={'target': 'label1'})
        target_dict = {'label1': 'label1'}

    elif 'mimic' in task_name:
        if 'ppc' in task_name:
            file_name = '../DataSet/mimic/data_preprocessed_row.csv'
            data = pd.read_csv(file_name, index_col=['subject_id'])
            target_dict = {'label1': 'label1'}
            # col_drop = data.filter(regex=r'icd').columns
            # train_data = data.drop(columns=col_drop)
            train_data = data.rename(columns={'label_dead': 'label1'})

            data_1 = train_data.loc[train_data[target_dict['label1']] == 1]
            print(f'class 1 : {data_1.shape[0]}')
            data_0 = train_data.loc[train_data[target_dict['label1']] == 0]
            print(f'class 0 : {data_0.shape[0]}')
            if data_1.shape[0] < data_0.shape[0]:
                data_less = data_1
                data_more = data_0
            else:
                data_less = data_0
                data_more = data_1
            data_more = data_more.sample(n=data_less.shape[0] * 1, random_state=seed)
            # train_data = pd.concat([data_more, data_less]).sample(frac=1, random_state=seed)
        elif 'preprocessed' in task_name:
            file_name = '../DataSet/mimic/data_s_v3.csv'
            data = pd.read_csv(file_name)
            col_time_dead_pred = data.filter(regex=r'location|anchor|_id|careunit|services|admission_type|insurance|seq_num').columns
            train_data = data.drop(columns=col_time_dead_pred)
            train_data['label_dead'] = 0
            train_data['label_dead'][train_data['deathtime'].notna()] = 1
            train_data['label_dead'][train_data['dod'].notna()] = 1
            train_data['label_dead'][train_data['hospital_expire_flag'] == 1] = 1
            col_sp = train_data.filter(regex=r'drg_code').columns
            train_data[col_sp] = train_data[col_sp].astype('object')
            col_time = train_data.filter(regex=r'microbio|time|location|anchor|dod|hospital_expire_flag|drg_code|_version|_id|careunit|services|admission_type|insurance|seq_num').columns
            train_data = train_data.drop(columns=col_time)
            train_data['icustays_los_sum'].replace({0: np.nan}, inplace=True)
            target_dict = {'label1': 'label_dead'}
            train_data = train_data[(train_data['icd_code'] < 'I44') & (train_data['icd_code'] > 'I41')]
        else:
            file_name = '../DataSet/mimic/data_s.csv'
            columns_list = []
            chunker = pd.read_csv(file_name, chunksize=1000)
            for i, piece in enumerate(chunker):
                chunker_columns_dropna = piece.dropna(axis=1, thresh=5)
                columns_list.extend(chunker_columns_dropna.columns)
                print(chunker_columns_dropna.shape[1])
                print(piece.shape[1])
            columns_set = set(columns_list)
            data_all = pd.DataFrame([])
            chunker = pd.read_csv(file_name, chunksize=1000)
            for i, piece in enumerate(chunker):
                piece_ = piece.loc[:, columns_set]
                data_all = pd.concat([data_all, piece_], axis=0)
            data_all_sample = data_all.sample(frac=0.1, random_state=seed)
            data_all.to_csv('./data_all.csv', index=False)
            train_data = data_all_sample

    return train_data, test_data, target_dict
