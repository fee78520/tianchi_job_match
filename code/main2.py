# -*- coding:utf-8 _*-
import Levenshtein
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import copy
import lightgbm as lgb
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import TfidfModel
from gensim import corpora,similarities,models
import math
import networkx as nx
from nltk.metrics.distance import jaccard_distance, masi_distance
import difflib
from simhash import Simhash
from nltk.tokenize import word_tokenize
import re
import xgboost as xgb


from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 400)

scaler = StandardScaler()

train_user = pd.read_csv('../data/zhaopin_round1_train_20190716/table1_user', sep='\t')
train_jd = pd.read_csv('../data/zhaopin_round1_train_20190716/table2_jd.csv', sep='\t')
action = pd.read_csv('../data/zhaopin_round1_train_20190716/table3_action', sep='\t')

train = pd.merge(action, train_user, on='user_id', how='left')
train = pd.merge(train, train_jd, on='jd_no', how='left')

test_user = pd.read_csv('../data/zhaopin_round1_test_20190716/user_ToBePredicted', sep='\t')
test = pd.read_csv('../data/zhaopin_round1_user_exposure_A_20190723', sep=' ')
test = pd.merge(test, test_user, on='user_id', how='left')
test = pd.merge(test, train_jd, on='jd_no', how='left')

# print([i for i in train.columns if i not in list(test.columns)])  #['browsed', 'delivered', 'satisfied']

def creat_feas(train, type='train'):
    train['user_id_count_jd_no'] = train.groupby('user_id').jd_no.transform('count')
    train['jd_no_count_user_id'] = train.groupby('jd_no').user_id.transform('count')

    # train['desire_jd_city_id'] = train['desire_jd_city_id'].apply(lambda x: [i for i in x.split(',')])

    train['desire_city_nums'] = train['desire_jd_city_id'].apply(lambda x: len([i for i in x.split(',') if i != '-']))
    train['0_city'] = train['desire_jd_city_id'].apply(lambda x: x.split(',')[0] if x.split(',')[0] != '-' else np.nan)
    train['1_city'] = train['desire_jd_city_id'].apply(lambda x: x.split(',')[1] if x.split(',')[1] != '-' else np.nan)
    train['2_city'] = train['desire_jd_city_id'].apply(lambda x: x.split(',')[2] if x.split(',')[2] != '-' else np.nan)

    train['city'].fillna(-100, inplace=True)
    train['desire_job_city_same'] = train.apply(lambda x: 1 if str(int(x['city'])) in x['desire_jd_city_id'].split(',') else 0, axis=1)
    train['live_job_city_same'] = train.apply(lambda x: 1 if str(int(x['live_city_id'])) in x['desire_jd_city_id'].split(',') else 0, axis=1)

    train['jd_sub_type_desire_jd_type_id_same'] = train.apply(lambda x: 1 if x['jd_sub_type'] == x['desire_jd_type_id'] else 0, axis=1)

    # 切分工作，交集个数
    def get_job_same_nums(x, y):
        try:
            jd_sub_type = re.split(',|/', x)
            desire_jd_type_id = re.split(',|/', y)
            nums = len([i for i in jd_sub_type if i in desire_jd_type_id])
        except:
            return -1

        return nums



    train['desire_jd_type_nums'] = train.apply(lambda x: get_job_same_nums(x['jd_sub_type'], x['desire_jd_type_id']), axis=1)

    train['has_cur_jd_type'] = train['cur_jd_type'].apply(lambda x: 1 if x else 0)

    degree_dict = {'\\N': 1, '初中': 1, '高中': 2, '其他': 2, '请选择': 2, '中专': 3, '中技': 3, '大专': 4, '本科': 5, '硕士': 6, 'MBA': 6,
                   'EMBA': 6, '博士': 7}
    train['cur_degree_id'] = train['cur_degree_id'].fillna('高中')
    train['degree'] = train['cur_degree_id'].apply(lambda x: int(degree_dict[x.strip()]))

    train['min_edu_level'] = train['min_edu_level'].fillna('高中')
    train['min_edu_level'] = train['min_edu_level'].apply(lambda x: int(degree_dict[x.strip()]))

    train['degree_diff'] = train['degree'] - train['min_edu_level'] + 7

    train['start_work_date'] = train['start_work_date'].replace('-', 2019)

    train['work_year'] = train['start_work_date'].apply(lambda x: 2019 - int(x))
    def get_year(x):
        try:
            x = str(int(x))
            if len(x) == 1:
                return int(x)
            elif len(x) == 3:
                return int(x[0])
            else:
                return int(x[:2])
        except:
            return 0
    train['min_years'] = train['min_years'].apply(lambda x: get_year(x))
    train['work_year_diff'] = train['work_year'] - train['min_years']

    def get_salary(x, y):
        if x == 0:
            x = y

        if x == '-':
            x = y

        x = str(x)

        if len(x) == 10:
            return int(x[:5])
        elif len(x) == 9:
            return int(x[:4])
        elif len(x) == 11:
            return int(x[:5])
    train['desire_jd_salary_id'] = train.apply(lambda x: get_salary(x['desire_jd_salary_id'], x['cur_salary_id']), axis=1)
    train['cur_salary_id'] = train.apply(lambda x: get_salary(x['cur_salary_id'], x['desire_jd_salary_id']), axis=1)
    train['salary_diff'] = train['desire_jd_salary_id'] - train['cur_salary_id'] + 1000

    no_feas = ['user_id', 'jd_no', 'browsed', 'delivered', 'satisfied', 'company_name', 'key', 'job_description\r\n', 'label']

    encoder_feas = ['live_city_id', 'desire_jd_city_id', 'desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id',
                    'cur_jd_type', 'cur_degree_id', 'experience', 'jd_title', 'jd_sub_type', 'start_date', 'end_date',
                    'max_edu_level', 'is_mangerial', 'resume_language_required']
    # print(train[['city', 'desire_jd_city_id', 'desire_job_city_same']])

    # print(train.head())


    if type == 'train':
        # train['label'] = train.apply(lambda x: x['satisfied']*10 + x['delivered']*3 + x['browsed']*1,axis=1)
        train['label'] = train['satisfied']*0.7 + train['delivered']*0.3

    return train, no_feas, encoder_feas





def get_label(train, test, encoder_feas):

    df = pd.concat((train, test))

    for i in encoder_feas:
        le = LabelEncoder()
        df[i] = df[i].astype(str)
        le.fit(df[i].unique())
        df[i] = le.transform(df[i])

    train, test = df[:len(train)], df[len(train):]

    return train, test

def lgb_para_reg_model(X_train, y_train, X_test, y_test):
    param = {'num_leaves': 30,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': 4,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    num_round = 100000
    model = lgb.train(param,
                      trn_data,
                      num_round,
                      valid_sets=[trn_data, val_data],
                      verbose_eval=200,
                      early_stopping_rounds=500)
    return model


def xgb_model_re(X_train, y_train, X_test, y_test):

    model = xgb.XGBRegressor(colsample_bytree=0.3,
                             # objective= reg:gamma,
                             # eval_metric='mae',
                             gamma=0.0,
                             learning_rate=0.01,
                             max_depth=4,
                             min_child_weight=1.5,
                             n_estimators=1668,
                             reg_alpha=1,
                             reg_lambda=0.6,
                             subsample=0.2,
                             seed=42,
                             silent=1)

    model = model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=30, verbose=1)
    return model

def get_result(train, test, label, my_model, need_sca=True, splits_nums=5):
    if need_sca:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    elif not need_sca:
        train = np.array(train)
        test = np.array(test)

    oof = np.zeros(train.shape[0])

    score_list = []

    label = np.array(label)

    k_fold = KFold(n_splits=splits_nums, shuffle=True, random_state=1024)
    for index, (train_index, test_index) in enumerate(k_fold.split(train)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model = my_model(X_train, y_train, X_test, y_test)

        vali_pre = model.predict(X_test)

        oof[test_index] = vali_pre

        print(len(y_test), len(vali_pre))

        print(y_test[:10])
        print(vali_pre[:10])
        try:
            score = math.sqrt(mean_absolute_error(y_test, vali_pre))
            score_list.append(score)
        except:
            pass

        pred_result = model.predict(test)
        sub['score'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
        else:
            re_sub['score'] = re_sub['score'] + sub['score']

    re_sub['score'] = re_sub['score'] / splits_nums

    print('score list:', score_list)
    print(np.mean(score_list))

    return re_sub, oof


train, no_feas, encoder_feas = creat_feas(train)

test, no_feas, encoder_feas = creat_feas(test, type='test')

feas = [i for i in train.columns if i not in no_feas]
train_df = train[feas]
test_df = test[feas]
label = train['label']
sub = test[['user_id', 'jd_no']]


train_df, test_df = get_label(train_df, test_df, encoder_feas)

re_sub, oof = get_result(train_df, test_df, label, lgb_para_reg_model, need_sca=True, splits_nums=5)
print(re_sub)
re_sub = re_sub.sort_values(['user_id', 'score'], ascending=False)
re_sub[['user_id', 'jd_no']].to_csv('../result/sub.csv', index=None)
print(re_sub)
