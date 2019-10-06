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
# test = pd.read_csv('../data/zhaopin_round1_user_exposure_A_20190723', sep=' ')
test = pd.read_csv('../data/zhaopin_round1_user_exposure_B_20190819', sep=' ')
test = pd.merge(test, test_user, on='user_id', how='left')
test = pd.merge(test, train_jd, on='jd_no', how='left')

df = pd.concat((train, test))
df['user_nun_jd'] = df.groupby('user_id').jd_no.transform('nunique')
df['jd_nun_user'] = df.groupby('jd_no').user_id.transform('nunique')
df['user_cou_jd'] = df.groupby('user_id').jd_no.transform('count')
df['jd_cou_user'] = df.groupby('jd_no').user_id.transform('count')
df['use_jd_count'] = df.groupby(['user_id', 'jd_no']).jd_no.transform('count')
# for i in ['jd_title']:
#     df['count_%s' %i] = df.groupby(i).user_id.transform('count')
#     df['nun_%s' %i] = df.groupby(i).user_id.transform('nunique')

train, test = df[:len(train)], df[len(train):]


jd_nos = list(train_jd['jd_no'])
train = train[train['jd_no'].isin(jd_nos)]
train = train.drop_duplicates(subset=['user_id', 'jd_no'])

test = test.drop_duplicates(subset=['user_id', 'jd_no'])

df = pd.concat((train, test))
for i in ['jd_title']:
    df['count_%s' %i] = df.groupby(i).user_id.transform('count')
    df['nun_%s' %i] = df.groupby(i).user_id.transform('nunique')
train, test = df[:len(train)], df[len(train):]
print(len(train[train['satisfied'] == 1])/len(train[train['satisfied'] == 0]))
# train = train[train['experience'].notnull()]
# print(train.shape, ' <------------train shape')
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

    train['len_job_description'] = train['job_description'].apply(lambda x: len(str(x)))
    train["experience_num"] = train["experience"].apply(lambda x: len(str(x).split("|")) if str(x) != "nan" else 0)



    # def title_exp(x, y):
    #     if not x == x or not y == y:
    #         return 0
    #     elif str(x) in str(y):
    #         return 1
    #     else:
    #         return 0
    #
    # train['jd_title_in_experience'] = train.apply(lambda x: title_exp(x['jd_title'], x['experience']), axis=1)
    #
    #
    #
    # print(train['jd_title_in_experience'].value_counts())

    # def get_eng(x):
    #     try:
    #         return re.findall('[a-zA-Z0-9]+', x)
    #     except:
    #         print(x)
    #         return []
    #
    # train['eng_len_experience'] = train['experience'].apply(lambda x: len(get_eng(x)))
    # train['eng_len_job_description'] = train['job_description'].apply(lambda x: len(get_eng(x)))

    # import jieba
    #
    # def get_distance_seg(x, y, distance = Levenshtein.distance):
    #     if pd.isnull(x) or pd.isnull(y):
    #         return np.nan
    #     else:
    #         x = " ".join(jieba.cut(x, cut_all=False))
    #         y = " ".join(jieba.cut(y, cut_all=False))
    #         return distance(x, y)
    #
    # for i in [['job_description', 'experience'], ['desire_jd_type_id', 'jd_sub_type']]:
    #
    #     train['seg_levenshtein_%s_%s' % (i[0], i[1])] = train[[i[0], i[1]]].apply(
    #                 lambda x: get_distance_seg(x[i[0]], x[i[1]]), axis=1)

    # def str2date(x, type):
    #     try:
    #         return datetime.strptime(x, "%Y%m%d")
    #     except:
    #         if type == 'start_date':
    #             return datetime.strptime('20190325', "%Y%m%d")
    #         else:
    #             return datetime.strptime('20190424', "%Y%m%d")
    # for fea in ['start_date', 'end_date']:
    #     train[fea] = train[fea].apply(lambda x: str2date(x, type=fea))
    #
    # train['start_date_m'] = train['start_date'].apply(lambda x: x.month)
    # train['start_date_d'] = train['start_date'].apply(lambda x: x.day)
    #
    # train['end_date_m'] = train['start_date'].apply(lambda x: x.month)
    # train['end_date_d'] = train['end_date'].apply(lambda x: x.day)
    # train['date_diff'] = train['end_date'] - train['start_date']
    # train['date_diff'] = train['date_diff'].apply(lambda x: x.days)

    def get_date_diff(x, y):
        try:
            diff = datetime.strptime(y, "%Y%m%d") - datetime.strptime(x, "%Y%m%d")
            diff = diff.days
        except:
            diff = np.nan

        return diff

    train['date_diff'] = train.apply(lambda x: get_date_diff(x['start_date'], x['end_date']), axis=1)
    # print(train['date_diff'])

    # train['jd_no_lab'] = train['jd_no']

    no_feas = ['user_id', 'jd_no', 'browsed', 'delivered', 'satisfied', 'company_name', 'key', 'job_description', 'label']

    # train['min_ye_jd_type'] = train['min_years'].astype(str) + train['jd_sub_type']
    # print(train['min_ye_jd_type'])

    encoder_feas = ['live_city_id', 'desire_jd_city_id', 'desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id',
                    'cur_jd_type', 'cur_degree_id', 'experience', 'jd_title', 'jd_sub_type', 'start_date', 'end_date',
                    'max_edu_level', 'is_mangerial', 'resume_language_required']

    # print(train[['city', 'desire_jd_city_id', 'desire_job_city_same']])

    # print(train.head())


    # if type == 'train':
    #     # train['label'] = train.apply(lambda x: x['satisfied']*10 + x['delivered']*3 + x['browsed']*1,axis=1)
    #     train['label'] = train['satisfied']*0.7 + train['delivered']*0.3

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

def lgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 36,
        'learning_rate': 0.02,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.7,
        'random_state': 1024,
    }
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    num_round = 100000
    model = lgb.train(params,
                      trn_data,
                      num_round,
                      valid_sets=[trn_data, val_data],
                      verbose_eval=200,
                      early_stopping_rounds=500)
    return model

from keras import backend as K
import tensorflow as tf
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

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
            # print(auc(y_test, vali_pre))
            score = roc_auc_score(y_test, vali_pre)
            print('score ----------> ', score)
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

# 'delivered', 'satisfied'

label = train['satisfied']
print(label.value_counts())
print('*******************************************************************************')
sub = test[['user_id', 'jd_no']]

train_df, test_df = get_label(train_df, test_df, encoder_feas)

# train_df.to_csv('../nn/feas/train_feas.csv', index=None)
# test_df.to_csv('../nn/feas/test_feas.csv', index=None)

print(train_df.shape)
print(train_df.head())
re_sub, oof = get_result(train_df, test_df, label, lgb_para_binary_model, need_sca=True, splits_nums=5)

# sub_statis = copy.deepcopy(re_sub)
# sub_statis['score'] = sub_statis.apply(lambda x: -1 if x['jd_no'] not in jd_nos else x['score'], axis=1)
# sub_statis = sub_statis.sort_values(['user_id', 'score'], ascending=False)
# sub_statis[['user_id', 'jd_no']].drop_duplicates().to_csv('../result/sub_split_statis.csv', index=None)

label = train['delivered']
print(label.value_counts())
print('*******************************************************************************')
re_sub2, oof2 = get_result(train_df, test_df, label, lgb_para_binary_model, need_sca=True, splits_nums=5)

# sub_statis = copy.deepcopy(re_sub2)
# sub_statis['score'] = sub_statis.apply(lambda x: -1 if x['jd_no'] not in jd_nos else x['score'], axis=1)
# sub_statis = sub_statis.sort_values(['user_id', 'score'], ascending=False)
# sub_statis[['user_id', 'jd_no']].drop_duplicates().to_csv('../result/sub_split_delive.csv', index=None)

# label = train['browsed']
# print(label.value_counts())
# print('*******************************************************************************')
# re_sub3, oof3 = get_result(train_df, test_df, label, lgb_para_binary_model, need_sca=True, splits_nums=5)
#
# re_sub2['score'] = (re_sub2['score'] - re_sub2['score'].min()) / (re_sub2['score'].max() - re_sub2['score'].min())
# re_sub['score'] = (re_sub['score'] - re_sub['score'].min()) / (re_sub['score'].max() - re_sub['score'].min())
# re_sub3['score'] = (re_sub3['score'] - re_sub3['score'].min()) / (re_sub3['score'].max() - re_sub3['score'].min())
# re_sub['score'] = re_sub2['score']*0.3 + re_sub['score']*0.7 + re_sub3['score']*0.1

re_sub2['score'] = (re_sub2['score'] - re_sub2['score'].min()) / (re_sub2['score'].max() - re_sub2['score'].min())
re_sub['score'] = (re_sub['score'] - re_sub['score'].min()) / (re_sub['score'].max() - re_sub['score'].min())
re_sub['score'] = re_sub2['score']*0.2 + re_sub['score']*0.8

print(re_sub)
re_sub['score'] = re_sub.apply(lambda x: -1 if x['jd_no'] not in jd_nos else x['score'], axis=1)
re_sub = re_sub.sort_values(['user_id', 'score'], ascending=False)
re_sub[['user_id', 'jd_no']].drop_duplicates().to_csv('../result/sub_split_b.csv', index=None)
print(re_sub)

