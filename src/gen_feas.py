#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description:
"""
import pandas as pd
from config import *
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from functools import partial
# 加载数据
train_user=pd.read_csv(train_user_path,sep='\t')
train_jd=pd.read_csv(train_jd_path,sep='\t',error_bad_lines=False)
train_action=pd.read_csv(train_action_path,sep='\t')

test_user=pd.read_csv(test_user_path,sep='\t')
test_action=pd.read_csv(test_action_path,sep=' ')


test_big_table = pd.merge(test_action,test_user,how="inner",on="user_id")
test_big_table = pd.merge(test_big_table,train_jd,how="inner",on="jd_no")
big_table = pd.merge(train_action,train_user,how="inner",on="user_id")
big_table = pd.merge(big_table,train_jd,how="inner",on="jd_no")

test_big_table.drop(columns=["company_name","max_edu_level","is_mangerial","resume_language_required"],inplace=True)

def extract_city(citys,index):
    city_list = citys.split(",")
    if index < len(city_list):
        city = city_list[index]
        if city != "-":
            city = int(city)
        else:
            city = -1
    else:
        city = -1
    return city

def exp_in_desc(exp,desc):
    if str(exp) == "nan":
        exp = ""
    exps = exp.split("|")
    num = 0
    for item in exps:
        if item in desc:
            num+=1
    return num


min_year_dict = {
    305: 4,
    -1:-1,
    1:1,
    103:2,
    0:0,
    510:7,
    1099:10,
    399:4,
    599:7,
    199:1,
    299:2,
    110:1
}
degree_dict = {
    "初中":1,
    "中技":2,
    "高中":3,
    "中专":3,
    "大专":4,
    "本科":5,
    "硕士":6,
    "博士":7,
    "EMBA":7,
    "MBA":6,
    "其他":0,
    "请选择":0,
    "\\N":0,
    "na":0
}
min_salary_dict = {
    100002000:1000,
    400106000:4001,
    0:0,
    200104000:2001,
    600108000:6001,
    800110000:8001,
    1000115000:10001,
    2500199999:25001,
    1500125000:15001,
    3500150000:35001,
    70001100000:70001,
    1000:0,
    100001150000:100001,
    2500135000:25001,
    5000170000:50001
}
max_salary_dict = {
    100002000:2000,
    400106000:6000,
    0:0,
    200104000:4000,
    600108000:8000,
    800110000:10000,
    1000115000:15000,
    2500199999:99999,
    1500125000:25000,
    3500150000:50000,
    70001100000:100000,
    1000:1000,
    100001150000:150000,
    2500135000:35000,
    5000170000:70000
}


def fe(df):
    df["desire_jd_city_1"] = df["desire_jd_city_id"].apply(partial(extract_city, index=0))
    df["desire_jd_city_2"] = df["desire_jd_city_id"].apply(partial(extract_city, index=1))
    df["desire_jd_city_3"] = df["desire_jd_city_id"].apply(partial(extract_city, index=2))
    df["desire_jd_city_num"] = df[["desire_jd_city_1", "desire_jd_city_2", "desire_jd_city_3"]].sum(axis=1)

    df["city_equal_desired_city_1"] = df["desire_jd_city_1"] == df["city"]
    df["city_equal_desired_city_2"] = df["desire_jd_city_2"] == df["city"]
    df["city_equal_desired_city_3"] = df["desire_jd_city_3"] == df["city"]

    df["work_years"] = 2019 - df["start_work_date"].apply(lambda x: 2018 if x == "-" else int(x))

    df["desire_min_salary"] = df["desire_jd_salary_id"].apply(lambda x: min_salary_dict[x])
    df["desire_max_salary"] = df["desire_jd_salary_id"].apply(lambda x: max_salary_dict[x])
    df["desire_salary_diff"] = df["desire_max_salary"] - df["desire_min_salary"]

    df["min_years"] = df["min_years"].apply(lambda x: min_year_dict[x])

    df["work_years_statisfied"] = df["work_years"].astype(int) > df["min_years"]

    df["salary_large_than_desire"] = df["desire_min_salary"] > df["min_salary"]

    df["cur_salary_min"] = df["cur_salary_id"].apply(lambda x: min_salary_dict[int(x if str.isnumeric(x) else "0")])
    df["cur_salary_max"] = df["cur_salary_id"].apply(lambda x: max_salary_dict[int(x if str.isnumeric(x) else "0")])

    df["salary_large_than_cur"] = df["cur_salary_min"] > df["min_salary"]

    df["cur_degree_id"] = df["cur_degree_id"].fillna("na").apply(lambda x: degree_dict[x.strip()])

    df["job_description_len"] = df["job_description"].apply(len)

    df["experience_num"] = df["experience"].apply(lambda x: len(str(x).split("|")) if str(x) != "nan" else 0)

    df["min_edu_level"] = df["min_edu_level"].fillna("na").apply(lambda x: degree_dict[x.strip()])
    exp_in_desc_num = []
    for idx, data in df.iterrows():
        exp_in_desc_num.append(exp_in_desc(data["experience"], data["job_description"]))
    df["exp_in_desc_num"] = exp_in_desc_num

    #     "live_city_id","desire_jd_salary_id","cur_industry_id","cur_jd_type","cur_salary_id",
    #          "cur_degree_id","city","jd_sub_type",
    #          "max_salary","min_salary","is_travel","min_years","min_edu_level",
    #          "desire_jd_city_1","desire_jd_city_2","desire_jd_city_3","work_years_statisfied"
    cross_feature_tuple = [("live_city_id", "city"), ("live_city_id", "desire_jd_city_1"),
                           ("cur_industry_id", "jd_sub_type"),
                           ("cur_jd_type", "jd_sub_type"), ("cur_salary_id", "cur_degree_id"), ("city", "jd_sub_type"),
                           ("jd_sub_type", "min_salary"), ("jd_sub_type", "max_salary"), ("jd_sub_type", "is_travel"),
                           ("min_years", "jd_sub_type"), ("jd_sub_type", "require_nums")]
    cross_feature_names = list(feature[0] + "&" + feature[1] for feature in cross_feature_tuple)
    print("create cross features", cross_feature_names)
    for idx, (fa, fb) in enumerate(cross_feature_tuple):
        df[cross_feature_names[idx]] = df[fa].astype(str) + df[fb].astype(str)
    return cross_feature_names


cross_feature_names = fe(big_table)
fe(test_big_table)

all_big_table = pd.concat([big_table,test_big_table])


from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm as tqdm
import lightgbm as lgb

def feature_select(target,*df_list):
    result = []
    cat_features = ["live_city_id","desire_jd_salary_id","cur_industry_id","cur_jd_type","cur_salary_id",
         "cur_degree_id","city","jd_sub_type",
         "max_salary","min_salary","is_travel","min_years","min_edu_level",
         "desire_jd_city_1","desire_jd_city_2","desire_jd_city_3","work_years_statisfied"]+cross_feature_names
    lbl_dict = {}
    for f in cat_features:
        lbl = LabelEncoder()
        lbl.fit(all_big_table[f].astype(str))
        lbl_dict[f] = lbl
    for df in df_list:
        features = ["live_city_id","desire_jd_salary_id","cur_industry_id","cur_jd_type","cur_salary_id",
             "cur_degree_id","birthday","city","jd_sub_type","require_nums",
             "max_salary","min_salary","is_travel","min_years","min_edu_level",
             "desire_jd_city_1","desire_jd_city_2","desire_jd_city_3","exp_in_desc_num",
                   "city_equal_desired_city_1","city_equal_desired_city_2","city_equal_desired_city_3",
                   "desire_min_salary","desire_max_salary","salary_large_than_desire","cur_salary_min",
                   "cur_salary_max","salary_large_than_cur","job_description_len","experience_num","work_years_statisfied","work_years","desire_jd_city_num","desire_salary_diff"]+cross_feature_names

        x = df[features]
        if target in df.columns:
            y = df[target]
        else:
            y = None
        for f in cat_features:
            lbl = lbl_dict[f]
            x[f] = lbl.transform(x[f].astype(str))
        result.append((x,y))
    return result


from sklearn.model_selection import KFold


def cross_validate(
        param=dict(n_estimators=1000, metric="map", colsample_bytree=0.2, max_depth=7, importance_type="gain")
        , n_folds=5, target="satisfied"):
    train_users = big_table["user_id"].unique()
    folds = KFold(n_folds, shuffle=True, random_state=42)
    models = []
    test_pred = np.zeros(test_big_table.shape[0])
    scores = []
    for idx, (train_idx, valid_idx) in enumerate(folds.split(train_users)):
        t_user = train_users[train_idx]
        v_user = train_users[valid_idx]
        train_data = big_table[big_table["user_id"].isin(t_user)]
        valid_data = big_table[big_table["user_id"].isin(v_user)]
        train_group = train_data.groupby("user_id", as_index=False).count()["satisfied"].values
        valid_group = valid_data.groupby("user_id", as_index=False).count()["satisfied"].values
        test_group = test_big_table.groupby("user_id", as_index=False).count()["jd_no"].values

        result = feature_select(target, train_data, valid_data, test_big_table)
        t_x, t_y = result[0]
        v_x, v_y = result[1]
        test_x, _ = result[2]
        model = lgb.LGBMRanker(**param)
        print("Fold", idx, "-" * 30)
        model.fit(t_x, t_y, group=train_group, eval_set=[(t_x, t_y), (v_x, v_y)], eval_group=[train_group, valid_group],
                  early_stopping_rounds=100, verbose=10,
                  callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.01)]
                  )
        models.append(model)
        test_pred += model.predict(test_x) / n_folds
        scores.append(model.best_score_["valid_1"]["ndcg@1"])
    print("mean score", np.mean(scores))
    return models, test_pred


models,pred = cross_validate(target="satisfied",param=dict(n_estimators=1000,metric="ndcg",subsample=0.8,min_split_gain=10,colsample_bytree=0.8,max_depth=2,importance_type="gain"),n_folds=5)

import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.DataFrame()
for idx,model in enumerate(models):
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = ["live_city_id","desire_jd_salary_id","cur_industry_id","cur_jd_type","cur_salary_id",
             "cur_degree_id","birthday","city","jd_sub_type","require_nums",
             "max_salary","min_salary","is_travel","min_years","min_edu_level",
             "desire_jd_city_1","desire_jd_city_2","desire_jd_city_3","exp_in_desc_num",
                   "city_equal_desired_city_1","city_equal_desired_city_2","city_equal_desired_city_3",
                   "desire_min_salary","desire_max_salary","salary_large_than_desire","cur_salary_min",
                   "cur_salary_max","salary_large_than_cur","job_description_len","experience_num","work_years_statisfied","work_years","desire_jd_city_num","desire_salary_diff"]+cross_feature_names
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = idx
    feature_importance = pd.concat([feature_importance,fold_importance])
plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=feature_importance.sort_values(by="importance", ascending=False));
plt.title('satisfied LGB Features (avg over folds)');


submit = test_big_table[["user_id","jd_no"]]
submit["score"] = pred
submit = submit.reset_index(drop=True)
result = pd.merge(test_action,submit,how="left",on=["user_id","jd_no"])
result.fillna(-100,inplace=True)
result = result.groupby("user_id",as_index=False).apply(lambda x:x.sort_values("score",ascending=False))
result[["user_id","jd_no"]].to_csv("../result/submission.csv",index=False)