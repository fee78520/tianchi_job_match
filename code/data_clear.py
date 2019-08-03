import pandas as pd

def modeified_jd_df():
    tmp_list = []
    tmp_file = open(',,.data/zhaopin_round1_train_20190716/table2_jd', encoding='utf8')
    for i, j in enumerate(tmp_file.readlines()):
        if i == 175425:
            j = j.replace('销售\t|置业顾问\t|营销', '销售|置业顾问|营销')
        tmp = j.split('\t')
        tmp_list.append(tmp)
    tmp_file.close()
    new_jd = pd.DataFrame(tmp_list)
    new_jd.to_csv('../data/zhaopin_round1_train_20190716/table2_jd.csv', encoding='utf8', sep='\t', index=False, header=None)


modeified_jd_df()


