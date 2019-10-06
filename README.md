# tianchi_job_match
第二届阿里巴巴大数据智能云上编程大赛-智联招聘人岗智能匹配

### 一回归预测：

1，先通过data_clear.py修正一下数据

2，通过main2.py抛跑出结果

注意：暂时只用了部分统计特征，label暂定为 train['satisfied']*0.7 + train['delivered']*0.3，跟实际评测指标不同

后续可以考虑分别对satisfied和delivered做二分类，然后对结果融合。


### 二分类预测：

直接运行split_2.py即可出结果。

最好成绩为分类预测的结果，天池第一赛季排名第六。

因为第二赛季为线上比赛，用不习惯线上环境，只写了baseline。





