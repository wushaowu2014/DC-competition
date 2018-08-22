import datetime
import lightgbm as lgb
from utils import build_useful_data, build_stacking_features
import pandas as pd
import numpy as np
import os

files = os.listdir()
#确保datas文件夹存在, 默认数据存在datas里
if "datas" not in files:
    os.mkdir("datas")

#确保backups文件夹存在
if "backups" not in files:
    os.mkdir("backups")

#指定需要进行stacking融合的特征文件
path_to_feature_files = ["datas/1and2_1_421_protein_std.csv", "datas/3_1_661_protein_std0.42.csv",
                         "datas/4_1_679_protein_std0.16.csv"]
path_to_feature_files += ["datas/3_1_500_protein_"+str(i)+".csv" for i in range(16)]

path_to_feature_files += ["datas/4_1_500_protein_"+str(k)+".csv" for k in range(5)]

#确保每一个特征文件都存在
all_files = os.listdir("datas/")
for dpath in path_to_feature_files:
    assert dpath.split('/')[-1] in all_files

# 为了方便验证，以及不必要的计算，我们没有去掉datas里的特征文件
# 如果需要重头到尾验证模型，可以手动清空datas文件里的特征文件
# 并先运行build_features.py,构建特征文件，再运行本程序
build_stacking_features(path_to_feature_files=path_to_feature_files)


data = pd.read_csv("datas/original_data.csv")

backups = os.listdir("backups/")
num_files = len(backups)
if num_files > 0:
    backup_versions = list(map(lambda x: int(x.split("_")[-1].split(".")[0]), backups))
    newest_version = np.sort(backup_versions)[-1]
else:
    newest_version = None

assert newest_version != None

print("newest version is %d"%(newest_version))

stacking_features = [str(i)+"_preds" for i in range(3,newest_version+1)] + ["0_preds", "1_preds", "2_preds"]

print("feature length", len(stacking_features))

print("features", stacking_features)

newest_tempdata = pd.read_csv("backups/tempdata_"+str(newest_version)+".csv")

data = data.merge(newest_tempdata, on=["Protein_ID", "Molecule_ID"], how="left")

print("data.columns", list(data.columns))
del newest_tempdata

#train, test 划分
test = data[data['Ki'] == -11]
train = data[data["Ki"]!=-11]

# Improve2
train=train.ix[train.Ki>=0]

test.reset_index(drop=True, inplace = True)

dtrain = lgb.Dataset(train[stacking_features].values, train.Ki.values)

dtest = lgb.Dataset(test[stacking_features].values, test.Ki.values, reference=dtrain)

#TODO grid search 以确定参数
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression_l2',

    'metric': 'l2',

    'min_child_weight': 3,

    'num_leaves': 2 ** 9,

    'lambda_l2': 10,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'colsample_bylevel': 0.7,

    'learning_rate': 0.05,

    'tree_method': 'exact',

    'seed': 2018,

    'nthread': 12,

    'silent': True

}

num_round = 50

gbm = lgb.train(params,

                dtrain,

                num_round,

                verbose_eval=50,

                valid_sets=[dtrain, dtest]

                )

# 结果保存

print("training finished!")

preds_sub = gbm.predict(test[stacking_features].values)

nowTime = datetime.datetime.now().strftime('%m%d%H%M')  # 现在

submission = test[['Protein_ID', 'Molecule_ID']]

name = 'lgb_stacked_' + nowTime + '.csv'

submission['Ki'] = preds_sub

submission.to_csv(name, index=False)