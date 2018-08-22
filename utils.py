import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
import os


def gbm_model(train, test, features):
    """
    #TODO grid search　以确定参数
    简单的lgb模型
    :param train: 训练数据，　pd.DataFrame
    :param test: 测试数据，　pd.DataFrame
    :param features: 特征，　list or np.array
    :return:
    """
    dtrain = lgb.Dataset(train[features].values, train.Ki.values)

    dtest = lgb.Dataset(test[features].values, test.Ki.values, reference=dtrain)

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
    #TODO(Whitemoon) 降低 num_round 减少过拟合
    num_round = 2000

    gbm = lgb.train(params,

                    dtrain,

                    num_round,

                    verbose_eval=50,

                    valid_sets=[dtest]

                    )

    # 结果保存

    del dtrain, dtest

    print("training finished!")

    preds = gbm.predict(test[features].values)
    return preds

def findall(sequence, word_length = 3, stride = 1):
    """
    :param sequence: 蛋白质分子序列, string
    :param word_length: 词长, int
    :param stride: 步长, int
    :return: a list, 以stride为步长，包含蛋白质分子序列的所有长度为word_length的序列
    """
    total_length = len(sequence)
    flag = (total_length - word_length)%stride == 0
    final_length = int(float(total_length - word_length) / stride + 1)
    indexs = np.array(range(final_length))*stride
    result = [sequence[i:i+word_length] for i in indexs]
    if flag:
        return result
    else:
        quekou = 2*word_length - (total_length - indexs[-1])
        return result + [sequence[indexs[-1]+word_length - quekou:]]


def protein_embedding(protein_all, word_length = 3, stride = 1):
    """
    构建蛋白质词向量特征
    :param protein_all: 所有蛋白质词向量的序列, string
    :param word_length: 词长, int
    :param stride: 步长, int
    :return: 蛋白质词向量特征（pd.DataFrame实例）
    """

    texts_protein = list(protein_all["Sequence"].apply(lambda x: findall(x.upper(), word_length, stride)))

    n = 128

    model_protein = Word2Vec(texts_protein, size=n, window=4, min_count=1, negative=3,

                             sg=1, sample=0.001, hs=1, workers=4)

    vectors = pd.DataFrame([model_protein[word] for word in (model_protein.wv.vocab)])

    vectors['Word'] = list(model_protein.wv.vocab)

    vectors.columns = ["vec_{0}".format(i) for i in range(0, n)] + ["Word"]

    wide_vec = pd.DataFrame()

    result1 = []

    aa = list(protein_all['Protein_ID'])

    for i in range(len(texts_protein)):

        result2 = []

        for w in range(len(texts_protein[i])):
            result2.append(aa[i])

        result1.extend(result2)

    wide_vec['Id'] = result1

    result1 = []

    for i in range(len(texts_protein)):

        result2 = []

        for w in range(len(texts_protein[i])):
            result2.append(texts_protein[i][w])

        result1.extend(result2)

    wide_vec['Word'] = result1

    del result1

    wide_vec = wide_vec.merge(vectors, on='Word', how='left')

    wide_vec = wide_vec.drop('Word', axis=1)

    wide_vec.columns = ['Protein_ID'] + ["vec_{0}".format(i) for i in range(0, n)]

    del vectors

    name = ["vec_{0}".format(i) for i in range(0, n)]

    feat = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()

    del wide_vec

    feat.columns = ["Protein_ID"] + [str(word_length)+"_mean_ci_{0}".format(i) for i in range(0, n)]

    return feat



def tfidf_and_wordcounts(protein_all, PID, word_length = 2, stride = 1):
    """
    构建蛋白质序列的tfidf和wordcount特征
    :param protein_all: 所有蛋白质词向量的序列, pd.DataFrame
    :param PID: 所有蛋白质ID
    :param word_length:　词长
    :param stride: 步长
    :return: tfidf特征和wordcount特征（pd.DataFrame实例）
    """
    #用词长为word_length, 步长为stride来选择蛋白质文本信息
    texts_protein = list(protein_all["Sequence"].apply(lambda x: findall(x.upper(), word_length, stride)))

    #合并＂蛋白质文本＂，并用空格隔开每个蛋白质序列的＂单词＂，构建＂文本＂
    corpus = list(map(lambda x: " ".join(i for i in x), texts_protein))
    #计算每个＂单词＂的在蛋白质序列中的　term-frequence and inverse-document-frequence
    tfidf = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    tfidf_vals = tfidf.fit_transform(corpus)
    tfidf_vals = tfidf_vals.toarray()

    #计算每个＂单词＂在每个蛋白质序列出现的次数
    counts = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    word_counts = counts.fit_transform(corpus)
    word_counts = word_counts.toarray()

    del corpus

    tfidf_vals = pd.DataFrame(tfidf_vals, columns=[str(word_length)+"_ags_tfidfs_" + str(i) for i in range(tfidf_vals.shape[1])])
    word_counts = pd.DataFrame(word_counts, columns=[str(word_length)+"_ags_wordcounts_" + str(i) for i in range(word_counts.shape[1])])

    tfidf_vals["Protein_ID"] = PID
    word_counts["Protein_ID"] = PID

    return tfidf_vals, word_counts



def build_useful_data():
    """
    #TODO 利用pca降维，或者LDA降维......方式构建特征文件
    构建可用的初始特征数据, 默认原始竞赛数据储存在当前文件夹中的datas文件夹中.
    :return: 可用数据（pd.DataFrame实例）
    """

    # 读取蛋白质数据
    print("Loading and merging data")
    protein_train = pd.read_csv('datas/df_protein_train.csv')

    protein_test = pd.read_csv('datas/df_protein_test.csv')

    protein_all = pd.concat([protein_train, protein_test])

    #添加蛋白质序列长度作为特征
    protein_all['seq_len'] = protein_all['Sequence'].apply(len)

    #读取分子数据
    mol_train = pd.read_csv('datas/df_molecule.csv')

    aff_train = pd.read_csv('datas/df_affinity_train.csv')

    aff_test = pd.read_csv('datas/df_affinity_test_toBePredicted.csv')

    #初始化待预测的Ki值为-11
    aff_test['Ki'] = -11

    aff_all = pd.concat([aff_train, aff_test])

    data = aff_all.merge(mol_train, on="Molecule_ID", how='left')
    data = data.merge(protein_all, on='Protein_ID', how='left')

    #获取蛋白质ID
    PID = list(protein_all["Protein_ID"])

    #word_length = 1时的wordcount特征
    print("Processing wordcount1")
    _, word_counts1 = tfidf_and_wordcounts(protein_all, PID, word_length=1, stride=1)

    #word_length = 2时的wordcount特征
    print("Processing wordcount2")
    _, word_counts2 = tfidf_and_wordcounts(protein_all, PID, word_length=2, stride=1)

    word_counts1_2 = word_counts1.merge(word_counts2, on="Protein_ID", how="left")
    # 保存特征文件，以供后期训练
    word_counts1_2.to_csv("datas/1and2_1_421_protein_std.csv", index=False)

    del word_counts1_2, word_counts1, word_counts2

    print("Processing wordcount3")
    _, word_count3 = tfidf_and_wordcounts(protein_all, PID, word_length=3, stride=1)

    word_count3_features = list(word_count3.columns) #8000维的数据，需要降维
    word_count3_features.remove("Protein_ID")

    #利用标准差进行降维，设置标准差阈值为0.42，去掉标准差小于0.42的特征
    new_word_count3 = reduce_dims_with_std(word_count3, word_count3_features, std_threshold=0.42)
    #保存特征文件，以供后期训练
    new_word_count3.to_csv("datas/3_1_661_protein_std0.42.csv", index=False)
    del new_word_count3

    for i in range(len(word_count3_features) // 500):
        #每次划分500个特征，并保存在特征文件里，以供后期训练
        file = word_count3[["Protein_ID"] + word_count3_features[i * 500:(i + 1) * 500]]
        file_name = "3_1_500_protein_" + str(i)
        file.to_csv("datas/" + file_name + ".csv", index=False)
    del word_count3, word_count3_features

    print("Processing wordcount4")
    gc.collect()
    _, word_count4 = tfidf_and_wordcounts(protein_all, PID, word_length=4, stride=1)

    word_count4_features = list(word_count4.columns)#140000+　维的数据，需要降维
    word_count4_features.remove("Protein_ID")

    # 利用标准差进行降维，设置标准差阈值为0.16，去掉标准差小于0.16的特征
    new_word_count4 = reduce_dims_with_std(word_count4, word_count4_features, std_threshold=0.16)
    new_word_count4.to_csv("datas/4_1_679_protein_std0.16.csv", index=False)

    # 利用标准差进行降维，设置标准差阈值为0.13，去掉标准差小于0.13的特征
    new_word_count4 = reduce_dims_with_std(word_count4, word_count4_features, std_threshold=0.13)

    word_count4_features = list(new_word_count4.columns)
    word_count4_features.remove("Protein_ID")

    for i in range(len(word_count4_features) // 500):
        #每次划分500个特征，并保存在特征文件里，以供日后训练
        file = new_word_count4[["Protein_ID"] + word_count4_features[i * 500:(i + 1) * 500]]
        file_name = "4_1_500_protein_" + str(i)
        file.to_csv("datas/" + file_name + ".csv", index=False)

    del new_word_count4, word_count4

    #以下特征是蛋白质的词向量特征, 来自技术圈, 谢谢＂小武哥＂同学.但我们的最终提交版本没用这些特征
    "=====================================词向量特征==========================================="
    #feat2 = protein_embedding(protein_all, word_length = 2)
    #data = data.merge(feat2, on="Protein_ID", how="left")
    #del feat2
    #feat3 = protein_embedding(protein_all, word_length = 3)
    #data = data.merge(feat3, on="Protein_ID", how="left")
    #del feat3
    #feat4 = protein_embedding(protein_all, word_length = 4)
    #data = data.merge(feat4, on="Protein_ID", how="left")
    #del feat4
    "================================================================================"

    #分子指纹展开
    mol_fingerprints = list(mol_train["Fingerprint"].apply(lambda x: list(np.array(x.split(',')).astype(int))))
    mol_fingerprints = pd.DataFrame(mol_fingerprints, columns=["Fingerprint_"+str(i) for i in range(167)])
    mol_fingerprints["Molecule_ID"] = mol_train["Molecule_ID"]

    del PID
    "=================================================================================================="
    data = data.merge(mol_fingerprints, on="Molecule_ID", how='left')
    del mol_fingerprints
    del data["Sequence"], protein_train, protein_test, mol_train

    data.reset_index(drop = True, inplace = True)

    data.to_csv("datas/original_data.csv", index=False)

    del data
    print("Useful data have builded")


def get_pre_stacking_data(newest_version):
    """
    准备供stacking模型融合的数据
    :param newest_version:　指定的数据版本, int
    :return: 供模型融合的数据以及其初始特征
    """
    print("Loading original data")
    data = pd.read_csv("datas/original_data.csv")

    if newest_version != None:

        use_cols = ["Protein_ID", "Molecule_ID"] + [str(k) + "_preds" for k in range(newest_version + 1)]

        data_backups = pd.read_csv("backups/tempdata_"+str(newest_version)+".csv", usecols=use_cols)

        data = data.merge(data_backups, on=["Protein_ID", "Molecule_ID"], how="left")

        del data_backups

        init_features = [i for i in list(data.columns) if i not in
            ['Ki', 'Fingerprint', 'Protein_ID']+[str(k)+"_preds" for k in range(newest_version+1)]]

    else:

        print("There is no backups")

        init_features = [i for i in list(data.columns) if i not in
                ['Ki', 'Fingerprint', 'Protein_ID']]

    return data, init_features


def reduce_dims_with_std(dataframe, features, std_threshold = 0.3):
    """
    用标准差作为阈值进行数据降维
    :param dataframe: pd.DataFrame实例
    :param features: 需要降维的特征, list or np.array
    :param std_threshold: 标准差的阈值（标准差小于这个值的特征将被抛弃）, float
    :return: 降维后的数据（pd.DataFrame实例）
    """
    features = np.array(features)
    stds = dataframe[features].std()
    masks = (stds>std_threshold).values
    reduced_featrues = features[masks]
    reduced_featrues = np.concatenate([["Protein_ID"], reduced_featrues])
    print("After reduce dims, the final dim is", len(reduced_featrues), "while the original dim is: ",len(features))
    return dataframe[reduced_featrues]


def reduce_dims_with_pca(dataframe, features, n_conponents = 200):
    """
    :param dataframe: pd.DataFrame实例
    :param features: 需要降维的特征, list or np.array
    :param n_conponents: 主成分个数, int
    :return: pca 降维后的数据（pd.DataFrame实例）
    """
    features = np.array(features)
    tag = int(dataframe.columns[0].split('_')[0])
    assert features.shape[0] >= n_conponents
    PID = dataframe.Protein_ID.values
    pca = PCA(n_components=n_conponents)
    final_feature_names = [str(tag)+'_pca_'+str(i) for i in range(n_conponents)]
    dataframe = pd.DataFrame(pca.fit_transform((dataframe[features].values)), columns=final_feature_names)
    dataframe["Protein_ID"] = PID
    print("After reduce dims, the final dim is", n_conponents, "while the original dim is: ", len(features))
    return dataframe


def stacking(data, features, new_feature_name):
    """
    stacking模型融合
    :param data:　pd.DataFrame实例
    :param new_feature_name: 预测出来的特征名字, string
    :return: 特征数据
    """
    print("features", features)
    print("Features length", len(features))
    #划分训练和测试数据
    test = data[data['Ki'] == -11]
    train = data[data["Ki"] != -11]
    train.reset_index(drop = True, inplace = True)
    #将原有数据划分成５份，做stacking特征融合
    kf = KFold(n_splits=5, shuffle=True)
    final_result = pd.DataFrame()
    for train_idxs, vali_idxs in kf.split(train):
        gc.collect()
        temp_train = train.ix[train_idxs]
        temp_vali = train.ix[vali_idxs]
        preds = gbm_model(temp_train, temp_vali, features)
        temp_vali[new_feature_name] = preds
        final_result = pd.concat([final_result, temp_vali], axis=0)
    ultimate_preds = gbm_model(train, test, features)
    test[new_feature_name] = ultimate_preds
    del ultimate_preds
    final_result = pd.concat([final_result, test])
    del train, test
    return final_result



def build_stacking_features(path_to_feature_files):
    """
    构建stacking融合特征
    :param path_to_feature_files:　list, 特征文件的位置
    :return: None
    """
    all_files = os.listdir()
    if "backups" not in all_files:
        os.mkdir("backups")
    for i, feature_file in enumerate(path_to_feature_files):
        backups = os.listdir("backups/")
        num_files = len(backups)
        if num_files > 0:
            backup_versions = list(map(lambda x: int(x.split("_")[-1].split(".")[0]), backups))
            newest_version = np.sort(backup_versions)[-1]
        else:
            newest_version = None

        if newest_version != None:
            if i <= newest_version:
                continue
            print("We found backups, the number of backups is %d, and the newest version is %d" % (
                num_files, newest_version))
        else:
            print("No backups, we starting processing the %d th feature file."%(i))
        print("The feature file we are about to processing is %s"%path_to_feature_files[i])
        data, init_features = get_pre_stacking_data(newest_version)
        print("Loading and checking data")
        print("length of data columns", len(list(data.columns)))
        print("Process the %d th feature file"%i)
        gc.collect()
        feature_data = pd.read_csv(feature_file)  # 255 features
        temp_features = list(feature_data.columns)
        temp_features.remove("Protein_ID")
        temp_features = temp_features + init_features
        assert (str(i - 1) + "_preds" not in temp_features)
        data = data.merge(feature_data, on="Protein_ID", how='left')
        data = stacking(data, temp_features, new_feature_name = str(i)+"_preds")
        temp_data_items = ["Protein_ID", "Molecule_ID"] +[str(pp)+"_preds" for pp in range(i+1)]
        temp_data = data[temp_data_items]
        temp_data.to_csv("backups/tempdata_"+str(i)+".csv", index=False)
        #移除旧的版本
        if i > 0:
            os.remove("backups/tempdata_" + str(i-1) + ".csv")
