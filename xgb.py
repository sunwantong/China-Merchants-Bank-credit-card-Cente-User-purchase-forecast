import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import operator
import matplotlib.pyplot as plt

def xgb_model(train,test):
    train_x = train.drop(['USRID','FLAG','day'], axis=1).values
    train_y = train['FLAG'].values
    test_x = test.drop(['USRID','day'], axis=1).values

    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_test = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',  # 二分类的问题
              # 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
              'max_depth': 5,  # 构建树的深度，越大越容易过拟合
              # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
              'subsample': 0.7,  # 随机采样训练样本
              'colsample_bytree': 0.7,  # 生成树时进行的列采样
              'min_child_weight': 3,
              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
              # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
              # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
              'eta': 0.03,  # 如同学习率
              'nthread': 7,  # cpu 线程数
              'eval_metric': 'auc'  # 评价方式
              }

    plst = list(params.items())
    num_rounds = 500  # 迭代次数
    watchlist = [(xgb_train, 'train')]
    # early_stopping_rounds    当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)
    pred_value = model.predict(xgb_test)

    return pred_value


def gene_result(pred_value,test_range):
    tess = test_range[["USRID"]]
    a = pd.DataFrame(pred_value, columns=["RST"])
    res = pd.concat([tess, a["RST"]], axis=1)
    res.to_csv("../submit/test_result.csv", index=None,sep='\t')

def load_csv():
    train = pd.read_csv('../fea/train.csv', sep='\t')
    test = pd.read_csv('../fea/test.csv', sep='\t')

    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)

    return train,test


def main():
    train,test = load_csv()
    pred_value = xgb_model(train,test)
    gene_result(pred_value, test)


if __name__ == '__main__':
    main()




