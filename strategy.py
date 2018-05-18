import pandas as pd
import numpy as np

# Please add models here!!!
def model(model_name, train_x, train_y, test_x):
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    import sklearn
    if model_name == 'None':
        test_y = pd.Series(np.random.random_sample((len(test_x),)), index=test_x.index)
    if model_name == 'MLPRegressor':
        mlp = MLPRegressor(hidden_layer_sizes=(20, 20))
        mlp.fit(train_x, train_y)
        y_pred = mlp.predict(test_x)
        test_y = pd.Series(y_pred, index=test_x.index)
    if model_name == 'Lasso':
        model = sklearn.linear_model.Lasso(0.001)
        lasso = model.fit(train_x, train_y)
        test_y = pd.Series(lasso.predict(test_x), index=test_x.index)

    if model_name == 'Ridge':
        model = sklearn.linear_model.Ridge(50)
        ridge = model.fit(train_x, train_y)
        test_y = pd.Series(ridge.predict(test_x), index=test_x.index)
    if model_name == 'SVR':
        svr_rbf = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=0.1)
        svr_rbf.fit(train_x, train_y)
        y_pred_rbf = svr_rbf.predict(test_x)
        test_y = pd.Series(y_pred_rbf, index=test_x.index)
    return test_y

def fix_stock_order(test_y, threshold=0.05):
    '''
    test_y: series with tic index
    threshold: top 5% percent
    return: ratio in risky asset
    '''
    # case 1
    weight_new = pd.Series(np.zeros(len(test_y)), index=test_y.index)
    test_y.sort_values(ascending=False, inplace=True)
    pool = test_y[test_y >= test_y.quantile(q=1 - threshold)]
    weight_new.loc[pool.index.values] = 1.0 / len(pool)
    return weight_new

# stoploss function
def stoploss(df, re, i, max_new, weight_new):
        flag = 0
        stock = weight_new[weight_new != 0].index.intersection(df.columns)
        # creat indicator for position
        unit = pd.Series(np.full([len(weight_new.index), 1, ], 1)[:, 0], index=weight_new.index)
        unit[weight_new[weight_new < 0].index] *= -1.0
        stop_info = (df[stock].iloc[i - 1] - max_new[stock]) / max_new[stock] * unit[stock]
        if len(stop_info[stop_info < -re]) != 0:
            weight_new[stop_info[stop_info < -re].index] = 0
            weight_new = weight_new / weight_new.sum()
            flag = 1
        return weight_new, flag


