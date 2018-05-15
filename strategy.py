import os
import pandas as pd
import numpy as np
import utils as ut
from utils import context

path = os.getcwd()
# price_file = os.path.join(path + '/data/prices.csv')
# f_file = os.path.join(path + '/data/Fdata_new.csv')
# prices = pd.read_csv(price_file)
# fd = ut.input_data(f_file)
# prices = ut.input_data(price_file)
# print(fd.head())
# prices.to_csv('prices.csv')
# fd.to_csv('fd.csv')


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

# ==================read_data
prices = pd.read_csv('prices.csv',)
fd = pd.read_csv('fd.csv',)
fd = fd.set_index(['datadate'])
prices = prices.set_index(['datadate'])
variable_list = pd.read_csv('characteristics.csv',header=None).values.reshape((1,-1)).tolist()[0]
symbols = list(prices.columns[2:])
print(variable_list)

# ================selected variable
start_date = '2010-01-01'
roll = 12
horizon = 21*3
relative = True # whether subtract sp500 index return
select_context = context(prices,fd,horizon,start_date,variable_list,symbols)
select_context.generate_trains(relative)
train_x,train_y,test_x =select_context.extract_train(cur_date = '2018-05-04',roll=roll)
print(np.shape(train_x),np.shape(train_y))
# from scipy import stats
# import statsmodels.api as sm
# slope, intercept, r_value, p_value, std_err = stats.linregress(train_x,train_y)
# regressor = sm.OLS(np.array(train_y), np.array(train_x)).fit()
# print(regressor.summary())

# lm = linear_model.LinearRegression()
# model = lm.fit(np.array(train_x),np.array(train_y))
# lm.score(np.array(train_x),np.array(train_y))
# abs(lm.coef_)
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import GenericUnivariateSelect
x_new = GenericUnivariateSelect(mutual_info_regression,param= 20).fit(np.array(train_x),np.array(train_y))
variable_list=list(train_x.columns[x_new.get_support()])
print(variable_list)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# parameter initialization
threshold = 0.1
freq = 21*3
# how many quarters need to roll
roll = 12
start_date = '1998-01-01'
end_date = -1
trading_days = 252.0
# no transaction fee
leverage = 0.95
context = context(prices,fd,horizon,start_date,variable_list,symbols,end_date)
context.generate_trains(relative)
model_name = 'Lasso'

if __name__ == "__main__":
    # initial setting
    df = context.prices.copy()
    stock_num = len(symbols)
    back_test = context.price_date[pd.to_datetime(context.price_date) >= pd.to_datetime(start_date)]
    df = df.loc[back_test.values, :]
    unit = np.full((len(df.index), 1), 1)[:, 0]
    df['rebalancing'] = pd.Series()
    df['stoploss'] = pd.Series()
    df['nav'] = pd.Series(unit, index=df.index)
    weight_new = []
    max_new = []  # for computing max_drawdown
    unit = np.full((len(df.index), stock_num), 0)
    weights = pd.DataFrame(unit, index=df.index, columns=symbols)
    reb_index = 0
    s = 0  # counting date
    # ============================= Enter Back-testing ===================================
    for cur_date in back_test.values:
        # stoploss
        cur_date = pd.to_datetime(cur_date)
        if len(max_new) != 0:
            pass
            # weight_new, flag = stoploss(df, stop_loss, i, max_new, weight_new)
            # if flag == 1:
            #     print(df.index[i - 1])
            #     print('stoploss!: ')
            #     print(weight_new)
            #     weights.iloc[i - 1, :] = weight_new.values
            #     print('-' * 50)
            #     df['stoploss'].ix[i - 1] = 1
            #     reb_index = i - 1
        # rebalance in a fixed frequency based on freq
        if np.mod(s, freq) == 1:
            train_x, train_y, test_x = context.extract_train(cur_date, roll=roll)

            test_y = model(model_name, train_x, train_y, test_x)
            weight_new = fix_stock_order(test_y, threshold)

            print(df.index[s - 1])
            print(weight_new[weight_new != 0])
            weights.loc[df.index[s - 1], weight_new.index] = weight_new.values
            print('*' * 50)
            df['rebalancing'].iloc[s - 1] = 1
            reb_index = s - 1

        if len(weight_new) != 0:
            df = ut.record_return(df, s, reb_index, weight_new, leverage, trading_days)
            weights = ut.record_weights(df, s, weights)

        s += 1  # counting date
    perf = ut.comput_idicators(df, trading_days, 'perf.csv')
    weights.to_csv('weights_' + model_name + '.csv')
    print('back_test completed!')
