"""
Function list
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
# from cvxopt import solvers, matrix

# Function list******************************************************************************
# import data
def input_data(file_name):
    # deal with time format
    df = pd.read_csv(file_name, low_memory=False)
    df['datadate'] = pd.to_datetime(df['datadate'].values, format='%m/%d/%Y',
                                    infer_datetime_format=True, )
    if 'rdq' in df.columns:
        df['rdq'] = pd.to_datetime(df['rdq'].values, format='%m/%d/%Y',
                                   infer_datetime_format=True, )

    if 'tic' in df.columns:  # delete duplicated records of fd data
        df.drop_duplicates(subset=['datadate', 'tic'], keep='last', inplace=True)

    df.sort_values(by=['datadate'], inplace=True)
    df = df.set_index('datadate', drop=True)

    # fill in the na data
    if 'tic' in df.columns:
        stock_list = df['tic']
        stock_list = stock_list.drop_duplicates()
        s = 0
        for tics in stock_list.values:
            s += 1
            print(s, tics)
            indexer = df.loc[df['tic'] == tics, 'rdq'].isnull()
            # if rdq_date is missing, we replace it with quarter date
            if np.shape(df[df['tic'] == tics][indexer])[0] > 0:
                df.loc[(df['tic'] == tics) & indexer, 'rdq'] = \
                    df.loc[(df['tic'] == tics) & indexer, 'rdq'].index.values
                # print(df.loc[(df['tic'] == tics) & indexer, 'rdq'])
            df.loc[df['tic'] == tics, :] = df.loc[df['tic'] == tics, :].fillna(method='ffill')
    else:
        df = df.fillna(method='ffill')

    return df


def record_weights(df, i, weights):
    return_vector = np.log(df.iloc[i, :][weights.columns]) - np.log(df.iloc[i - 1, :][weights.columns])
    #   if the stock had not listed in market or no data available,we need set the return as zero
    return_vector[return_vector.isnull()] = 0.0
    sum_return = np.dot(return_vector, weights.iloc[i - 1, :].values)
    every_re = np.multiply(weights.iloc[i - 1, :], (return_vector + 1))
    weights.iloc[i, :] = every_re / (1 + sum_return)

    return weights


def record_return(df, i, reb_index, weight_new, leverage, trading_days=252.0,
                  interest_rate=2.5):
    cum_return = np.dot(np.log(df.iloc[i, :][weight_new.index.values]) - \
                        np.log(df.iloc[reb_index, :][weight_new.index.values]), weight_new.values)
    df['nav'].iloc[i] = df['nav'].iloc[reb_index] * (1 + cum_return * leverage + (1 - leverage) * (i - reb_index) \
                                                     * interest_rate / trading_days/100)
    return df


# computing indicators
def comput_idicators(df, trading_days, save_address, required=0.08, whole=1):
    # columns needed
    col = ['sp500', 'nav', 'rebalancing', 'stoploss', 'Interest_rate']
    df_valid = df.loc[:, col]
    start_balance = df.index[df['rebalancing'] == 1][0]
    df_valid = df_valid[df_valid.index >= start_balance]

    # daily return
    df_valid['return'] = np.log(df['nav']) - np.log(df['nav'].shift(1))
    # benchmark_net_value
    df_valid['benchmark'] = df_valid['sp500'] / df_valid['sp500'].iloc[0]
    # benchmark_return
    df_valid['benchmark_return'] = (df_valid['benchmark'] -
                                    df_valid['benchmark'].shift(1)) / \
                                   df_valid['benchmark'].shift(1)
    # Annualized return
    #      pd.expanding_mean(df_valid['return']) * trading_days
    df_valid['Annu_return'] = df_valid['return'].expanding(min_periods=1).mean() * trading_days
    # Volatility
    df_valid.loc[:, 'algo_volatility'] = df_valid['return']. \
                                             expanding(min_periods=1).std() * np.sqrt(trading_days)
    df_valid.loc[:, 'xret'] = df_valid['return'] - df_valid[
                                                       'Interest_rate'] / trading_days / 100
    df_valid.loc[:, 'ex_return'] = df_valid['return'] - df_valid['benchmark_return']

    def ratio(x):
        return np.nanmean(x) / np.nanstd(x)

    # sharpe ratio
    df_valid.loc[:, 'sharpe'] = df_valid['xret'].expanding(min_periods=1).apply(ratio) \
                                * np.sqrt(trading_days)
    # information ratio
    df_valid.loc[:, 'IR'] = df_valid['ex_return'].expanding().apply(ratio) \
                            * np.sqrt(trading_days)

    # Sortino ratio
    def modify_ratio(x, re):
        re /= trading_days
        ret = np.nanmean(x) - re
        st_d = np.nansum(np.square(x[x < re] - re)) / x[x < re].size
        return ret / np.sqrt(st_d)

    df_valid.loc[:, 'sortino'] = df_valid['return'].expanding().apply(modify_ratio,\
                                                         args=(required,)) * np.sqrt(trading_days)
    # Transfer infs to NA
    df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan
    df_valid.loc[np.isinf(df_valid.loc[:, 'IR']), 'IR'] = np.nan
    # hit_rate
    wins = np.where(df_valid['return'] >= df_valid[
        'benchmark_return'], 1.0, 0.0)
    df_valid.loc[:, 'hit_rate'] = wins.cumsum() / pd.Series(wins).expanding().apply(len)
    # 95% VaR
    df_valid['VaR'] = -df_valid['return'].expanding().quantile(0.05) * \
                      np.sqrt(trading_days)
    # 95% CVaR
    df_valid['CVaR'] = -df_valid['return'].expanding().apply(lambda x:\
                                                x[x < np.nanpercentile(x, 5)].mean())* np.sqrt(trading_days)

    if whole == 1:
        # max_drawdown
        def exp_diff(x, type):
            if type == 'dollar':
                xret = x.expanding().apply(lambda xx:(xx[-1] - xx.max()))
            else:
                xret = x.expanding().apply(lambda xx:(xx[-1] - xx.max()) / xx.max())
            return xret
            # dollar
            #     xret = exp_diff(df_valid['cum_profit'],'dollar')
            #     df_valid['max_drawdown_profit'] = abs(pd.expanding_min(xret))
            # percentage

        xret = exp_diff(df_valid['nav'], 'percentage')
        df_valid['max_drawdown_ret'] = abs(xret.expanding().min())

        # max_drawdown_duration:
        # drawdown_enddate is the first time for restoring the max
        def drawdown_end(x, type):
            xret = exp_diff(x, type)
            minloc = xret[xret == xret.min()].index[0]
            x_sub = xret[xret.index > minloc]
            # if never recovering,then return nan
            try:
                return x_sub[x_sub == 0].index[0]
            except:
                return np.nan

        def drawdown_start(x, type):
            xret = exp_diff(x, type)
            minloc = xret[xret == xret.min()].index[0]
            x_sub = xret[xret.index < minloc]
            try:
                return x_sub[x_sub == 0].index[-1]
            except:
                return np.nan

        df_valid['max_drawdown_start'] = pd.Series()
        df_valid['max_drawdown_end'] = pd.Series()
        df_valid['max_drawdown_start'].iloc[-1] = drawdown_start(
            df_valid['nav'], 'percentage')
        df_valid['max_drawdown_end'].iloc[-1] = drawdown_end(
            df_valid['nav'], 'percentage')
    df_valid.to_csv(save_address)

class context(object):
    def __init__(self, prices, fd, horizon, start_date, variable_list, symbols, end_date=-1, ):
        '''
        :param prices: daily price
        :param fd: fundamental data
        :param start_date: start_point of back_test
        :param variable_list: features
        :param symbols: stock
        :param end_date: test ending date
        '''
        self.start_date = start_date
        self.horizon = horizon
        self.prices = prices
        self.fd = fd
        if end_date != -1:
            self.prices = prices[prices.index <= end_date]
            self.fd = fd[fd.index <= end_date]
        self.price_date = self.prices.index.drop_duplicates()
        self.f_calendar = self.fd.index.drop_duplicates()
        self.variable_list = variable_list
        self.symbols = symbols
        self.allstock = self.prices.columns[2:]  # get rid of sp500 and interest_rate

    def generate_trains(self, relative=False, normalize=True, ):

        v_list = ['tic', 'rdq'] + self.variable_list
        fd_data = self.fd[v_list].copy()
        p_data = self.prices.loc[self.price_date, :]
        # ===== Deal with Y: future returns
        returns = (p_data.shift(-self.horizon) - p_data) / p_data

        if relative:
            ben = returns['sp500']
            returns = pd.DataFrame(np.subtract(np.array(returns),
                                               np.array(ben).reshape(len(ben), 1)),
                                   index=returns.index, columns=returns.columns)

        def normalized(x):
            x = pd.Series(x)
            clean_x = x[~x.isnull()]
            if len(x) > 3:
                miu = np.nanmedian(clean_x)
                sigma = np.nanstd(clean_x)
                # maxs = miu + 3*sigma
                # mins = miu-3*sigma
                if sigma > 0:
                    x = (x - miu) / sigma
                    x[(~x.isnull()) & ((x - miu) / sigma > 3)] = 3
                    x[(~x.isnull()) & ((x - miu) / sigma < -3)] = -3
            # print(len(x))
            return x

        # Deal with Xs: normalize in all stocks each quarter
        if normalize:
            for time in self.f_calendar.values:
                temp = fd_data.loc[time, self.variable_list]
                fd_data.loc[time, self.variable_list] \
                    = np.apply_along_axis(normalized, 0, np.array(temp))

        def append_y(x, re_st):
            dateindex = pd.to_datetime(re_st.index, infer_datetime_format=True)
            temp = re_st.index[dateindex >= pd.to_datetime(x)]

            if len(temp) > 0:
                return re_st.loc[temp[0]]
            else:
                return np.nan

        train = fd_data.copy()
        train.loc[:, 'y'] = pd.Series()
        for tics in self.allstock:
            re_st = returns[tics]
            rdq = train.loc[train['tic'] == tics, 'rdq']
            train.loc[train['tic'] == tics, 'y'] = rdq.apply(append_y, args=(re_st,))
            print(tics)
        print('generating train completed!')
        # train.to_csv('train.csv')
        self.train = train

    def extract_train(self, cur_date, roll=-1, ):
        '''
        :param cur_date:  current_time in back_test
        :param roll:    train_data rolling window
        :return:   x_train,y_train with datetiem index; x_test with tic index
        '''
        # avoid y_train horizon overlaps with future data
        cur_date = pd.to_datetime(cur_date)
        train_calendar = self.f_calendar[(pd.to_datetime(self.f_calendar) -
                                          cur_date).days < -(self.horizon / 21.0 + 1) * 31]
        train_date = self.price_date[pd.to_datetime(self.price_date) < cur_date]
        test_calendar = self.f_calendar[pd.to_datetime(self.f_calendar) < cur_date]
        #         print(test_calendar)
        if (roll == -1) or (roll > len(train_calendar)):
            pass
        else:
            train_calendar = train_calendar[-roll:]
        trains = self.train.loc[train_calendar.values, :].copy()
        #         print(trains)
        trains = trains.dropna(how='any', axis=0)
        y_train = trains['y']
        x_train = trains[self.variable_list]
        # TODO: add PCA on x_train
        tests = self.train.loc[test_calendar.values, :].copy()

        tests = tests.dropna(how='any', axis=0)
        # find the active stocks at current time
        p_data = self.prices.loc[train_date, :]
        stocks = p_data.columns[~p_data.iloc[-1, :].isnull()]
        valid_symbols = list(set(stocks).intersection(set(self.symbols)))

        tests = tests[np.isin(tests['tic'], valid_symbols)]

        x_test = pd.DataFrame()
        s = 0
        for tic in valid_symbols:
            # Some stocks may not update lately so we still use the latest but old data available
            test_temp = tests[tests['tic'] == tic]
            test_date = test_temp['rdq'].max()

            if s == 0:
                x_test = test_temp[test_temp['rdq'] == test_date]
            else:
                x_test = pd.concat([x_test,
                                    test_temp[test_temp['rdq'] == test_date]], ignore_index=True)
            s += 1
        x_test = x_test[self.variable_list + ['tic']]
        x_test = x_test.set_index(['tic'], drop=True)  # need know data belongs to whom
        return x_train, y_train, x_test

