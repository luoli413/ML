# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import utils as ut
import strategy as strats
path = os.getcwd()
data_path = os.path.join(path + '\\sc_data\\')

# Notes:
# eps earning per share
# np net_profit_growth
# or operation_revenue
# res research cost
# prr p/research
# pb p/asset
# pe p/eps
# ps p/revenue

def compute_indicators(df, save_address,trading_days, required=0.08, whole=1):
    # columns needed
    col = ['ben', 'nav', 'rebalancing', 'stoploss', 'Interest_rate']
    # df = self.book
    df_valid = df.loc[:, col]
    start_balance = df.index[df['rebalancing'] == 1][0]
    df_valid = df_valid[pd.to_datetime(df_valid.index) >= \
                        pd.to_datetime(start_balance)]

    # daily return
    df_valid['return'] = np.log(df['nav']) - np.log(df['nav'].shift(1))
    # benchmark_net_value
    df_valid['benchmark'] = df_valid['ben'] / df_valid['ben'].iloc[0]
    # benchmark_return
    df_valid['benchmark_return'] = (df_valid['benchmark'] -
                                    df_valid['benchmark'].shift(1)) / \
                                   df_valid['benchmark'].shift(1)
    # Annualized return
    #      pd.expanding_mean(df_valid['return']) * trading_days
    df_valid['Annu_return'] = df_valid['return'].expanding(min_periods=1).mean() * trading_days
    # Volatility
    df_valid.loc[:, 'algo_volatility'] = df_valid['return'].\
                                             expanding(min_periods=1).std() * np.sqrt(trading_days)
    df_valid.loc[:, 'xret'] = df_valid['return'] - \
                              df_valid['Interest_rate'] / trading_days / 100
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

    df_valid.loc[:, 'sortino'] = df_valid['return'].expanding().\
                                     apply(modify_ratio, args=(required,)) * np.sqrt(trading_days)
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
    df_valid['CVaR'] = -df_valid['return'].expanding().apply(lambda x: \
                np.nanmean(x[x < np.nanpercentile(x,5)])) * np.sqrt(trading_days)

    if whole == 1:
        # max_drawdown
        def exp_diff(x, type):
            if type == 'dollar':
                xret = x.expanding().apply(lambda xx: (xx[-1] - xx.max()))
            else:
                xret = x.expanding().apply(lambda xx: (xx[-1] - xx.max()) / xx.max())
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
    def __init__(self,start_day,leverage,trading_days,end_day=-1,):
        self.lists = ['dividendyield2','eps_ttm','pb_lf','mkt_cap_float','ps_ttm','res_ttm','or_ttm2',\
         'np_growth_1y2','industry_d2','pe_ttm','zz500','prr_ttm','trade_status','un_st_flag','ipo_listdays',\
           'or_growth_1y' ]
        # tech_list
        self.tech = ['化工', '通信', '采掘', '建筑材料', '轻工制造', '电气设备', '建筑装饰', '汽车', '钢铁', '有色金属', '农林牧渔',
                '纺织服装', '食品饮料', '计算机', '电子', '传媒', '国防军工', '家用电器', '医药生物', '机械设备']
        # non_tech_list
        self.non_tech = ['房地产', '银行', '金融服务','综合', '休闲服务', '商业贸易', '非银金融', '交通运输', '公用事业']
        self.start_day = start_day
        self.end_day = end_day
        self.trading_data_list = ['industry_d2','close','trade_status','un_st_flag',\
                                  'ipo_listdays','or_growth_1y','mkt_cap_float']
        self.leverage = leverage
        self.trading_days = trading_days

    def pre_processing(self):
        for i in self.lists:
            A = pd.read_excel(os.path.join(data_path + i+'.xlsx'))
            if i=='industry_d2':
                # print(A.head())
                def set_flag(x,bool):
                    if bool:
                        x[np.isin(x.values,self.tech)]=1
                    else:
                        x[np.isin(x.values, self.non_tech)] = 0
                    return x
                A.apply(set_flag,axis = 0,args=(True,))
                A.apply(set_flag,axis = 0,args=(False,))
                # print(A.head())
            if i== 'trade_status':
                # print(A.head())
                A[(A == '交易') | (A == '停牌1小时')] = 1
                A[A != 1] = np.nan

            A.T.to_csv(os.path.join(data_path + i+'.csv'),)
        print('preprocessing completed!')

    def import_trading_data(self):
        self.context_dict = dict()
        for i in self.trading_data_list:
            temp = pd.read_csv(os.path.join(data_path + i + '.csv'))
            temp_col = temp.columns.values
            temp_col[0] = 'Date'
            temp.columns = temp_col
            temp['Date'] = pd.to_datetime(temp['Date'])
            temp.sort_values(['Date'], inplace=True)
            temp.set_index(['Date'], drop=True, inplace=True)
            self.context_dict[i] = temp
        print('import trading data completed!')

    def import_features(self, variable_list):
        self.variable_list=variable_list
        panel_dict = dict()
        for i in variable_list:
            temp = pd.read_csv(os.path.join(data_path + i + '.csv'))
            temp_col = temp.columns.values
            temp_col[0] = 'Date'
            temp.columns = temp_col
            temp['Date'] = pd.to_datetime(temp['Date'])
            temp.sort_values(['Date'],inplace=True)
            temp.set_index(['Date'], drop=True, inplace=True)
            panel_dict[i] = temp
        indicator_panel = pd.Panel.from_dict(panel_dict)
        xx = indicator_panel.to_frame()
        xx.reset_index(inplace=True)
        xx.set_index(['Date'], inplace=True, drop=True)
        temp = xx.columns.values
        temp[0] = 'tic'
        xx.columns = temp
        # print(xx)
        self.features = xx
        print('import features completed!')

    def generate_train(self,horizon,relative,normalize =False):

        # v_list = ['tic', 'rdq'] + self.variable_list
        v_list = ['tic'] + self.variable_list
        fd_data = self.features[v_list].copy()
        p_data = self.context_dict['close'].copy()
        if self.end_day!=-1:
            end_day = pd.to_datetime(self.end_day)
            p_data = p_data[pd.to_datetime(p_data.index) <= end_day]
            fd_data = fd_data[pd.to_datetime(fd_data.index) <= end_day]
        f_calendar = fd_data.index.drop_duplicates()
        # symbols = p_data.columns[1:]
        cols = self.context_dict['ipo_listdays'].columns
        symbols = cols[self.context_dict['ipo_listdays'].iloc[-1,:]>90] # at least listed for 1 quarters
        fd_data = fd_data[np.isin(fd_data['tic'],symbols)]
        # ===== Deal with Y: future returns
        returns = (p_data.shift(-horizon) - p_data) / p_data
        if relative:
            ben = returns['zz500']
            returns = pd.DataFrame(np.subtract(np.array(returns),
                                               np.array(ben).reshape(len(ben), 1)),
                                   index=returns.index, columns=returns.columns)

        def normalized(x):
            x = pd.Series(x)
            clean_x = x[~x.isnull()]
            if len(clean_x) > 3:
                miu = np.nanmedian(clean_x)
                sigma = np.nanstd(clean_x)
                if sigma > 0:
                    x = (x - miu) / sigma
                    x[(~x.isnull()) & (x > 3)] = 3
                    x[(~x.isnull()) & (x < -3)] = -3
            # print(len(x))
            return x

        # Deal with Xs: normalize in all stocks each quarter
        if normalize:
            for time in f_calendar.values:

                temp = fd_data.loc[time, self.variable_list]
                # print(temp.shape)
                if len(temp.shape)>1:
                    fd_data.loc[time, self.variable_list] \
                        = np.apply_along_axis(normalized, 0, np.array(temp))

        def append_y(x, re_st,):
            dateindex = pd.to_datetime(re_st.index, infer_datetime_format=True)
            temp = re_st.index[dateindex >= pd.to_datetime(x)]

            if len(temp) > 0:
                return re_st.loc[temp[0]]
            else:
                return np.nan

        # def apply_append_y(x,returns,train):
        #     re_st = returns[x]
        #     rdq = pd.Series(train.loc[train['tic'] == x, :].index, \
        #                     index=train.loc[train['tic'] == x, :].index)
        #     if len(rdq) > 0:
        #         train.loc[train['tic'] == x, 'y'] = rdq.apply(append_y, args=(re_st,))
        #         print(x)
        #     return train

        train = fd_data.copy()
        train.loc[:, 'y'] = pd.Series()
        # train = pd.Series(symbols).apply(apply_append_y, args=(returns, train,))
        for tics in symbols:
            re_st = returns[tics]
            rdq = pd.Series(train.loc[train['tic'] == tics, :].index, \
                            index=train.loc[train['tic'] == tics, :].index)
            if len(rdq) > 0:
                train.loc[train['tic'] == tics, 'y'] = rdq.apply(append_y, args=(re_st,))
                print(tics)
        self.train = train
        train.to_csv('trains.csv')
        print('generating train completed!')

    def extract_train(self,cur_date,istech,horizon,roll=-1,large=True,top=True):
        # select industry&non-st&trade_status
        def select_step1(context_dict,f_calendar):
            keys = ['un_st_flag', 'trade_status',]
            s=0
            for i in keys:
                fd = context_dict[i]
                indexing = f_calendar[-1]
                if s==0:
                    symbols = fd.columns[fd.loc[indexing, :] == 1].values
                else:
                    temp = fd.columns[fd.loc[indexing, :] == 1].values
                    symbols = list(set(symbols).intersection(set(temp)))
                s+=1
            return symbols

        def select_stocks_case1(context_dict,f_calendar,istech,large=True):
            keys = ['industry_d2','mkt_cap_float']
            symbols = select_step1(context_dict,f_calendar)
            indexing = f_calendar[-1]
            for i in keys:
                fd = context_dict[i]
                fd = fd.loc[:,symbols]
                if i == 'industry_d2':
                    # istech
                    if istech:
                        symbols = fd.columns[fd.loc[indexing, :] == 1].values
                    else:
                        symbols = fd.columns[fd.loc[indexing, :] == 0].values
                else:
                    per = 20
                    temp_cap = fd.loc[indexing, :]
                    if large:
                        temp = temp_cap[temp_cap > np.nanpercentile(temp_cap, (100-per))].index.values
                    else:
                        temp = temp_cap[temp_cap < np.nanpercentile(temp_cap, per)].index.values
                    symbols = list(set(symbols).intersection(set(temp)))
            return symbols

        def select_stocks_case2(context_dict,f_calendar,top):
            df = context_dict['or_growth_1y']
            indexing = f_calendar[-1]
            symbols = select_step1(context_dict,f_calendar)
            df = df.loc[indexing,symbols]
            per = 20
            if top:
                symbols = df[df > np.nanpercentile(df, 100-per)].index.values
            else:
                symbols = df[df<np.nanpercentile(df,per)].index.values
            return symbols

        cur_date = pd.to_datetime(cur_date)
        v_list = ['tic','y'] + self.variable_list
        train = pd.read_csv('trains.csv')
        train.sort_values(['Date'],inplace=True)
        train.set_index('Date',drop=True,inplace=True)
        # train = self.train[v_list].copy()
        train = train[v_list].copy()
        train = train[pd.to_datetime(train.index) < cur_date]
        bool = False
        if not train.empty:
            f_calendar = train.index.drop_duplicates()

            # rolling
            if (roll == -1) or (roll+1 > len(f_calendar)):
                pass
            else:
                f_calendar = f_calendar[-(roll+1):]

            if len(f_calendar)>=2:
                # select stocks
                symbols = select_stocks_case1(self.context_dict, f_calendar, istech, large=large)
                train_all = train[np.isin(train['tic'], symbols)]
                train_calendar = f_calendar[(pd.to_datetime(f_calendar) -
                                             cur_date).days <= -(horizon+1) * 31.0] #lag back
                if len(list(set(train_calendar).intersection(set(train_all.index.values))))>0:

                    train = train_all[np.isin(train_all.index.values,train_calendar.values)].copy()
                    train.dropna(how='any', axis=0,inplace=True)
                    self.y_train = train['y']
                    self.x_train = train[self.variable_list]

                    x_test = pd.DataFrame()
                    s = 0
                    for tic in symbols:
                        # Some stocks may not update lately so we still use the latest but old data available
                        test_temp = train_all[train_all['tic'] == tic]
                        if len(test_temp)>0:# fd_data does not cover some stocks in certain early date
                            test_date = test_temp.index[-1]
                            # print(test_date)
                            if s == 0:
                                x_test = test_temp.loc[test_date,:]
                            else:
                                x_test = pd.concat([x_test,
                                                    test_temp.loc[test_date,:]],axis=1)
                            s += 1
                    print(s,'stocks completed in x_test')
                    x_test = x_test.T
                    if np.shape(x_test)[0]>0:
                        x_test.set_index(['tic'], drop=True,inplace = True)# need know data belongs to whom
                        x_test.drop(['y'],axis = 1,inplace =True)
                        x_test.dropna(how='any', axis=0,inplace=True)
                        self.x_test = x_test
                        bool = True
                        # return bool
        return bool

    def back_test(self,horizon,istech,large,freq,model_name,address,roll=-1,threshold=0.1):
        # initial setting
        df = self.context_dict['close'].copy()
        symbols = self.context_dict['close'].columns[1:]
        cols = self.context_dict['close'].columns.values
        cols[0]='ben'
        df.columns = cols# set zz500 as benchmark
        stock_num = len(symbols)
        back_testing = df.index[pd.to_datetime(df.index) >= pd.to_datetime(self.start_day)]
        df = df.loc[back_testing.values, :]
        unit = np.full((len(df.index), 1), 1)[:, 0]

        df['rebalancing'] = pd.Series()
        df['stoploss'] = pd.Series()
        df['nav'] = pd.Series(unit, index=df.index)
        df['Interest_rate'] = pd.Series(np.full((len(df.index),), 2.5),index=df.index) # 2.5% interest_rate
        weight_new = []
        # max_new = []  # for computing max_drawdown
        unit = np.full((len(df.index), stock_num), 0)
        weights = pd.DataFrame(unit, index=df.index, columns=symbols)
        reb_index = 0
        s = 0  # counting date
        # ============================= Enter Back-testing ===================================
        for cur_date in back_testing.values:

            cur_date = pd.to_datetime(cur_date)

            # rebalance in a fixed frequency in freq rate
            if s>0:# begin to rebalance at least after the second recordings
                if np.mod(s, freq) == 0:
                    # print(s)
                    if self.extract_train(cur_date, istech,horizon,large =large,roll=roll,):
                        if np.shape(self.x_test)[0]>0:
                            test_y = strats.model(model_name, self.x_train, self.y_train, self.x_test)
                            weight_new = strats.fix_stock_order(test_y, threshold)

                            print(df.index[s - 1])
                            print(weight_new[weight_new != 0].head())
                            weights.loc[df.index[s - 1], weight_new.index] = weight_new.values
                            print('*' * 50)
                            df['rebalancing'].iloc[s - 1] = 1
                            reb_index = s - 1

                if len(weight_new) != 0:
                    df = ut.record_return(df, s, reb_index, weight_new, self.leverage, self.trading_days)
                    weights = ut.record_weights(df, s, weights)

            s += 1 # counting date
        compute_indicators(df, 'perf_'+address,self.trading_days)
        weights.to_csv('weights_' + address)
        print('back_test completed!')

