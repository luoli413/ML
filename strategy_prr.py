from env_context import context
import numpy as np
import pandas as pd
from env_context import context

if __name__ == "__main__":

    variable_list = ['prr_ttm', 'eps_ttm', 'pb_lf', 'ps_ttm','np_growth_1y2']
    leverage = 0.95
    end_day = -1
    start_day = '2008-01-01'
    trading_days = 12.0 #because it's mnonthly data
    horizon = 1
    istech = 1
    large = True
    freq = 1
    roll = 6
    model_name ='Lasso'
    address = model_name+'_istech_large.csv'
    relative = True
    context = context(start_day, leverage, trading_days, )
    # context.pre_processing()
    context.import_trading_data()
    context.import_features(variable_list)
    # context.generate_train(horizon,relative,normalize =True)
    context.back_test(horizon,istech,large,freq,model_name,address,roll,threshold = 0.5)

