'''Use this code to execute a portfolio optimization model. This code 
   will select the portfolio with the maximum Sharpe Ratio. The parameters 
   are set to use 180 days of historical data and rebalance every 30 days.
   
   This is the code used in the following article:
   https://blog.enigma.co/markowitz-portfolio-optimization-for-cryptocurrencies-in-catalyst-b23c38652556

   You can run this code using the Python interpreter:

   $ python portfolio_optimization.py
'''

from __future__ import division
import os
import pytz
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from catalyst.api import record, symbol, symbols, order_target_percent
from catalyst.utils.run_algo import run_algorithm
from catalyst.exchange.exchange_utils import get_exchange_symbols

np.set_printoptions(threshold='nan', suppress=True)



def universe(context, lookback_date, current_date):
    json_symbols = get_exchange_symbols(context.exchange)  # get all the pairs for the exchange
    universe_df = pd.DataFrame.from_dict(json_symbols).transpose().astype(str)  # convert into a dataframe
    universe_df['base_currency'] = universe_df.apply(lambda row: row.symbol.split('_')[1],
                                                                       axis=1)
    universe_df['market_currency'] = universe_df.apply(lambda row: row.symbol.split('_')[0],
                                                                         axis=1)

    # Filter all the exchange pairs to only the ones for a give base currency
    universe_df = universe_df[universe_df['base_currency'] == context.base_currency]

    # Filter all the pairs to ensure that pair existed in the current date range
    universe_df = universe_df[universe_df.start_date < lookback_date]
    universe_df = universe_df[universe_df.end_daily >= current_date]
    context.coins = symbols(*universe_df.symbol)  # convert all the pairs to symbols
    
    # print(universe_df.symbol.tolist())
    return universe_df.symbol.tolist()

def initialize(context):
   # Portfolio assets list
   #context.assets = symbols('btc_usdt', 'eth_usdt', 'ltc_usdt', 'dash_usdt',
   #                         'xmr_usdt')
   #context.assets = symbols(u'xrp_usd', u'etc_usd',      
   #                              u'rrt_usd', u'ltc_usd',
   #                              u'eth_usd', u'xmr_usd',
   #                              u'btc_usd', u'dsh_usd',
   #                              u'zec_usd', u'bcu_usd')
   # Set the time window that will be used to compute expected return 
   # and asset correlations
   context.coins = coins = [ 
         #'ada',
         'bch', 'btg', 'eos',
         'eth', 'ltc', 'ppt',
         #'xlm',
         'xrp', 'bcc', 'btc', 'dsh',
         'etc', 'iot', 'neo', 
         #'xem',
         'xmr', 'zec',]
   
   ### read vol and cap data
   
   context.volcapdata = pd.DataFrame.from_csv( 'voldata.csv', index_col = 1 )
   
       
   
   context.window = 7
   # Set the number of days between each portfolio rebalancing
   context.rebalance_period = 7                   
   context.i = 0
   context.base_currency = 'usd'
   context.exchange = context.exchanges.values()[0].name.lower()

   
def handle_data(context, data):
   # Only rebalance at the beggining of the algorithm execution and 
   # every multiple of the rebalance period
   if context.i%context.rebalance_period == 0:

       #date = data.current_dt.strftime('%Y-%m-%d')
       #lookback_date = data.current_dt - timedelta(days= context.window )
       #lookback_date = lookback_date.strftime('%Y-%m-%d')
       #context.universe = universe(context, lookback_date, date) 
       #context.assets = symbols(*context.universe)        
           
       
       #volume = data.history(context.assets, 
       #                       fields='volume', 
       #                      #fields='price',
       #                      bar_count= 7, frequency='1d')
       
       v = context.volcapdata[context.volcapdata['coin'].isin(context.coins)]
       weights = v[ v.index < data.current_dt ].groupby('coin').tail(7).groupby('coin').mean()['volume'].nlargest(10)
       
       
       #weights = volume.mean().nlargest(10) 
       weights = weights/weights.sum()
       
              #order optimal weights for each asset
       for coin in weights.index:
           assetname = coin+ "_" + context.base_currency
           try:
               asset = symbol(assetname)
               if data.can_trade(asset):
                   order_target_percent(asset, weights[coin])
           except:
               print (assetname + ' Not found')
       
       #order optimal weights for each asset
       #for asset in weights.index:
           #if data.can_trade(asset):
           #order_target_percent(asset, weights[asset])
           #order_target_percent(asset, max_sharpe_port[asset])
       

       record(weights=weights)
   context.i += 1
   
   
   
   
       
def analyze(context=None, results=None):
   ax1 = plt.subplot(211)
   results.portfolio_value.plot(ax=ax1)
   ax1.set_ylabel('portfolio value')
   ax2 = plt.subplot(212, sharex=ax1)
   results.sharpe.plot(ax=ax2)
   ax2.set_ylabel('sharpe ratio')
   plt.show() 

   daily_returns = (results.portfolio_value - results.portfolio_value.shift(1))/results.portfolio_value
   daily_vol = daily_returns.rolling(len(daily_returns), min_periods=1).std()
   
   ax1 = plt.subplot(211)
   daily_returns.plot(ax=ax1)
   ax1.set_ylabel('daily returns')
   ax2 = plt.subplot(212, sharex=ax1)
   #print daily_vol
   daily_vol.plot(ax=ax2)
   
   ax2.set_ylabel('daily vol')
   plt.show() 
   
   weekly_results= results.portfolio_value.resample('1W').last() 
   weekly_returns = (weekly_results - weekly_results.shift(1))/weekly_results
   weekly_vol = weekly_returns.rolling(len(weekly_returns), min_periods=1).std()
   
   ax1 = plt.subplot(211)
   weekly_returns.plot(ax=ax1)
   ax1.set_ylabel('weekly returns')
   ax2 = plt.subplot(212, sharex=ax1)
   #print daily_vol
   weekly_vol.plot(ax=ax2)
   
   ax2.set_ylabel('weekly vol')
   plt.show() 

   monthly_results= results.portfolio_value.resample('1M').last() 
   monthly_returns = (monthly_results - monthly_results.shift(1))/monthly_results
   monthly_vol = monthly_returns.rolling(len(monthly_returns), min_periods=1).std()
   
   ax1 = plt.subplot(211)
   monthly_returns.plot(ax=ax1)
   ax1.set_ylabel('monthly returns')
   ax2 = plt.subplot(212, sharex=ax1)
   #print daily_vol
   monthly_vol.plot(ax=ax2)
   
   ax2.set_ylabel('monthly vol')
   plt.show()    
   
   
   
   #results.portfolio_value.resample('1W').last()   
   #results.portfolio_value.resample('1M').last()
    
   # Form DataFrame with selected data
   data = results[['weights']]
   
   # Save results in CSV file
   filename = os.path.splitext(os.path.basename(__file__))[0]
   data.to_csv(filename + '.csv')


# Bitcoin data is available from 2015-3-2. Dates vary for other tokens.    
start = datetime(2017, 5, 30, 0, 0, 0, 0, pytz.utc)
end = datetime(2018, 1, 5, 0, 0, 0, 0, pytz.utc) 
#end = datetime(2017, 11, 30, 0, 0, 0, 0, pytz.utc) 
results = run_algorithm(initialize=initialize,
                        handle_data=handle_data,
                        analyze=analyze,
                        start=start,
                        end=end,
                        exchange_name='bitfinex',
                        capital_base=100000, )
