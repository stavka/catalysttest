# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import json
import pandas as pd


import requests

from datetime import datetime

coins = [ 
         #'ada',
         'bch', 'btg', 'eos',
         'eth', 'ltc', 'ppt',
         #'xlm',
         'xrp', 'bcc', 'btc', 'dsh',
         'etc', 'iot', 'neo', 
         #'xem',
         'xmr', 'zec',]


alldata = {}

for coin in coins:
    
    filename = coin + 'all'
    js = open(filename + '.json').read()

    data = json.loads(js)
#data = pd.read_json(js)


#data['market_cap']

#print(data)

#print(data['price'][0:2])
#df = None

#for key, d in data.items():
#    if df is None:
#        print(d)
#        df = pd.DataFrame.from_records(d, columns=(0, key,))
#        df.set_index([0], inplace=True)
        

#df = [ pd.DataFrame.from_records(d, columns=(0, key,)).set_index([0], inplace=True ) 

    convert = lambda x: [ datetime.fromtimestamp(x[0]/1e3).date(), x[1] ]

    df = pd.concat( [ pd.DataFrame.from_records([ convert(data) for data in d], columns=('date', key,)).set_index(['date']) 
                  for key, d in data.items()  
               ], axis=1, join='inner' )

          
        
          
#    print df.head()
#    print df.tail()

    #df.to_csv(filename + '.csv')
    
    alldata[coin] = df
    
alldf = pd.concat(alldata)
alldf.index = alldf.index.rename([u'coin', u'date'])
alldf.to_csv('alldata.csv')
    

          
#print(df)              

#print(df.head())

#print(data['volume'][0:2])
#print(data['market_cap'][0:2])


#print(datetime.fromtimestamp(int(data['price'][0][0])/1000))
#print(datetime.fromtimestamp(int(data['price'][1][0])/1000))
#print(datetime.fromtimestamp(int(data['price'][2][0])/1000))



#print(data.keys())