from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from ..td.tdapi_test import Td
from datetime import datetime
import time

class RegressionAnalysis:


    def __init__(self, start_date, end_date, symbol , prediction_date):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.prediction_date = prediction_date
        tdapi = Td()
        start_date = datetime.strptime(start_date, '%m %d %Y %I:%M%p')
        end_date = datetime.strptime(end_date, '%m %d %Y %I:%M%p')
        df=tdapi.get_price_history(symbol,tdapi.unix_time_millis(start_date),tdapi.unix_time_millis(end_date))
        # df=tdapi.get_price_history('F')
        df['date'] = pd.to_datetime(df['datetime'],unit='ms')
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")
        df['date'] = pd.to_datetime(df['date'])
        self.data_set = df


    def trainModels(self,):
       # print(self.features)
       # print(self.features.iloc[:-1,:])
        data = self.data_set
        label = self.data_set
        # print(data)
        target = data
        label = label
        # print(target)
        zero_to_penultimate =target.iloc[:-1,:]
        one_to_end = label.iloc[1:,:]
        last_day_trading_day = label.iloc[1:,:].tail(1)
        # print(zero_to_penultimate)
        x= zero_to_penultimate[['close','low','high','open','datetime']]
        y= one_to_end[['close','low','high','open']]
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model_linear_regression = linear_model.LinearRegression(normalize=True, n_jobs=-1)
        # model_linear_regression.fit(x_train, y_train)
        model_linear_regression.fit(x, y)
        last_day_trading_day = last_day_trading_day[['close','low','high','open','datetime']]
        y_pred = model_linear_regression.predict(last_day_trading_day)
        # # print(y_pred)
        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred.rename(columns={0: 'close', 1: 'low', 2:'high' , 3:'open'})

        # test = [y_pred,y_test]
        # test = pd.concat([y_pred, y_test], keys=['GOOG', 'AAPL'], axis=1)
        # print(y_pred)
        return (y_pred,data)



#
# test = RegressionAnalysis('04 3 2008  1:33PM','05 10 2018  5:33PM','F','05 20 2018  5:33PM')
# test.trainModels()
