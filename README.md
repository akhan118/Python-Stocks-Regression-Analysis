#  Time Series Stocks-Regression-Analysis


Forcast stock price one day in the future .


Prediction_date can be more than one day in the future 😥🤯....


it uses time series data manipulation and sklearn lineaer model 



Let me know if you have questions .

   Usage Example :

    
    RegressionAnalysis import RegressionAnalysis

    Regression = RegressionAnalysis(self, start_date, end_date, symbol , prediction_date)
    Regression = RegressionAnalysis('04 3 2008  1:33PM','05 10 2018  5:33PM','F','05 20 2018  5:33PM')
    results= Regression.trainModels()
    print(results)
