import requests as r
import datetime as date
import pandas as pd

def validDate(fromDate, toDate):
    if fromDate > toDate:
        print('Please enter a valid date...')
        return False
    else:
        return True

def stockMinData(stock, fromDate, toDate):
    if not validDate(fromDate, toDate):
        return
    URL = []
    #dot = '.'
    while fromDate <= toDate:
        URL += r.request('GET', 'https://api.iextrading.com/1.0/stock/' + stock + '/chart/date/' + fromDate).json()
        fromDate = incDate(fromDate)
     #   print(dot)
     #   dot += '.'
     #   if len(dot) > 10:
     #       dot = '.'
    df = pd.DataFrame(URL)
    return df

def stockDailyData(stock, fromDate, toDate):
    if not validDate(fromDate, toDate):
        return
    URL = r.request('GET', 'https://api.iextrading.com/1.0/stock/' + stock + '/chart/5y').json()
    df = pd.DataFrame(URL)
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
    df = df[date.datetime.strptime(fromDate, '%Y%m%d'):date.datetime.strptime(toDate, '%Y%m%d')]
    return df

def incDate(convDate):
    convDate = date.datetime.strptime(convDate, '%Y%m%d')
    convDate += date.timedelta(days=1)
    convDate = convDate.strftime('%Y%m%d')
    return convDate