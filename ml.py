import pandas as pd
from sklearn import model_selection
import stockstats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import datetime as date
import requests as r
import iex
import numpy as np

'''Runs KNN using a range of neighbors and finds the one that yields the highest score'''
def bestKNNParams(X_train, X_val, Y_train, Y_val):
    neighMax = 0
    optNeigh = 1
    for nneighbors in range(1,30):
        clf = KNeighborsClassifier(n_neighbors=nneighbors).fit(X_train, Y_train)
        results = clf.score(X_val, Y_val)
        if results > neighMax:
            neighMax = results
            optNeigh = nneighbors
    weightMax = 0
    optWeight = ''
    for weights in ['uniform', 'distance']:
        clf = KNeighborsClassifier(weights=weights).fit(X_train, Y_train)
        results = clf.score(X_val, Y_val)
        if results > weightMax:
            weightMax = results
            optWeight = weights
    return optNeigh, optWeight

'''Normalization using STD'''
def normalizeSTD(df):
    norm = (df-df.mean())/df.std()
    return norm

'''Prints scores for the specified model'''
def getScore(classifier, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    valresults = classifier.score(X_val, Y_val)
    results = classifier.score(X_test, Y_test)
    print('Score: {}\nValidation score: {}'.format(results, valresults))
    Y_train_pred = classifier.predict(X_train)
    accScore = metrics.accuracy_score(Y_train, Y_train_pred)
    print('Accuracy Score:', accScore, sep=' ')
    precScore = metrics.precision_score(Y_train, Y_train_pred, average='weighted')
    print('Precision Score:', precScore, sep=' ')
    confMatrix = metrics.confusion_matrix(Y_train, Y_train_pred)
    print('Confusion Matrix:', confMatrix, sep='\n')
    return valresults, results, accScore, precScore, confMatrix

def getTechIndFeature(stockDF, indicator):
    ind = pd.DataFrame()
    try:
        ind = stockDF.get(indicator).to_frame()
        ind = ind.rename(columns={'':indicator})
        ind[indicator] = normalizeSTD(ind[indicator])
    except:
        pass
    return ind

def techIndicators(stockDF, indicators):
    merged = pd.DataFrame()
    validTechInd = False
    # Loops until all technical indicators entered are valid (stockstats lib)
    while not validTechInd:
        try:
            whatFeatures = indicators.split(',')
            featureList = list()
            for i in range(0, len(whatFeatures)):
                try:
                    featureList.append(getTechIndFeature(stockDF, whatFeatures[i]))
                    if featureList[-1].empty == True:
                        raise ValueError(whatFeatures[i])
                    validTechInd = True
                except ValueError as errInd:
                    print('{} is not a valid technical indicator'.format(errInd))
                    validTechInd = False
                    break
            if not validTechInd: continue

            merged = featureList[0]
            if len(featureList) > 1:
                for j in range(1, len(featureList)):
                    merged = merged.merge(featureList[j], right_index=True, left_index=True)
        except:
            pass
    return merged

def convertToStockStatsDF(df):
    df = stats.StockDataFrame.retype(df)
    return df

def closeLabel(df):
    close = df['close'].shift(-1).to_frame()
    close.rename(columns = {'':'close'})
    close['close'] = close['close'].pct_change()
    mean = close['close'].mean()
    std = 0.25*close['close'].std()

    for dates in close.index:
        if close.loc[dates, 'close'] > mean + std:
            close.loc[dates, 'close'] = 1
        elif close.loc[dates, 'close'] < mean - std:
            close.loc[dates, 'close'] = -1
        else:
            close.loc[dates, 'close'] = 0
    return close

def runModel(features, close):
    X_train, X, Y_train, Y = model_selection.train_test_split(features, close, train_size=0.7,
                                                                    test_size=0.3, shuffle=False)
    # VALIDATION #
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X, Y, train_size=0.5, test_size=0.5, shuffle=False)

    neighbors, weight = bestKNNParams(X_train, X_val, Y_train, Y_val)
    clf = KNeighborsClassifier(n_neighbors=neighbors, weights=weight).fit(X_train, Y_train)
    valresults, results, accScore, precScore, confMatrix = getScore(clf, X_train, X_val, X_test, Y_train, Y_val, Y_test)
    return valresults, results, neighbors, accScore, precScore, confMatrix

def dailyRoutine(ticker, fromDate, toDate, indicators):
    df = convertToStockStatsDF(iex.stockDailyData(ticker, fromDate, toDate))
    close = closeLabel(df)
    # merged = secondary(df, indicators) # need to fix this method...
    rsi=getTechIndFeature(df, 'rsi_14')
    macd=getTechIndFeature(df, 'macd')
    merged = rsi.merge(macd, left_index=True, right_index=True)
    # this merge doesnt work for some reason when using the merged from the techIndicators function
    merged = merged.merge(close, left_index=True, right_index=True)
    merged = merged.dropna()

    close = merged['close']
    features = merged.drop(columns='close')
    features = features.replace(np.inf, np.nan).dropna()
    return features, close

def minuteRoutine(ticker, fromDate, toDate, indicators):
    df = convertToStockStatsDF(iex.stockMinData(ticker, fromDate, toDate))
    close = closeLabel(df)
    # merged = secondary(df, indicators) # need to fix this method...
    rsi=getTechIndFeature(df, 'rsi_14')
    macd=getTechIndFeature(df, 'macd')
    merged = rsi.merge(macd, left_index=True, right_index=True)
    # this merge doesnt work for some reason when using the merged from the techIndicators function
    merged = merged.merge(close, left_index=True, right_index=True)
    merged = merged.dropna()

    close = merged['close']
    features = merged.drop(columns='close')
    features = features.replace(np.inf, np.nan).dropna()
    return features, close

#secondary techIndicators w/o try catch
def secondary(stockDF, indicators):
    df = stockDF
    indList = indicators.split(',')
    featureList = list()

    for i in range(0, len(indList)):
        featureList.append(getTechIndFeature(df, indList[i]))
    merged = featureList[0]
    if len(featureList) > 1:
        for j in range(1, len(featureList)):
            merged = merged.merge(featureList[j], left_index=True, right_index=True)
    return merged

# features, close = minuteRoutine('amd', '20180401', '20180601', 'macd,rsi_14,adx')

# valresults, results, neighbors, accScore, precScore, confMatrix = runModel(features, close)
