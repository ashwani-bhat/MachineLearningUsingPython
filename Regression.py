import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

data = quandl.get('WIKI/GOOGL')
data = data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close']*100.0
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open']*100.0
data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
data.fillna(-99999,inplace = True)

forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] =data[forecast_col].shift(-forecast_out)
data.dropna(inplace = True)

X = np.array(data.drop(['label'],1))
y = np.array(data['label'])

X = preprocessing.scale(X)
y= np.array(data['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)















