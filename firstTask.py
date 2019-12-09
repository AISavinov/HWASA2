import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
data = pd.read_excel('HWASA23.xlsx', dtype=float)
print(data.head(1))
data = data.apply(np.log)

Y = np.array(data['Y'])
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])
X3 = np.array(data['X3'])
X4 = np.array(data['X4'])


l1 = LinearRegression()
l1.fit(X1.reshape(-1, 1), Y)
print(l1.coef_, l1.intercept_)
print(r2_score(Y, l1.predict(X1.reshape(-1, 1))))

l2 = LinearRegression()
l2.fit(X2.reshape(-1, 1), Y)
print(l2.coef_, l2.intercept_)
print(r2_score(Y, l2.predict(X2.reshape(-1, 1))))

l3 = LinearRegression()
l3.fit(data.drop(axis=0,columns=['Y', 'X3', 'X4']), Y)
print(l3.coef_, l3.intercept_)
print(r2_score(Y, l3.predict(data.drop(axis=0,columns=['Y', 'X3', 'X4']))))

l4 = LinearRegression()
l4.fit(data.drop(axis=0,columns=['Y', 'X1']), Y)
print(l4.coef_, l4.intercept_)
print(r2_score(Y, l4.predict(data.drop(axis=0,columns=['Y', 'X1']))))