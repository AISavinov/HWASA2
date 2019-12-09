import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


data = pd.read_csv('Smarket.csv')
ax = sns.heatmap(data.drop(axis=0, columns=['Direction']).corr())
ax.plot()
#plt.show()

#print(data.head())

X = data.drop(axis=0, columns=['Today', 'Year', 'Direction'])
Y = [int(x == 'Up') for x in data['Direction']]
logit = sm.Logit(Y,X,missing='drop')
result = logit.fit()
#print(result.summary())
#print(result.predict(X))
print(accuracy_score(Y, [round(x) for x in result.predict(X)]))