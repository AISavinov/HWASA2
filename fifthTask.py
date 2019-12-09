from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import mean_squared_error

data = pd.read_excel('HDI.xlsx')
X = data.drop(axis=0, columns=['hdi'])
Y = data['hdi']
model = PCA(n_components=1)
model.fit(X)

#print(mean_squared_error(Y, model.transform(X)))
print(Y)
for x in model.transform(X):
    print(x[0])
