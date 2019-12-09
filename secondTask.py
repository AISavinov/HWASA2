import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
import matplotlib.pyplot as plt


data = pd.read_excel('HWASA23.xlsx', dtype=float)

Y = np.array(data['Y'])
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])
X3 = np.array(data['X3'])
X4 = np.array(data['X4'])

def plot_plys(deg):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(X1.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, Y)
    y_poly_pred = model.predict(x_poly)

    r2 = r2_score(Y, y_poly_pred)

    print(r2)

    plt.scatter(X1, Y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X1, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()

for i in [1,2,3,4,5]:
    plot_plys(i)

