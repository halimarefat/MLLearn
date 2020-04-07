import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")
print(df.head())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color ='blue')
# plt.xlabel('Engine Size')
# plt.ylabel('Emission')
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
clf.fit(train_x_poly, train_y)

print('Coefficients (deg=2): ', clf.coef_)
print('Intercept (deg=2): ', clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='black')
XX = np.arange(0.0, 10.1, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
plt.plot(XX,yy, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_pred = clf.predict(test_x_poly)

print("Polynomial degree = 2")
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_pred - test_y)))
print("Residual sum of square (MSE): %.2f" % np.mean((test_y_pred - test_y) ** 2))
print("R2_score: %0.2f" % r2_score(test_y_pred, test_y))

poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
print(train_x_poly3)

clf3 = linear_model.LinearRegression()
clf3.fit(train_x_poly3, train_y)

print('Coefficients (deg=3): ', clf3.coef_)
print('Intercept (deg=3): ', clf3.intercept_)

XX = np.arange(0.0, 10.1, 0.1)
yy = clf3.intercept_[0] + clf3.coef_[0][1] * XX + clf3.coef_[0][2] * np.power(XX,2) + clf3.coef_[0][3] * np.power(XX,3)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'black')
plt.plot(XX, yy, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

test_x_poly3 = poly3.fit_transform(test_x)
test_y_pred3 = clf3.predict(test_x_poly3)

print("Polynomial degree = 3")
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_pred3 - test_y)))
print("Residual sum of square (MSE): %.2f" % np.mean((test_y_pred3 - test_y) ** 2))
print("R2_score: %0.2f" % r2_score(test_y_pred3, test_y))
