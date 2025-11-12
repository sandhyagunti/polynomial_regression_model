import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Projects\polynomial_regression_model\positions.csv")

x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear regression model(Linear Regression)')
plt.xlabel('Position Level')
plt.xlabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6]])
lin_model_pred

#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(x)

poly_reg.fit(X_poly,y)

lin_reg_2 =LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(x,y,color ='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('polymodel(Polynomial Regression)')
plt.xlabel('Position Level')
plt.xlabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[7]]))
poly_model_pred

#svr model

from sklearn.svm import SVR
svr_reg = SVR(kernel="sigmoid",degree=4)
svr_reg.fit(x,y)

svr_pred = svr_reg.predict([[6]])
print(svr_pred)


# knn model

from sklearn.neighbors import KNeighborsRegressor
knn_reg =KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(x,y)

knn_pred = knn_reg.predict([[6]])
print(knn_pred)

#decission tree

from sklearn.tree import DecisionTreeRegressor
dt_reg =DecisionTreeRegressor()
dt_reg.fit(x,y)

dt_pred = dt_reg.predict([[6]])
print(dt_pred)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators=27)
rf_reg.fit(x,y)

rf_pred = rf_reg.predict([[6]])
print(rf_pred)





