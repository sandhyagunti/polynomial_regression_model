import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\DELL\Downloads\New folder\positions.csv")

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

lin_model_pred = lin_reg.predict([[7]])
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














