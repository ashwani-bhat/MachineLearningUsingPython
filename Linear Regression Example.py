import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:,np.newaxis,2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

reg = LinearRegression()

reg.fit(diabetes_X_train,diabetes_y_train)

print("Coefficient :\n ", reg.coef_)

print("Mean Square Error : %.2f" % np.mean((reg.predict(diabetes_X_test) -diabetes_y_test)**2))

print("Variance score: %.2f " %reg.score(diabetes_X_test,diabetes_y_test))

plt.scatter(diabetes_X_test,diabetes_y_test,color="black")
plt.plot(diabetes_X_test,reg.predict(diabetes_X_test),color = 'blue', linewidth = 3)

plt.xticks()
plt.yticks()
plt.show()