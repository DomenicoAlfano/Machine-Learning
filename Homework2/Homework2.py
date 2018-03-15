import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X_train=np.load("/Users/domenicoalfano/Università/Master/Machinelearning/Homework/Homework2/regression/regression_Xtrain.npy")
y_train=np.load("/Users/domenicoalfano/Università/Master/Machinelearning/Homework/Homework2/regression/regression_ytrain.npy")

X_test=np.load("/Users/domenicoalfano/Università/Master/Machinelearning/Homework/Homework2/regression/regression_Xtest.npy")
y_test=np.load("/Users/domenicoalfano/Università/Master/Machinelearning/Homework/Homework2/regression/regression_ytest.npy")

Lin_regression=LinearRegression()
Lin_regression.fit(X_train.reshape(-1,1), y_train)

prediction=Lin_regression.predict(X_test.reshape(-1,1))

mean_square_error=np.zeros((9,1))
mean_square_error[0]=mean_squared_error(y_test,prediction)
value = mean_square_error[0]

plt.plot(X_test, prediction , label="Model")
plt.scatter(X_test, y_test, label="Training Data")
plt.title('Degree '+str(1)+'\nMSE = '+str(value).strip("[]"))
plt.show()

for j in range(2,10):
    poly=PolynomialFeatures(degree = j , include_bias=False)
    polynomial_X_train = poly.fit_transform(X_train.reshape(-1,1))
    Polynomial_Regression = LinearRegression()
    Polynomial_Regression.fit(polynomial_X_train, y_train)
    polynomial_test=poly.fit_transform(X_test.reshape(-1,1))
    prediction_poly=Polynomial_Regression.predict(polynomial_test)
    mean_square_error[j-1] = mean_squared_error(y_test,prediction_poly)
    plt.plot(X_test, prediction_poly , label="Model")
    plt.scatter(X_test, y_test, c='r',label="Training Data")
    plt.title('Degree '+str(j)+'\nMSE = '+str(mean_square_error[j-1]).strip("[]"))
    plt.show()

plt.title("Mean Square error over polynomial degree")
plt.plot(np.linspace(1,9,9).reshape(-1,1),mean_square_error)
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()
