import numpy as np
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from bottleneck import bottleneck2D
from sklearn.model_selection import GridSearchCV
from plot import plot_best_params,Classification_params_Validation
import matplotlib.pyplot as plt

#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_temp, X_test, y_temp, y_test = train_test_split(X,Y,test_size=0.3,random_state=69)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp,y_temp,test_size=0.14,random_state=69)

# Define usefull quantities

accuracy_vector=np.zeros((7,5))
C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
gamma_value = np.array([10**-2,10**-1,1,10,10**2])
h=0.01

#Classification for C and Gamma with Validation

for j, counter in enumerate(C_value):
    plt.figure(figsize=(20, 12))
    for i, counter2 in enumerate(gamma_value):
        Classification_params_Validation(counter,counter2,X_train,y_train,X_validation,y_validation,accuracy_vector,h,i,j,C_value, gamma_value)
    plt.show()


#Check the max value in the accurancy_vector
        
max = bottleneck2D(accuracy_vector)

#Evaluate the best parameters

param_grid = dict(gamma=gamma_value, C=C_value)
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid).fit(X_train, y_train)
print("The best parameters are %s with a accuracy of %0.2f" % (grid.best_params_, accuracy_vector[max[0],max[1]]))

plot_best_params(grid,X_train,y_train,X_test,y_test,h)