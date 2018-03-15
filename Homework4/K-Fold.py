import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from bottleneck import bottleneck3D
from plot import plot_best_params,Classification_params_KFolds
#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=50)

# Define usefull quantities

C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
gamma_value = np.array([10**-2,10**-1,1,10,10**2])
h=0.01
scores = np.zeros((7,5,5))

#Classification for C and Gamma with K-Folds=5

for j, counter in enumerate(C_value):
    plt.figure(figsize=(20, 12))
    for i, counter2 in enumerate(gamma_value):
        Classification_params_KFolds(counter2,counter,X_train,y_train,X_test,scores,h,i,j,C_value,gamma_value)
    plt.show()

#Check the max value in the scores matrix
max = bottleneck3D(scores)
            
#Evaluate the best parameters

param_grid = dict(gamma=gamma_value, C=C_value)
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=5).fit(X_train, y_train)
print("The best parameters are %s with a accuracy of %0.2f" % (grid.best_params_, scores[max[0],max[1],max[2]]))

plot_best_params(grid,X_train,y_train,X_test,y_test,h)