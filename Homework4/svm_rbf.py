import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from plot import plot_best_C, plot_Classification_C

#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_temp, X_test, y_temp, y_test = train_test_split(X,Y,test_size=0.3,random_state=69)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp,y_temp,test_size=0.14,random_state=69)

# Define usefull quantities

accuracy_vector=np.zeros((7,1))
C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
h=0.01

#Classification for C in [10^-3,10^3]
plt.figure(figsize=(20, 12))
for j, counter in enumerate(C_value):
    clf = svm.SVC(kernel='rbf', C=counter).fit(X_train,y_train)
    prediction = clf.predict(X_validation)
    plt.subplot(len(C_value),len(C_value),j+1)
    plot_Classification_C(prediction, clf, X_validation, y_validation, accuracy_vector, h, j, counter)

plt.show()

plt.plot(accuracy_vector)
plt.show()
print("\n\n"+str(accuracy_vector))

m=np.amax(accuracy_vector)
max=[i for i, j in enumerate(accuracy_vector) if j == m]
     
best_C=C_value[max[1]]
clf_best = svm.SVC(kernel='rbf',C=best_C).fit(X_train,y_train)

plot_best_C(best_C, clf_best, X_test, y_test, h)