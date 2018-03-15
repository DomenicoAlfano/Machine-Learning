from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.model_selection import cross_val_score

def plot_best_params(grid, X_train, y_train, X_test, y_test, h):
    best_C=grid.best_params_['C']
    best_gamma=grid.best_params_['gamma']
    clf = svm.SVC(kernel='rbf',gamma=best_gamma,C=best_C).fit(X_train,y_train)
    prediction=clf.predict(X_test)
    accuracy=clf.score(X_test,y_test)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:,0], X_test[:,1] , c=prediction)
    plt.title('C=' +str(best_C) + '   Gamma=' +str(best_gamma))
    plt.show()
    print('Accuracy: '+(str(accuracy*100)+ '%'))
    
    
def plot_best_C(best_C, clf_best, X_test, y_test, h):
    prediction_test=clf_best.predict(X_test)
    accuracy_test=clf_best.score(X_test,y_test) 
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf_best.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_test)
    plt.title('C=' +str(best_C))
    plt.show()
    print("\n\nAccuracy on test data : "+str(accuracy_test*100)+" % \n")
    
def plot_Classification_C(prediction, clf, X_validation, y_validation, accuracy_vector, h, j, counter):
    accuracy_vector[j]=clf.score(X_validation,y_validation)
    x_min, x_max = X_validation[:, 0].min() - 1, X_validation[:, 0].max() + 1
    y_min, y_max = X_validation[:, 1].min() - 1, X_validation[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_validation[:,0], X_validation[:,1] , c=prediction)
    plt.title('C=' +str(counter))

def Classification_params_Validation(counter, counter2, X_train, y_train, X_validation, y_validation, accuracy_vector, h, i, j, C_value, gamma_value):
    clf = svm.SVC(kernel='rbf',gamma=counter2, C=counter).fit(X_train,y_train)
    prediction = clf.predict(X_validation)
    accuracy_vector[j,i]=clf.score(X_validation,y_validation)
    x_min, x_max = X_validation[:, 0].min() - 1, X_validation[:, 0].max() + 1
    y_min, y_max = X_validation[:, 1].min() - 1, X_validation[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.subplot(len(C_value), len(gamma_value), i + 1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_validation[:,0], X_validation[:,1] , c=prediction)
    plt.title('C=' +str(counter) + '   Gamma =' +str(counter2))
    
def Classification_params_KFolds(counter2,counter,X_train,y_train,X_test,scores,h,i,j,C_value,gamma_value):
    clf = svm.SVC(kernel='rbf', gamma=counter2, C=counter).fit(X_train,y_train)
    prediction = clf.predict(X_test)
    scores[j,i,:]=cross_val_score(clf, X_train, y_train, cv=5)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = (ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA']))
    plt.subplot(len(C_value), len(gamma_value), i + 1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:,0], X_test[:,1] , c=prediction)
    plt.title('C=' +str(counter) + '   Gamma =' +str(counter2))