from sklearn import neighbors, datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Load data
iris=datasets.load_iris()
X=iris.data
Y=iris.target

#Standardizzation and PCA
X = preprocessing.scale(X)
X_2D = PCA(2).fit_transform(X)

#Train test split 

X_train, X_test, Y_train, Y_test = train_test_split(X_2D, Y, test_size=0.4, random_state=100) 
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1] , c=Y_test)
plt.show()