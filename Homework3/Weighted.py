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

# Boundary plot granularity
h=0.01

# Classifier Uniform weights

clf_uniform = neighbors.KNeighborsClassifier(n_neighbors=3, weights="uniform" )
clf_uniform.fit(X_train,Y_train)
prediction_uniform = clf_uniform.predict(X_test)
accuracy_uniform = clf_uniform.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Uniform weights')

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_uniform.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot

Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_uniform)
plt.show()
print("Uniform weights : "+str(accuracy_uniform*100))

#Classifier Distance weights

clf_distance = neighbors.KNeighborsClassifier(n_neighbors=3, weights="distance" )
clf_distance.fit(X_train,Y_train)
prediction_distance = clf_distance.predict(X_test)
accuracy_distance = clf_distance.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Distance')

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_distance.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot

Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_distance)
plt.show()
print("Distance weights' : "+str(accuracy_distance*100))