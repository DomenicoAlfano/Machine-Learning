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

# Gaussian distance function

def GaussianD_01(d):
	return np.exp(- 0.1 *(d**2))

def GaussianD_10(d):
	return np.exp(- 10 *(d**2))

def GaussianD_100(d):
	return np.exp(- 100 *(d**2))

def GaussianD_1000(d):
	return np.exp(- 1000 *(d**2))

#Standardizzation and PCA
X = preprocessing.scale(X)
X_2D = PCA(2).fit_transform(X)

#Train test split 

X_train, X_test, Y_train, Y_test = train_test_split(X_2D, Y, test_size=0.4, random_state=100) 

# Decision Boundary granularity

h=0.01

#Classifier Gaussian weights

accuracy_gaussian=np.zeros((4,1))

# Alpha 0.01

clf_gaussian = neighbors.KNeighborsClassifier(n_neighbors=3, weights=GaussianD_01 )
clf_gaussian.fit(X_train,Y_train)
prediction_gaussian = clf_gaussian.predict(X_test)
accuracy_gaussian[0] = clf_gaussian.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Gaussian weights')
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_gaussian)
plt.title("Gaussian weight for alpha=0,1:")
plt.show()

# Alpha 10

clf_gaussian = neighbors.KNeighborsClassifier(n_neighbors=3, weights=GaussianD_10 )
clf_gaussian.fit(X_train,Y_train)
prediction_gaussian = clf_gaussian.predict(X_test)
accuracy_gaussian[1] = clf_gaussian.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Gaussian weights')
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_gaussian)
plt.title("Gaussian weight for alpha=10:")
plt.show()

# Alpha 100

clf_gaussian = neighbors.KNeighborsClassifier(n_neighbors=3, weights=GaussianD_100 )
clf_gaussian.fit(X_train,Y_train)
prediction_gaussian = clf_gaussian.predict(X_test)
accuracy_gaussian[2] = clf_gaussian.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Gaussian weights')
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_gaussian)
plt.title("Gaussian weight for alpha=100:")
plt.show()

# Alpha 1000

clf_gaussian = neighbors.KNeighborsClassifier(n_neighbors=3, weights=GaussianD_1000 )
clf_gaussian.fit(X_train,Y_train)
prediction_gaussian = clf_gaussian.predict(X_test)
accuracy_gaussian[3] = clf_gaussian.score(X_test,Y_test)
plt.figure()
plt.title('KNN 3 neightbors--Gaussian weights')
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# + Plot data
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction_gaussian)
plt.title("Gaussian weight for alpha=1000:")
plt.show()

print("Gaussian weight for alpha=0,1:", str(accuracy_gaussian[0]*100))
print("Gaussian weight for alpha=10:", str(accuracy_gaussian[1]*100))
print("Gaussian weight for alpha=100:", str(accuracy_gaussian[2]*100))
print("Gaussian weight for alpha=1000:", str(accuracy_gaussian[3]*100))