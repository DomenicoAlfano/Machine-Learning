from sklearn import neighbors, datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#UNWEIGHTED PART

#Load data
iris=datasets.load_iris()
X=iris.data
Y=iris.target

#Standardizzation and PCA
X = preprocessing.scale(X)
X_2D = PCA(2).fit_transform(X)

#Train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X_2D, Y, test_size=0.4, random_state=100) 

# Accuracy for all the K-neightbors case
accuracy_vector= np.zeros((10,1))

h=0.01 #granularity of the decision boundaries

# Prediction part

for k in range(1,11):
    KNN_classifier = neighbors.KNeighborsClassifier(n_neighbors=k)
    KNN_classifier.fit(X_train, Y_train)
    accuracy_vector[k-1] = KNN_classifier.score(X_test, Y_test)
    prediction = KNN_classifier.predict(X_test)
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	# + Plot data
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = KNN_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:,0], X_test[:,1] , c=prediction)
    plt.title(('KNN classification with ' + str(k) +' neighbors'))
    plt.show()
    print('Accuracy of the classification with ' + str(k) + ' neighbors : ' + str(accuracy_vector[k-1]))
    print('')
