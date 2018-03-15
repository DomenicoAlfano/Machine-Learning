from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap

imageFolderPath = '/Users/domenicoalfano/Università/Master/Machinelearning/Homework/Homework1/coil-100/'   #define image path folder.

firstchooses = input('Enter the four numbers of objects, is recommended to use a space between the numbers:') #User interaction: decide the 4 classes to process.


chooses=firstchooses.split(" ")   #create an array with the user input: chooses[0] = first dataset class chooses.

classes=[]

imageChoosesPath=[]

#take four classes from dataset and then, insert in imageChoosePath the images of the four classes.
for i in range(0,len(chooses)):
	imagePath=(glob.glob(imageFolderPath+'/obj'+str(chooses[i])+'__*'))
	classes.append(imagePath)
	imageChoosesPath=imageChoosesPath+(classes[i])

#Load in img_data all the images contained in imageChoosesPath.
img_data = np.asarray([np.asarray(Image.open(imageChoosesPath[i]).convert('L'), 'f')  for i in range(len(imageChoosesPath))])

X = img_data.ravel()        #with ravel create from a (288,128,128) matrix a (4718592,) vector.

y=np.zeros((0,1))

#each class will match with a different "y" color.
for i in range(0,len(chooses)):
	for j in range(0,len(imagePath)):
		y=np.append(y,i+1)

#Principal Component Visualization

X  = np.array(X).reshape(288, -1) #model x and create from (4718592,) vector a (288,16384) matrix

X = preprocessing.normalize(X)
X = preprocessing.scale(X)     # unity-variance and mean=0

X_t = PCA(2).fit_transform(X)  # Compute first and second PCA vectors

plt.scatter(X_t[:, 0], X_t[:, 1], c=y) 
plt.title('First and Second principal components', color='#000000')
plt.show()

#CLASSIFICATION First and Second principal component

X_train, X_test, y_train, y_test = train_test_split(X_t,y,test_size=0.50,random_state=100)    #divided my data into the training set and the test set.

clf = GaussianNB()

clf.fit(X_train,y_train)     #Fit gaussian naive bayes according X,Y.

prediction = clf.predict(X_test)  #Perform classification on an array of test vector
plt.scatter(X_test[:, 0], X_test[:,1],c=prediction)
plt.title('Classification of the First and the Second principal component', color='#000000')
plt.show()
accuracy=clf.score(X_test, y_test)
print("Accurancy of Classification of first two component: ", accuracy)

#Decision Boundaries of Classification
h=0.1
x_min, x_max = X_test[:, 0].min() -1, X_test[:, 0].max() +1
y_min, y_max = X_test[:, 1].min() -1, X_test[:, 1].max() +1
xx, yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#EEE8AA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:, 0], X_test[:,1],c=prediction)
plt.title('Classification of the First and the Second principal component with Decision Boundaries', color='#000000')
plt.show()

#Next two PRINCIPALCOMPONENT
# Compute the third and fourth PCA vectors
X_t = PCA(4).fit_transform(X)

plt.scatter(X_t[:, 2], X_t[:, 3], c=y) 
plt.title('Third and Fourth principal components', color='#000000')
plt.show()

#CLASSIFICATION Third and Fourth principal component

X_t = PCA(4).fit_transform(X)
X_t_1=np.array([X_t[:,2], X_t[:,3]])
X_t_1=np.transpose(X_t_1)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_t_1,y,test_size=0.50,random_state=0)    #divided my data into the training set and the test set.

clf_1 = GaussianNB()
clf_1.fit(X_train_1,y_train_1)     #Fit gaussian naive bayes according X,Y.
prediction_1= clf_1.predict(X_test_1)  #Perform classification on an array of test vector

plt.scatter(X_test_1[:, 0], X_test_1[:, 1],c=prediction_1)
plt.title('Classification of the Third and the Fourth principal component', color='#000000')
plt.show()
accuracy_1=clf_1.score(X_test_1, y_test_1)
print("Accurancy of Classification of next two component: ", accuracy_1)

#Decision Boundaries of Classification
'''
h=0.1
x1_min, x1_max = X_test_1[:, 0].min() -1, X_test_1[:, 0].max() +1
y1_min, y1_max = X_test_1[:, 1].min() -1, X_test_1[:, 1].max() +1
xx_1, yy_1=np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(y1_min, y1_max, h))
Z1 = clf.predict(np.c_[xx_1.ravel(), yy_1.ravel()])
Z1 = Z1.reshape(xx_1.shape)
plt.figure()
plt.pcolormesh(xx_1, yy_1, Z1, cmap=cmap_light)
plt.scatter(X_test_1[:, 0], X_test_1[:,1],c=prediction_1)
plt.title('Classification of the First and the Second principal component with Decision Boundaries', color='#000000')
plt.show()
'''
#10 e 11 PRINCIPALCOMPONENT
# Compute the tenth and eleventh PCA vectors

X_t = PCA(11).fit_transform(X)
plt.scatter(X_t[:, 9], X_t[:, 10], c=y) 
plt.title('Tenth and Eleventh principal components', color='#000000')
plt.show()

#Variance percentage of each component

X_t=PCA(n_components=20)
X_t.fit_transform(X)

variance=X_t.explained_variance_ratio_
plt.title('Percentage of variance explained by each of the selected components.', color='#000000')
plt.xlabel('n components')
plt.ylabel('% of variance explained')
plt.plot(variance)