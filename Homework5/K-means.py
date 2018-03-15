from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from plot import means_5_plot,k_means_plot
#Data loading
digits = datasets.load_digits()

X = digits.data
y = digits.target

X = X[y<5]
y = y[y<5]

#Apply standardization and PCA
X = preprocessing.scale(X)
X_t = PCA(2).fit_transform(X)

#Plot Cluster X into 5 clusters using K-Means
kmeans = KMeans(5)
kmeans.fit(X_t)
means_5_plot(X_t, kmeans)

#Varying the number of clusters from 3 to 10

for i in range(3,11):
    if i==5:
        continue
    kmeans = KMeans(i)
    kmeans.fit(X_t)
    k_means_plot(i,X_t,kmeans)