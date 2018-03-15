from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn import mixture
from plot import gmm_plot,evaluation_plot

#Data loading
digits = datasets.load_digits()

X = digits.data
y = digits.target

X = X[y<5]
y = y[y<5]

#Apply standardization and PCA
X = preprocessing.scale(X)
X_t = PCA(2).fit_transform(X)

#Varying the number of clusters from 2 to 10
norm_mutual=np.zeros((100,1))
homogeneity=np.zeros((100,1))
purity=np.zeros((100,1))

for i in range(2,101):
    gmm = mixture.GaussianMixture(n_components=i,covariance_type='full').fit(X_t)
    gmm_plot(X_t,gmm,i,y,norm_mutual,homogeneity,purity)

#Evaluation Plot
evaluation_plot(norm_mutual,homogeneity,purity)