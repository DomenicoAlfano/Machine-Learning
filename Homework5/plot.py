import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
from Purity import purity_score
import matplotlib.patches as mpatches


def gmm_plot(X_t, gmm, i, y, norm_mutual, homogeneity, purity):
    h = 0.1
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')
    plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
    
    # Plot the centroids as a white X 
    centroids = gmm.means_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    plt.title('GMM clustering on the digits dataset with number of clusters = '+str(i)+'\nCentroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    norm_mutual[i-2] = normalized_mutual_info_score(y,gmm.predict(X_t))
    homogeneity[i-2] = homogeneity_score(y,gmm.predict(X_t))
    purity[i-2]=purity_score(gmm.predict(X_t),y)
    print('Purity score: '+str(purity[i-2]))
    print('Normalized Mutual Information score: '+str(norm_mutual[i-2]))
    print('Homogeneity score: '+str(homogeneity[i-2]))
    
def means_5_plot(X_t, kmeans):
    h = 0.1
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')
    plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
    
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset withK = 5\n''Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
def k_means_plot(i, X_t, kmeans):
    h = 0.1
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')  
    plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
    
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset with K = '+str(i)+'\nCentroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
def evaluation_plot(norm_mutual,homogeneity,purity):
    k=np.linspace(0,100,100)
    plt.figure()
    plt.plot(k,norm_mutual)
    plt.plot(k,homogeneity)
    plt.plot(k,purity)
    red_patch = mpatches.Patch(color='red', label='Purity')
    blue_patch = mpatches.Patch(color='blue', label='NMI')
    green_patch = mpatches.Patch(color='green', label='Homogeneity')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0, handles=[red_patch,blue_patch,green_patch])
    plt.show()