import numpy as np 
import pandas as pd
import scipy
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs   
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import pylab
import os.path
import wget

def randomData():
    X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o') 
    plt.show()

    agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
    agglom.fit(X1, y1)

    # show cluster
    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(6,4))

    # These two lines of code are used to scale the data points down,
    # Or else the data points will be scattered very far apart.

    # Create a minimum and maximum range of X1.
    x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

    # Get the average distance for X1.
    X1 = (X1 - x_min) / (x_max - x_min)

    # This loop displays all of the datapoints.
    for i in range(X1.shape[0]):
        # Replace the data points with their respective cluster value 
        # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
        plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
                color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
        
    # Remove the x ticks, y ticks, x and y axis
    plt.xticks([])
    plt.yticks([])
    #plt.axis('off')

    # Display the plot of the original data before clustering
    plt.scatter(X1[:, 0], X1[:, 1], marker='.')
    # Display the plot
    plt.show()

    dist_matrix = distance_matrix(X1,X1) 
    print(dist_matrix)

    Z = hierarchy.linkage(dist_matrix, 'average')
    dendro = hierarchy.dendrogram(Z)
    plt.show()

def vecDataSet():
    filename = '/home/leonardo/Python/ML/cars_clus.csv'

    if not(os.path.isfile(filename)):
        filename = wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv',
                    out=filename)

    pdf = pd.read_csv(filename)
    print(pdf.shape)
    print(pdf.head())

    columns = list(pdf.columns[2:-1])
    print(columns)

    # drop rows with null value
    pdf[columns] = pdf[columns].apply(pd.to_numeric, errors='coerce')
    pdf.dropna(inplace=True)
    pdf.reset_index(drop=True, inplace=True)
    print(pdf.shape)
    print(pdf.head())

    featureset = pdf[columns[4:-1]]

    # normalize data
    x = featureset.values
    min_max_scaler = MinMaxScaler()
    feature_mtx = min_max_scaler.fit_transform(x)
    print(feature_mtx[0:5])

    # Using scipy

    leng = feature_mtx.shape[0]
    D = scipy.zeros([leng, leng])
    D = scipy.spatial.distance.cdist(feature_mtx, feature_mtx)

    Z = hierarchy.linkage(D, 'complete')

    max_d = 3
    clusters = hierarchy.fcluster(Z, max_d, criterion="distance")
    print(clusters)

    # k = 5
    # clusters = fcluster(Z, k, criterion='maxclust')

    fig = pylab.figure(figsize=(18,50))

    def llf(id):
        return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )

    dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
    plt.show()

    # Using scikit learn
    dist_matrix = distance_matrix(feature_mtx, feature_mtx)

    agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
    agglom.fit(feature_mtx)
    print(agglom.labels_)

    pdf['cluster_'] = agglom.labels_

    n_clusters = max(agglom.labels_)+1
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_labels = list(range(0, n_clusters))

    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(16,14))

    for color, label in zip(colors, cluster_labels):
        subset = pdf[pdf.cluster_ == label]
        for i in subset.index:
                plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
        plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
    #    plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')
    plt.show()

    pdf.groupby(['cluster_','type'])['cluster_'].count()
    agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
    print(agg_cars)

    plt.figure(figsize=(16,10))
    for color, label in zip(colors, cluster_labels):
        subset = agg_cars.loc[(label,),]
        for i in subset.index:
            plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')
    plt.show()


if __name__ == "__main__":
    # randomData()
    vecDataSet()