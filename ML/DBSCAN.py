import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler 
import sklearn.utils
import matplotlib.pyplot as plt 
import os.path
import wget
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

def randomData():
    X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)
    epsilon = 0.3
    minimumSamples = 7
    db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
    labels = db.labels_
    print(labels)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters)

    unique_labels = set(labels)
    print(unique_labels)

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

     # Plot the points with colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        # Plot the datapoints that are clustered
        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

        # Plot the outliers
        xy = X[class_member_mask & ~core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)
    plt.show(block=False)

    k = 3
    k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
    k_means3.fit(X)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(k), colors):
        my_members = (k_means3.labels_ == k)
        plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
    plt.show()

def weatherDataSet():
    filename = '/home/leonardo/Python/ML/weather-stations20140101-20141231.csv'

    if not(os.path.isfile(filename)):
        filename = wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv',
                    out=filename)
        print()
    
    pdf = pd.read_csv(filename)

    # Cleaning
    pdf = pdf[pd.notnull(pdf["Tm"])]
    pdf = pdf.reset_index(drop=True)
    print(pdf.head())

    rcParams['figure.figsize'] = (14,10)

    llon = -140
    ulon = -50
    llat = 40
    ulat = 65

    pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

    my_map = Basemap(projection='merc',
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color = 'white', alpha = 0.3)
    my_map.shadedrelief()

    # To collect data based on stations        

    xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
    pdf['xm']= xs.tolist()
    pdf['ym'] =ys.tolist()

    #Visualization1
    for index,row in pdf.iterrows():
    #   x,y = my_map(row.Long, row.Lat)
        my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
    #plt.text(x,y,stn)
    plt.show()

    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm','ym']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

    # Compute DBSCAN
    db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"]=labels

    realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels)) 

    # A sample of clusters
    print(pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5))

    print(set(labels))

    my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    #my_map.drawmapboundary()
    my_map.fillcontinents(color = 'white', alpha = 0.3)
    my_map.shadedrelief()

    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



    #Visualization1
    for clust_number in set(labels):
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]                    
        my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
        if clust_number != -1:
            cenx=np.mean(clust_set.xm) 
            ceny=np.mean(clust_set.ym) 
            plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
            print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))
    plt.show()

    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"]=labels

    realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels)) 

    my_map = Basemap(projection='merc',
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    #my_map.drawmapboundary()
    my_map.fillcontinents(color = 'white', alpha = 0.3)
    my_map.shadedrelief()

    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



    #Visualization1
    for clust_number in set(labels):
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]                    
        my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
        if clust_number != -1:
            cenx=np.mean(clust_set.xm) 
            ceny=np.mean(clust_set.ym) 
            plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
            print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))
    plt.show()

if __name__ == "__main__":
    # randomData()
    weatherDataSet()