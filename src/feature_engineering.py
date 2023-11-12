import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# For sensors datasets

def sensor_for_irradiation(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time'])
    target = res.pop("IRRADIATION")
    # Mathematical function
    res['relationshipWgoal'] = ( res['MODULE_TEMPERATURE'] / res['AMBIENT_TEMPERATURE'] ) - 1

    # Doing both a KMeans clustering horizontally & vertically
    km_V = KMeans(n_clusters=3, max_iter=350, n_init=50)
    km_H = KMeans(n_clusters=3, max_iter=350, n_init=50)
    # For vertical clustering, we're not selecting time_id as a parameter for the clustering
    res['vClustering'] = km_V.fit(res[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']]).labels_
    # For horizontal we add the time
    res['hClustering'] = km_H.fit(res[['time_id','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']]).labels_
    # In case of using tree based models, adding a sum of both clustering
    res['SumClusterings'] = res['vClustering'] + res['hClustering']

    # Add PCA analysis and retrieve only PC1
    pca = PCA()
    res_pca = pca.fit_transform(res)
    res_pca = pd.DataFrame(res_pca, columns=[f"PC{i+1}" for i in range(res_pca.shape[1])])
    res = res.merge(res_pca['PC1'], left_index=True, right_index=True)

    return res, target

def sensor_for_ambient_temp(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time'])
    target = res.pop("AMBIENT_TEMPERATURE")
    # Mathematical function
    res['relationshipWgoal'] = ( res['IRRADIATION'] + 1 ) / res['MODULE_TEMPERATURE']
    # Doing both a KMeans clustering horizontally & vertically
    km_V = KMeans(n_clusters=3, max_iter=350, n_init=50)
    km_H = KMeans(n_clusters=3, max_iter=350, n_init=50)
    # For vertical clustering, we're not selecting time_id as a parameter for the clustering
    res['vClustering'] = km_V.fit(res[['IRRADIATION','MODULE_TEMPERATURE']]).labels_
    # For horizontal we add the time
    res['hClustering'] = km_H.fit(res[['time_id','IRRADIATION','MODULE_TEMPERATURE']]).labels_
    # In case of using tree based models, adding a sum of both clustering
    res['SumClusterings'] = res['vClustering'] + res['hClustering']

    # Add PCA analysis and retrieve only PC1
    pca = PCA()
    res_pca = pca.fit_transform(res)
    res_pca = pd.DataFrame(res_pca, columns=[f"PC{i+1}" for i in range(res_pca.shape[1])])
    res = res.merge(res_pca['PC1'], left_index=True, right_index=True)
    
    return res, target

def sensor_for_module_temp(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time'])
    target = res.pop("MODULE_TEMPERATURE")
    # Mathematical function
    res['relationshipWgoal'] = ( res['IRRADIATION'] + 1 ) * res['AMBIENT_TEMPERATURE']
    # Doing both a KMeans clustering horizontally & vertically
    km_V = KMeans(n_clusters=3, max_iter=350, n_init=50)
    km_H = KMeans(n_clusters=3, max_iter=350, n_init=50)
    # For vertical clustering, we're not selecting time_id as a parameter for the clustering
    res['vClustering'] = km_V.fit(res[['IRRADIATION','AMBIENT_TEMPERATURE']]).labels_
    # For horizontal we add the time
    res['hClustering'] = km_H.fit(res[['time_id','IRRADIATION','AMBIENT_TEMPERATURE']]).labels_
    # In case of using tree based models, adding a sum of both clustering
    res['SumClusterings'] = res['vClustering'] + res['hClustering']

    # Add PCA analysis and retrieve only PC1
    pca = PCA()
    res_pca = pca.fit_transform(res)
    res_pca = pd.DataFrame(res_pca, columns=[f"PC{i+1}" for i in range(res_pca.shape[1])])
    res = res.merge(res_pca['PC1'], left_index=True, right_index=True)
    
    return res, target

def lone_sensor_for_irradiation(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE' ])
    target = res.pop("IRRADIATION")

    # Doing both a KMeans clustering horizontally & vertically
    km = KMeans(n_clusters=3, max_iter=350, n_init=50)
    # For vertical clustering, we're not selecting time_id as a parameter for the clustering
    # res['Clustering'] = km.fit(res[['time_id']]).labels_

    return res, target

def lone_sensor_for_ambient_temp(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time', 'MODULE_TEMPERATURE', "IRRADIATION"])
    target = res.pop("AMBIENT_TEMPERATURE")  
    return res, target

def lone_sensor_for_module_temp(df):
    res = df.drop(columns=["DATE_TIME", "date", 'time', 'AMBIENT_TEMPERATURE', "IRRADIATION"])
    target = res.pop("MODULE_TEMPERATURE")  
    return res, target
# For Generator datasets

# WIP
o
# --------------
