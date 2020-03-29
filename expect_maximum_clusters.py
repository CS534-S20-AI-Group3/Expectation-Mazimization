import csv
import math
import numpy
import random
from common_functions import *

# initial k_means function that finds clusters based on initial estimates
# initial estimates are chosen from existing points
def init_k_means(data, num_sets):
    clusters = []
    means = []

    for x in range(0, num_sets):
        clusters.append([])
        randMean = random.randint(0, len(data))
        means.append(data[randMean])

    clusters = create_clusters(data, means, clusters)
    return clusters

# Uses k-means to cluster the points into the specified number of clusters
# Ends when there is no change in the clusters
def k_means(data, clusters):
    prev_clusters = [point[:] for point in clusters]
    curr_clusters = []
    prev_means = []
    notSame = True

    while notSame:
        curr_clusters = []
        means_list = []
        for clust in prev_clusters:
            curr_clusters.append([])
            means_list.append(find_mean(clust))
        curr_clusters = create_clusters(data, means_list, curr_clusters)
        if prev_means == means_list:
            notSame = False
        else:
            prev_means = [point[:] for point in means_list]

    return (curr_clusters, means_list)

# Returns the mean of a cluster
def find_mean(cluster):
    x_vals = []
    y_vals = []
    for point in cluster:
        x_vals.append(point[0])
        y_vals.append(point[1])

    mean_point = [sum(x_vals)/len(x_vals), sum(y_vals)/len(y_vals)]
    return mean_point

# Separates the data into the clusters given the centers
def create_clusters(data, centers, clusters):
    for point in data:
        index = get_cluster(centers, point)
        clusters[index].append(point)
    return clusters

# Returns the closest cluster from the point
def get_cluster(means, point):
    dist_set = []
    for mean in means:
        dist_set.append(distance(mean, point))

    return dist_set.index(min(dist_set))

