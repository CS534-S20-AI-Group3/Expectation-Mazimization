import csv
import math
import numpy as np
import random
from common_functions import *
import time
from matplotlib import pyplot as plt

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
        print("prev means",prev_means)
        print("means lis",means_list)
        if np.array_equal(prev_means,means_list):
            notSame = False
        else:
            prev_means = [point[:] for point in means_list]

    return (curr_clusters, means_list)

# Returns the mean of a cluster
def find_mean(cluster):
    vals = np.array(cluster[0])
    for point in cluster:
        vals = vals + np.array(point)
        # x_vals.append(point[0])
        # y_vals.append(point[1])

    #mean_point = [sum(x_vals)/len(x_vals), sum(y_vals)/len(y_vals)]
    mean_point = vals/float(len(cluster))
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
def k_means_clustering(data_set, num_clust):
    start_ktime = time.perf_counter()
    print("Running K means")
    init_clusters = init_k_means(data_set, int(num_clust))
    kmean_clusters = k_means(data_set, init_clusters)
    clusters = kmean_clusters[0] # clusters generated by k-means clustering
    means = kmean_clusters[1] # means generated by k-means clustering
    print("K means completed")
    print("K means clusters")
    for i in clusters:
        print(i)
    print("K means means")
    for m in means:
        print(m)
    end_ktime = time.perf_counter()
    diff_ktime = end_ktime - start_ktime
    print("time taken to do k means",diff_ktime)
    #plotting the k means if dimension is 2

    # color_bar = []
    # for e in range(len(means)):
    #     f = (random.uniform(0.1, 1), random.uniform(0, 1), random.uniform(0, 1))
    #     color_bar.append(f)
    # plt.figure(1)
    # ax1 = plt.subplot()
    # i = 0
    # for c in clusters:
    #     for p in c:
    #             # print(c_string[i])
    #         if(len(p)==1):
    #             ax1.plot(p[0], p[0], color=color_bar[i], marker='o', markersize=15, alpha=0.4, mec=color_bar[i],
    #                      mew=0)
    #         else:
    #             ax1.plot(p[0], p[1], color = color_bar[i], marker='o', markersize=15, alpha=0.4, mec=color_bar[i],
    #                      mew=0)
    #     i = i + 1
    #
    # ax1.set_title('K-means')

    return means

